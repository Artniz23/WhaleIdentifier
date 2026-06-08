from typing import Dict, Tuple
import timm
from pytorch_lightning import LightningModule
import numpy as np
import torch

from services.reid.arc import GeM, ArcMarginProductSubcenter, ArcFaceLossAdaptiveMargin
from services.reid.config import Config
from services.reid.utils import map_dict
from services.reid.warmup import WarmupCosineLambda


class SphereClassifier(LightningModule):
    """
    Модель для whale re-identification.

    Архитектура:
    backbone (timm)
        ->
    GeM pooling по нескольким feature maps
        ->
    feature concatenation
        ->
    normalization neck
        ->
    ArcFace SubCenter classifier

    Во время инференса используется только get_feat(),
    который возвращает L2-нормализованные эмбеддинги.
    """

    def __init__(self, cfg: dict, id_class_nums=None, backbone_pretrained: bool | None = None):
        super().__init__()
        if not isinstance(cfg, Config):
            cfg = Config(cfg)
        self.save_hyperparameters(cfg, ignore=["id_class_nums"])
        self.test_results_fp = None

        # Позволяет принудительно переопределить использование
        # pretrained весов при загрузке модели.
        pretrained = cfg.pretrained if backbone_pretrained is None else backbone_pretrained

        # Извлекаем промежуточные feature maps из backbone.
        # Используем несколько уровней признаков вместо одного.
        self.backbone = timm.create_model(
            cfg.model_name,
            in_chans=3,
            pretrained=pretrained,
            num_classes=0,
            features_only=True,
            out_indices=cfg.out_indices,
        )
        # Размерности feature maps выбранных уровней backbone.
        feature_dims = self.backbone.feature_info.channels()
        print(f"feature dims: {feature_dims}")
        # Отдельный GeM pooling для каждого уровня признаков.
        # После pooling все признаки будут объединены в один вектор.
        self.global_pools = torch.nn.ModuleList(
            [GeM(p=cfg.global_pool.p, requires_grad=cfg.global_pool.train) for _ in cfg.out_indices]
        )
        # Итоговая размерность эмбеддинга после объединения всех уровней.
        self.mid_features = np.sum(feature_dims)
        # Нормализация эмбеддингов перед ArcFace.
        # Обычно улучшает стабильность обучения и качество retrieval.
        if cfg.normalization == "batchnorm":
            self.neck = torch.nn.BatchNorm1d(self.mid_features)
        elif cfg.normalization == "layernorm":
            self.neck = torch.nn.LayerNorm(self.mid_features)
        # ArcFace-классификатор с несколькими центрами на класс.
        # Используется только на этапе обучения.
        self.head_id = ArcMarginProductSubcenter(self.mid_features, cfg.num_classes, cfg.n_center_id)
        # Если известна статистика количества примеров по классам,
        # строим индивидуальный margin для каждого класса.
        #
        # Обычно помогает при сильном дисбалансе датасета.
        if id_class_nums is not None:
            margins_id = np.power(id_class_nums, cfg.margin_power_id) * cfg.margin_coef_id + cfg.margin_cons_id
            print("margins_id", margins_id)
            self.margin_fn_id = ArcFaceLossAdaptiveMargin(margins_id, cfg.num_classes, cfg.s_id)
            self.loss_fn_id = torch.nn.CrossEntropyLoss()

    def get_feat(self, x: torch.Tensor) -> torch.Tensor:
        # Извлекаем feature maps разных уровней backbone.
        ms = self.backbone(x)
        # Применяем GeM pooling к каждому уровню
        # и объединяем признаки в единый embedding.
        h = torch.cat([global_pool(m) for m, global_pool in zip(ms, self.global_pools)], dim=1)
        # Нормализация признаков перед классификационной головой.
        return self.neck(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Во время обучения возвращаем ArcFace logits.
        # Для retrieval обычно используется get_feat().
        feat = self.get_feat(x)
        return self.head_id(feat)

    def training_step(self, batch, batch_idx):
        # Изображения и идентификаторы китов.
        x, ids = batch["image"], batch["label"]
        # Косинусные логиты ArcFace.
        logits_ids = self(x)
        # Добавляем адаптивный angular margin.
        margin_logits_ids = self.margin_fn_id(logits_ids, ids)
        # Основной классификационный loss.
        loss_ids = self.loss_fn_id(margin_logits_ids, ids)
        # Логируем retrieval/classification метрики.
        self.log_dict({"train/loss_ids": loss_ids.detach()}, on_step=False, on_epoch=True)
        with torch.no_grad():
            self.log_dict(map_dict(logits_ids, ids, "train"), on_step=False, on_epoch=True)
        # Итоговый loss с возможностью масштабирования.
        return loss_ids * self.hparams.loss_id_ratio

    def validation_step(self, batch, batch_idx):
        # Test-Time Augmentation:
        # усредняем предсказания оригинального изображения
        # и его горизонтального отражения.
        x, ids = batch["image"], batch["label"]
        out1 = self(x)
        out2 = self(x.flip(3))
        output = (out1 + out2) / 2
        self.log_dict(map_dict(output, ids, "val"), on_step=False, on_epoch=True)

    def configure_optimizers(self):
        # Backbone обычно обучаем с меньшим learning rate,
        # чем новую классификационную голову.
        backbone_params = list(self.backbone.parameters()) + list(self.global_pools.parameters())
        head_params = (
                list(self.neck.parameters()) + list(self.head_id.parameters())
        )
        params = [
            {"params": backbone_params, "lr": self.hparams.lr_backbone},
            {"params": head_params, "lr": self.hparams.lr_head},
        ]
        # Поддерживаются несколько оптимизаторов.
        if self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(params)
        elif self.hparams.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(params)
        elif self.hparams.optimizer == "RAdam":
            optimizer = torch.optim.RAdam(params)

        # Warmup + Cosine Decay scheduler.
        #
        # Сначала плавный разогрев learning rate,
        # затем косинусное уменьшение.
        warmup_steps = self.hparams.max_epochs * self.hparams.warmup_steps_ratio
        cycle_steps = self.hparams.max_epochs - warmup_steps
        lr_lambda = WarmupCosineLambda(warmup_steps, cycle_steps, self.hparams.lr_decay_scale)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "frequency": 1}]

    def on_test_start(self) -> None:
        # Буферы для накопления результатов тестирования.
        #
        # Собираем данные локально на каждом GPU,
        # затем объединяем их через all_gather().
        self._test_outputs = {
            "original_index": [],
            "label": [],
            "pred_logit": [],
            "pred_idx": [],
            "embed_features1": [],
            "embed_features2": [],
        }

    def test_step(self, batch, batch_idx):
        """Собираем результаты в локальные буферы (тензоры на устройстве)."""
        x = batch["image"]
        # Извлекаем признаки для оригинального изображения.
        feat1 = self.get_feat(x)
        out1 = self.head_id(feat1)
        # Извлекаем признаки для горизонтального отражения.
        feat2 = self.get_feat(x.flip(3))
        out2 = self.head_id(feat2)

        # TTA-усреднение логитов.
        logits = (out1 + out2) / 2.0
        # Сортируем классы по вероятности.
        pred_logit, pred_idx = logits.sort(descending=True)

        # Ограничиваем количество сохраняемых кандидатов.
        # Это уменьшает размер итогового файла результатов.
        top_k = min(pred_logit.shape[1], 1000)
        pred_logit = pred_logit[:, :top_k]
        pred_idx = pred_idx[:, :top_k]

        # приводим original_index и label к тензорам на устройстве
        def to_device_tensor(v):
            if isinstance(v, torch.Tensor):
                return v.to(self.device)
            else:
                return torch.as_tensor(v, device=self.device)

        oidx = to_device_tensor(batch["original_index"])
        lab = to_device_tensor(batch["label"])

        # Сохраняем результаты в локальные буферы.
        # Сами тензоры остаются на устройстве до конца эпохи.
        self._test_outputs["original_index"].append(oidx)
        self._test_outputs["label"].append(lab)
        self._test_outputs["pred_logit"].append(pred_logit.to(self.device))
        self._test_outputs["pred_idx"].append(pred_idx.to(self.device))
        self._test_outputs["embed_features1"].append(feat1.to(self.device))
        self._test_outputs["embed_features2"].append(feat2.to(self.device))

        # вернём короткий словарь
        return {
            "original_index": oidx,
            "label": lab,
            "pred_logit": pred_logit,
            "pred_idx": pred_idx,
            "embed_features1": feat1,
            "embed_features2": feat2,
        }

    def on_test_epoch_end(self) -> None:
        # Объединяем результаты всех батчей данного процесса.
        if not hasattr(self, "_test_outputs") or len(self._test_outputs["original_index"]) == 0:
            return

        # локально конкатенируем списки в тензоры
        local_cat: Dict[str, torch.Tensor] = {}
        for k, v in self._test_outputs.items():
            if len(v) == 0:
                local_cat[k] = torch.tensor([], device=self.device)
            else:
                local_cat[k] = torch.cat(v, dim=0)

        # all_gather каждого тензора (возвращаемое значение — тензор, содержащий данные со всех рангов)
        gathered: Dict[str, torch.Tensor] = {}
        for k, tensor in local_cat.items():
            if tensor.numel() == 0:
                gathered[k] = torch.tensor([], device="cpu")
            else:
                g = self.all_gather(tensor)  # PL конкатенирует по dim=0
                gathered[k] = g

        # сохраняем только на глобальном ранге 0
        global_rank = getattr(self.trainer, "global_rank", 0)
        if global_rank == 0:
            epoch_results: Dict[str, np.ndarray] = {}
            for k, tensor in gathered.items():
                if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
                    epoch_results[k] = tensor.detach().cpu().numpy()
                else:
                    epoch_results[k] = np.array([])

            out_fp = self.test_results_fp or "test_results.npz"
            np.savez_compressed(out_fp, **epoch_results)
            # опционально: логируем факт сохранения
            if hasattr(self, "log"):
                try:
                    self.log("test/saved", 1.0)
                except Exception:
                    pass

        # очищаем буферы
        self._test_outputs = {k: [] for k in self._test_outputs.keys()}
