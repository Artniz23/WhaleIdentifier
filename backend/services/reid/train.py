import os
import pandas as pd
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
import argparse
from typing import Optional
import torch

from services.reid.classifier import SphereClassifier
from services.reid.config import Config, load_config
from services.reid.data_module import WhaleDataModule
from services.reid.utils import load_df


def build_callbacks(args):
    # Базовые callback'и:
    # - логирование learning rate
    # - отображение прогресс-бара
    callbacks = [
        LearningRateMonitor("epoch"),
        TQDMProgressBar(refresh_rate=10),
    ]

    # При необходимости сохраняем лучший и последний чекпоинт модели.
    if args.save_checkpoint:
        callbacks.append(
            ModelCheckpoint(
                dirpath=args.out_dir,
                monitor="train/mapNone",
                mode="max",
                save_last=True,
                save_top_k=1,
                verbose=True,
            )
        )

    return callbacks


def build_loggers(args):
    # Всегда сохраняем локальные логи обучения в CSV.
    loggers = [
        pl_loggers.CSVLogger(args.out_dir),
    ]

    # Опциональная интеграция с Weights & Biases.
    if args.wandb_logger:
        loggers.append(
            pl_loggers.WandbLogger(
                project="whale-reid",
                group=args.exp_name,
                name=f"{args.exp_name}/{args.fold}",
                save_dir=args.out_dir,
            )
        )

    return loggers


def run_inference(
        trainer,
        model,
        data_module,
        args,
        cfg,
):
    # Генерируем предсказания и эмбеддинги для всего train набора.
    model.test_results_fp = f"{args.out_dir}/train_results.npz"
    trainer.test(model, data_module.all_dataloader())

    # Загружаем отдельный тестовый набор.
    df_test = load_df(
        args.in_base_dir,
        cfg,
        args.test_csv,
    )

    test_data_module = WhaleDataModule(
        df=df_test,
        cfg=cfg,
        image_dir=f"{args.in_base_dir}/test_images",
        fold=-1,
    )

    # Генерируем предсказания и эмбеддинги для test набора.
    model.test_results_fp = f"{args.out_dir}/test_results.npz"
    trainer.test(model, test_data_module.all_dataloader())


def train(
        df: pd.DataFrame,
        args: argparse.Namespace,
        cfg: Config,
        do_inference: bool = True,
) -> Optional[float]:
    # Количество изображений каждого кита используется
    # для расчёта адаптивных ArcFace margin.
    id_class_nums = (
        df.individual_id
        .value_counts()
        .sort_index()
        .values
    )

    # Создаём модель ReID.
    model = SphereClassifier(
        cfg,
        id_class_nums=id_class_nums,
        backbone_pretrained=cfg.pretrained,
    )

    # Подготавливаем train/validation dataloader'ы.
    data_module = WhaleDataModule(
        df=df,
        cfg=cfg,
        image_dir=f"{args.in_base_dir}/train_images",
        fold=args.fold,
    )

    # Конфигурируем Lightning Trainer.
    trainer = Trainer(
        accelerator="gpu",
        max_epochs=cfg.max_epochs,
        logger=build_loggers(args),
        callbacks=build_callbacks(args),
        enable_checkpointing=args.save_checkpoint,
        precision=32,
        sync_batchnorm=True,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    # При необходимости продолжаем обучение
    # с последнего сохранённого чекпоинта.
    ckpt_path = f"{args.out_dir}/last.ckpt"

    if not os.path.exists(ckpt_path) or not args.load_snapshot:
        ckpt_path = None

    # Основной цикл обучения.
    trainer.fit(
        model,
        ckpt_path=ckpt_path,
        datamodule=data_module,
    )

    # После обучения запускаем инференс
    # и сохраняем результаты в npz-файлы.
    if do_inference:
        run_inference(
            trainer=trainer,
            model=model,
            data_module=data_module,
            args=args,
            cfg=cfg,
        )

    # Корректно завершаем W&B-сессию.
    if args.wandb_logger:
        wandb.finish()

    return None


def parse_args():
    # Аргументы командной строки для запуска обучения.
    parser = argparse.ArgumentParser()

    # Конфиг эксперимента.
    parser.add_argument(
        "--config-path",
        required=True,
    )

    # Базовый конфиг со значениями по умолчанию.
    parser.add_argument(
        "--default-config-path",
        default="reid_config/default.yaml",
    )

    # Корневая директория датасета.
    parser.add_argument(
        "--in-base-dir",
        required=True,
    )

    # Корневая директория результатов.
    parser.add_argument(
        "--out-base-dir",
        required=True,
    )

    # Название эксперимента.
    parser.add_argument(
        "--exp-name",
        required=True,
    )

    # Номер fold для кросс-валидации.
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
    )

    # CSV-файл тестового набора.
    parser.add_argument(
        "--test-csv",
        default="test.csv",
    )

    # Продолжить обучение с последнего чекпоинта.
    parser.add_argument(
        "--load-snapshot",
        action="store_true",
    )

    # Сохранять чекпоинты модели.
    parser.add_argument(
        "--save-checkpoint",
        action="store_true",
    )

    # Включить логирование в Weights & Biases.
    parser.add_argument(
        "--wandb-logger",
        action="store_true",
    )

    return parser.parse_args()


def main():
    # Более быстрые float32 матричные операции на современных GPU.
    torch.set_float32_matmul_precision("medium")

    args = parse_args()

    # Загружаем конфигурацию обучения.
    cfg = load_config(
        args.config_path,
        args.default_config_path,
    )

    # Загружаем train-разметку.
    df = load_df(
        args.in_base_dir,
        cfg,
        "train.csv",
    )

    # Формируем директорию конкретного запуска.
    args.out_dir = (
        f"{args.out_base_dir}/"
        f"{args.exp_name}/"
        f"{args.fold}"
    )

    # Если используется полный датасет без CV,
    # работаем в специальном режиме fold=-1.
    fold = -1 if cfg.n_splits == -1 else args.fold
    args.fold = fold

    # Запускаем обучение.
    train(
        df=df,
        args=args,
        cfg=cfg,
        do_inference=True,
    )


if __name__ == "__main__":
    main()
