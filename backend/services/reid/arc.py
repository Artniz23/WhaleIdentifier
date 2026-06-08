import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch


class GeM(nn.Module):
    """
    Generalized Mean Pooling.

    Используется вместо обычного Average Pooling для получения
    более информативного глобального признака из feature map.

    При:
    - p = 1 - работает как Average Pooling
    - p -> ∞ - приближается к Max Pooling

    Часто используется в задачах retrieval и re-identification,
    так как обычно даёт более качественные эмбеддинги.
    """

    def __init__(self, p=3, eps=1e-6, requires_grad=False):
        super().__init__()
        # Параметр степени усреднения.
        # Может быть фиксированным или обучаемым.
        self.p = nn.Parameter(torch.ones(1) * p, requires_grad=requires_grad)
        # Защита от численных проблем.
        self.eps = eps

    def forward(self, x: torch.Tensor):
        # GeM pooling:
        # (mean(x^p))^(1/p)
        return x.clamp(min=self.eps).pow(self.p).mean((-2, -1)).pow(1.0 / self.p)


class ArcMarginProductSubcenter(nn.Module):
    """
    ArcFace-классификатор с несколькими центрами (sub-centers) на класс.

    Вместо одного эталонного центра для каждого класса
    хранится k центров.

    Это помогает:
    - лучше моделировать внутриклассовое разнообразие;
    - быть устойчивее к шумным данным;
    - улучшать качество ReID и retrieval моделей.
    """

    def __init__(self, in_features: int, out_features: int, k: int = 3):
        super().__init__()
        # Матрица центров:
        # [num_classes * k, embedding_dim]
        self.weight = nn.Parameter(torch.FloatTensor(out_features * k, in_features))
        self.reset_parameters()
        # Количество центров на класс.
        self.k = k
        # Количество классов.
        self.out_features = out_features

    def reset_parameters(self):
        """
        Инициализация центров случайными значениями.
        """
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Возвращает cosine similarity между эмбеддингом
        и ближайшим sub-center каждого класса.
        """
        # Косинусное сходство со всеми центрами.
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        # Преобразуем:
        # [batch, classes * k]
        # ->
        # [batch, classes, k]
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        # Для каждого класса выбираем
        # наиболее похожий центр.
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine


class ArcFaceLossAdaptiveMargin(nn.modules.Module):
    """
    ArcFace с адаптивным margin для каждого класса.

    Вместо одного общего margin используется отдельный margin
    для каждого класса.

    Обычно применяется при сильном дисбалансе классов:
    - редким классам можно дать больший margin;
    - частым классам меньший.

    Это помогает получить более качественные эмбеддинги.
    """

    def __init__(self, margins: np.ndarray, n_classes: int, s: float = 30.0):
        super().__init__()
        # Масштаб ArcFace logits.
        self.s = s
        # Margin для каждого класса.
        self.margins = margins
        # Количество классов.
        self.out_dim = n_classes

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Применяет ArcFace margin только
        к истинному классу каждого объекта.
        """

        # Margin соответствующего класса.
        ms = self.margins[labels.cpu().numpy()]
        # Предвычисляем тригонометрические коэффициенты.
        cos_m = torch.from_numpy(np.cos(ms)).float().cuda()
        sin_m = torch.from_numpy(np.sin(ms)).float().cuda()
        th = torch.from_numpy(np.cos(math.pi - ms)).float().cuda()
        mm = torch.from_numpy(np.sin(math.pi - ms) * ms).float().cuda()
        # One-hot представление классов.
        labels = F.one_hot(labels, self.out_dim).float()
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        # Численно стабильная версия ArcFace.
        phi = cosine * cos_m.view(-1, 1) - sine * sin_m.view(-1, 1)
        phi = torch.where(cosine > th.view(-1, 1), phi, cosine - mm.view(-1, 1))
        # Margin применяется только к правильному классу
        # Масштабирование логитов.
        return ((labels * phi) + ((1.0 - labels) * cosine)) * self.s
