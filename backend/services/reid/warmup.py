import math

class WarmupCosineLambda:
    """
    Scheduler-функция для LambdaLR.

    Состоит из двух фаз:
    1. Warmup — плавное увеличение learning rate.
    2. Cosine decay — плавное уменьшение learning rate по косинусной кривой.
    """

    def __init__(self, warmup_steps: int, cycle_steps: int, decay_scale: float, exponential_warmup: bool = False):
        # Количество эпох для warmup-фазы.
        self.warmup_steps = warmup_steps
        # Количество эпох для cosine decay.
        self.cycle_steps = cycle_steps
        # Минимальный множитель learning rate.
        # Например 0.1 означает снижение LR до 10% от начального.
        self.decay_scale = decay_scale
        # Использовать экспоненциальный warmup вместо линейного.
        self.exponential_warmup = exponential_warmup

    def __call__(self, epoch: int):
        """
        Возвращает коэффициент масштабирования learning rate
        для текущей эпохи.
        """

        # Фаза warmup.
        if epoch < self.warmup_steps:
            # Экспоненциальный рост learning rate.
            if self.exponential_warmup:
                return self.decay_scale * pow(self.decay_scale, -epoch / self.warmup_steps)
            # Линейный рост learning rate от decay_scale до 1.0.
            ratio = epoch / self.warmup_steps
        else:
            # Cosine annealing после завершения warmup.
            ratio = (1 + math.cos(math.pi * (epoch - self.warmup_steps) / self.cycle_steps)) / 2
        # Масштабируем коэффициент в диапазон:
        # [decay_scale, 1.0]
        return self.decay_scale + (1 - self.decay_scale) * ratio