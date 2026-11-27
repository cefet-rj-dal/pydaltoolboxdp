"""Shared PyTorch training primitives used by Deep ++ models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import ttest_ind
from torch import nn

VALIDATION_STRATEGIES = {"static", "dynamic"}
STOPPING_RULES = {"none", "patience", "sma", "ema", "h"}


@dataclass
class TrainingConfig:
    batch_size: int = 1
    epochs: int = 100
    learning_rate: float = 0.001
    validation_strategy: str = "static"
    stopping_rule: str = "none"
    val_ratio: float = 0.33
    patience: int = 3
    min_delta: float = 0.0
    sma_window: int = 30
    ema_alpha: float = 0.2
    test_window: int = 30
    p_value: float = 0.05
    reshuffle_freq: int = 1


def int_list(
    values,
    default: Sequence[int] | None = None,
    *,
    allow_empty: bool = False,
) -> list[int]:
    """Normalize an integer or integer sequence into a list."""
    if values is None:
        values = default if default is not None else []
    if isinstance(values, (int, np.integer)):
        values = [int(values)]
    result = [int(value) for value in values]
    if not allow_empty and not result:
        raise ValueError("at least one integer value is required")
    if any(value <= 0 for value in result):
        raise ValueError("layer sizes must be positive integers")
    return result


def activation(name: str, negative_slope: float = 0.01) -> nn.Module:
    """Build one of the activation modules supported by the R interface."""
    name = str(name).lower()
    modules = {
        "relu": lambda: nn.ReLU(inplace=True),
        "leaky_relu": lambda: nn.LeakyReLU(float(negative_slope), inplace=True),
        "elu": lambda: nn.ELU(inplace=True),
        "gelu": nn.GELU,
        "selu": lambda: nn.SELU(inplace=True),
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "softplus": nn.Softplus,
        "identity": nn.Identity,
        "none": nn.Identity,
    }
    try:
        return modules[name]()
    except KeyError as error:
        raise ValueError(f"unsupported activation: {name}") from error


def dense_stack(
    input_dim: int,
    hidden_sizes,
    output_dim: int,
    *,
    hidden_activation: str = "relu",
    output_activation: str = "none",
    negative_slope: float = 0.01,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = int(input_dim)
    for hidden_size in int_list(hidden_sizes, allow_empty=True):
        layers.append(nn.Linear(previous, hidden_size))
        layers.append(activation(hidden_activation, negative_slope))
        previous = hidden_size
    layers.append(nn.Linear(previous, int(output_dim)))
    if output_activation.lower() not in {"none", "identity"}:
        layers.append(activation(output_activation, negative_slope))
    return nn.Sequential(*layers)


def validate_training_options(validation_strategy: str, stopping_rule: str) -> tuple[str, str]:
    validation_strategy = str(validation_strategy).lower()
    stopping_rule = str(stopping_rule).lower()
    if validation_strategy not in VALIDATION_STRATEGIES:
        raise ValueError(
            f"validation_strategy must be one of {sorted(VALIDATION_STRATEGIES)}"
        )
    if stopping_rule not in STOPPING_RULES:
        raise ValueError(f"stopping_rule must be one of {sorted(STOPPING_RULES)}")
    return validation_strategy, stopping_rule


class StopController:
    """Apply the stopping rules exposed by the original Deep ++ R API."""

    def __init__(self, config: TrainingConfig) -> None:
        self.rule = config.stopping_rule
        self.min_delta = float(config.min_delta)
        self.patience = int(config.patience)
        self.sma_window = max(1, int(config.sma_window))
        self.ema_alpha = float(config.ema_alpha)
        self.test_window = max(2, int(config.test_window))
        self.p_value = float(config.p_value)
        self.best_value = float("inf")
        self.best_state = None
        self.patience_counter = 0
        self.ema_value = None
        self.history: list[float] = []

    @staticmethod
    def _clone_state(model: nn.Module) -> dict:
        return {key: value.detach().clone() for key, value in model.state_dict().items()}

    def _h_improved(self) -> bool:
        if len(self.history) < 2 * self.test_window:
            return len(self.history) == 1 or self.history[-1] < min(self.history[:-1])
        previous = self.history[-2 * self.test_window : -self.test_window]
        recent = self.history[-self.test_window :]
        _, p_value = ttest_ind(previous, recent, equal_var=False, alternative="greater")
        return bool(p_value < self.p_value)

    def step(self, model: nn.Module, current: float) -> bool:
        self.history.append(float(current))
        if self.rule == "none":
            self.best_state = self._clone_state(model)
            return False
        if self.rule == "h":
            improved = self._h_improved()
            if improved:
                self.best_state = self._clone_state(model)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            return self.patience_counter > self.patience

        if self.rule == "patience":
            monitored = float(current)
        elif self.rule == "sma":
            monitored = float(np.mean(self.history[-self.sma_window :]))
        elif self.rule == "ema":
            if self.ema_value is None:
                self.ema_value = float(current)
            else:
                self.ema_value = (
                    self.ema_alpha * float(current)
                    + (1.0 - self.ema_alpha) * self.ema_value
                )
            monitored = self.ema_value
        else:  # validated at construction; defensive for mutated configs
            raise ValueError(f"unsupported stopping rule: {self.rule}")

        if self.best_value - monitored > self.min_delta:
            self.best_value = monitored
            self.best_state = self._clone_state(model)
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        return self.patience_counter > self.patience
