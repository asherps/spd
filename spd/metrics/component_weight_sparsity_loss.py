"""Weight sparsity loss for component parameters."""

from typing import Any, ClassVar, override

import torch
from jaxtyping import Float
from torch import Tensor

from spd.metrics.base import Metric
from spd.models.component_model import CIOutputs, ComponentModel


class ComponentWeightSparsityLoss(Metric):
    """L-p norm penalty on component weights to encourage sparse components."""

    metric_section: ClassVar[str] = "loss"

    def __init__(self, model: ComponentModel, device: str, pnorm: float = 1.0) -> None:
        self.model = model
        self.device = device
        self.pnorm = pnorm
        self.loss_value = torch.tensor(0.0, device=device)

    @override
    def update(
        self,
        *,
        batch: Any,
        target_out: Any,
        pre_weight_acts: Any,
        ci: CIOutputs,
        current_frac_of_training: float,
        weight_deltas: Any,
        **_: Any,
    ) -> None:
        total = torch.tensor(0.0, device=self.device)
        for components in self.model.components.values():
            total = (
                total
                + torch.norm(components.U, p=self.pnorm)
                + torch.norm(components.V, p=self.pnorm)
            )
        self.loss_value = total / len(self.model.components) if self.model.components else total

    @override
    def compute(self) -> Float[Tensor, ""]:
        return self.loss_value
