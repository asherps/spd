"""Weight sparsity loss for component parameters."""

from typing import Any, ClassVar, override

import torch
from jaxtyping import Float
from torch import Tensor

from spd.metrics.base import Metric
from spd.models.component_model import CIOutputs, ComponentModel


class ComponentWeightSparsityLoss(Metric):
    """L1/L2 penalty on component weight parameters to encourage sparse components.

    This loss penalizes the L-p norm of the component weight deltas (excluding bias terms),
    encouraging components to use fewer non-zero parameters.
    """

    metric_section: ClassVar[str] = "loss"

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        pnorm: float = 1.0,
    ) -> None:
        """Initialize weight sparsity loss.

        Args:
            model: ComponentModel to compute loss for
            device: Device for computation
            pnorm: P-norm to use (1.0 for L1, 2.0 for L2)
        """
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
        """Calculate weight sparsity loss across all component layers.

        This is a parameter-based loss, so batch data is unused.
        """
        total_loss = torch.tensor(0.0, device=self.device)
        n_layers = 0

        # Iterate through all component layers
        for components in self.model.components.values():
            # Each components object has U and V matrices
            # U: (C, u_dim), V: (v_dim, C)
            u_norm = torch.norm(components.U, p=self.pnorm)
            v_norm = torch.norm(components.V, p=self.pnorm)

            # Add combined norm for this layer
            total_loss = total_loss + u_norm + v_norm
            n_layers += 1

        # Average over number of layers
        if n_layers > 0:
            total_loss = total_loss / n_layers

        self.loss_value = total_loss

    @override
    def compute(self) -> Float[Tensor, ""]:
        """Return the computed loss value."""
        return self.loss_value
