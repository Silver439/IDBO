import torch
import botorch
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.objective import PosteriorTransform
from botorch.utils.transforms import t_batch_mode_transform
from botorch.models.model import Model
from torch import Tensor
from typing import Union, Optional
from botorch.utils.probability.utils import (
    log_ndtr as log_Phi,
    log_phi,
    ndtr as Phi,
    phi,
)
import warnings
warnings.filterwarnings("ignore")
torch.set_default_tensor_type(torch.DoubleTensor)

class AugmentedExpectedImprovement(AnalyticAcquisitionFunction):

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
    ):
        r"""Single-outcome Expected Improvement (analytic).

        Args:
            model: A fitted single-outcome model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        #legacy_ei_numerics_warning(legacy_name=type(self).__name__)
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Expected Improvement on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.
                Expected Improvement is computed for each point individually,
                i.e., what is considered are the marginal posteriors, not the
                joint.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Expected Improvement values at the
            given design points `X`.
        """
        mean, sigma = self._mean_and_sigma(X)
        u = _scaled_improvement(mean, sigma, self.best_f, self.maximize)
        return sigma * _ei_helper(u) * (1 - 0.5/(torch.sqrt(sigma*sigma+0.25)))
    

def _scaled_improvement(
mean: Tensor, sigma: Tensor, best_f: Tensor, maximize: bool
) -> Tensor:
    """Returns `u = (mean - best_f) / sigma`, -u if maximize == True."""
    u = (mean - best_f) / sigma
    return u if maximize else -u

def _ei_helper(u: Tensor) -> Tensor:
    """Computes phi(u) + u * Phi(u), where phi and Phi are the standard normal
    pdf and cdf, respectively. This is used to compute Expected Improvement.
    """
    return phi(u) + u * Phi(u)