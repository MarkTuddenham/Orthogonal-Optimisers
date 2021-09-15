from typing import TypeVar
from typing import Type
from typing import Any

import logging

import torch

OptimType = TypeVar('U', bound=torch.optim.Optimizer)

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.NullHandler())


@torch.no_grad()
def _orth_grads(optimiser: OptimType) -> None:
    # Orthogonalise the gradients using SVD
    for group in optimiser.param_groups:
        orth = group['orth']
        for i, p in enumerate(group['params']):
            if orth and p.grad is not None and p.ndim > 1:
                G: torch.Tensor = p.grad.flatten(start_dim=1)
                try:
                    u, s, vt = torch.linalg.svd(G, full_matrices=False)
                    orth_G: torch.Tensor = u @ vt
                except RuntimeError:
                    logger.error('Failed to perform SVD, adding some noise.')
                    try:
                        u, s, v = torch.svd_lowrank(
                            G,
                            q=1,    # assume rank is at least 1
                            M=1e-4 * G.mean() * torch.randn_like(G))
                        orth_G: torch.Tensor = u @ v.T
                    except RuntimeError:
                        logger.error(('Failed to perform SVD with noise,'
                                      ' skipping gradient orthogonalisation'))
                        return
                p.grad = orth_G.reshape_as(p)


def orthogonalise(cls: Type[OptimType]) -> Type[OptimType]:
    og_init = cls.__init__
    og_step = cls.step

    def new_init(self, *args, orth=False, **kwargs):
        # Add orth hyperparam to defaults
        og_init(self, *args, **kwargs)
        self.defaults['orth'] = orth
        for grp in self.param_groups:
            grp.setdefault('orth', orth)

    def new_step(self, *args, **kwargs):
        # Orthogonalise the grads before the original optim's step method
        _orth_grads(self)
        og_step(self, *args, **kwargs)

    cls.__init__ = new_init
    cls.step = new_step
    return cls


def hook() -> None:
    from inspect import isclass
    for mod in dir(torch.optim):
        if mod.startswith('_'):
            continue
        _optim: Any = getattr(torch.optim, mod)
        if (
            isclass(_optim)
            and issubclass(_optim, torch.optim.Optimizer)
            and _optim is not torch.optim.Optimizer
        ):
            setattr(torch.optim, mod, orthogonalise(_optim))
