from typing import TypeVar
from typing import Type
from typing import Any

import logging

import torch

OptimType = TypeVar('OptimType', bound=torch.optim.Optimizer)

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.NullHandler())


def svb(s, eps=1e-3):
    one_eps = 1 + eps
    s.clamp_(min=1 / one_eps, max=one_eps)
    return s


def _norm_p(p):
    """Normalise parameter's gradients --- Normalised SGD."""
    p.grad.div_(p.grad.norm())


def _comp_norm_p(p):
    """Normalise components's gradients."""
    G: torch.Tensor = p.grad.flatten(start_dim=1)
    norms = G.norm(dim=1)
    p.grad = G.T.div(norms).T.reshape_as(p)


def _orth_p(p):
    """Orthogonalise components's gradients."""
    g: torch.tensor = p.grad.flatten(start_dim=1)
    try:
        u, _, vt = torch.linalg.svd(g, full_matrices=False)
        # s = torch.where(s > 1e-3, s, 0.)
        # orth_g: torch.tensor = u @ torch.diag(s) @ vt
        # orth_g: torch.tensor = u @ torch.diag(svb(s)) @ vt
        orth_g: torch.tensor = u @ vt
    except RuntimeError:
        logger.error('failed to perform svd, adding some noise.')
        try:
            u, _, v = torch.svd_lowrank(
                g,
                q=1,    # assume rank is at least 1
                m=1e-4 * g.mean() * torch.randn_like(g))
            # orth_g = u @ torch.diag(svb(s)) @ v.t
            orth_g = u @ v.t
        except RuntimeError:
            logger.error(('failed to perform svd with noise,'
                          ' skipping gradient orthogonalisation'))
            return
    p.grad = orth_g.reshape_as(p)


def _svb_weight(p):
    """Orthogonalise components's gradients."""
    g: torch.tensor = p.flatten(start_dim=1)
    try:
        u, _, vt = torch.linalg.svd(g, full_matrices=False)
        # s = torch.where(s > 1e-3, s, 0.)
        # orth_g: torch.tensor = u @ torch.diag(s) @ vt
        orth_g: torch.tensor = u @ torch.diag(svb(s)) @ vt
    except RuntimeError:
        logger.error('failed to perform svd, adding some noise.')
        try:
            u, s, v = torch.svd_lowrank(
                g,
                q=1,    # assume rank is at least 1
                m=1e-4 * g.mean() * torch.randn_like(g))
            orth_g = u @ torch.diag(svb(s)) @ v.t
        except RuntimeError:
            logger.error(('failed to perform svd with noise,'
                          ' skipping gradient orthogonalisation'))
            return
    p = orth_g.reshape_as(p)


@torch.no_grad()
def _orth_grads(optimiser: OptimType) -> None:
    """Apply the specified transform to all the parameters."""
    for group in optimiser.param_groups:
        for p in group['params']:
            if p.grad is not None and p.ndim > 1:
                if group['orth']:
                    _orth_p(p)
                elif group['norm'] == 'norm':
                    _norm_p(p)
                elif group['norm'] == 'comp_norm':
                    _comp_norm_p(p)


@torch.no_grad()
def apply_svb(optimiser: OptimType) -> None:
    """Apply the specified transform to all the parameters."""
    for group in optimiser.param_groups:
        for p in group['params']:
            if p.grad is not None and p.ndim > 1:
                if group['orth']:
                    _svb_weight(p)


def orthogonalise(cls: Type[OptimType]) -> Type[OptimType]:
    og_init = cls.__init__
    og_step = cls.step

    def new_init(self, *args, orth: bool = False, norm: str = '', **kwargs):
        # Add orth hyperparam to defaults
        og_init(self, *args, **kwargs)
        self.defaults['orth'] = orth
        self.defaults['norm'] = norm
        for grp in self.param_groups:
            grp.setdefault('orth', orth)
            grp.setdefault('norm', norm)

    def new_step(self, *args, **kwargs):
        # Orthogonalise the grads before the original optim's step method
        _orth_grads(self)
        og_step(self, *args, **kwargs)

    setattr(cls, '__init__', new_init)
    setattr(cls, 'step', new_step)
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
