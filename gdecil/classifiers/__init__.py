"""Classifier registry: --method name -> (class, fixed keyword arguments, description)."""
import inspect

from .gaussian import DiagCov, FullCov, LedoitWolfCov, OASCov
from .mixture import MixtureFull, MixtureShared
from .shared import SharedDiagCov, SharedFullCov

METHODS = {
    'diag':            (DiagCov, {}, 'Per-class diagonal covariance (Diag-Cov)'),
    'full':            (FullCov, {}, 'Per-class full covariance (Full-Cov)'),
    'shared_diag':     (SharedDiagCov, {}, 'Diagonal covariance shared by all classes (Shared-Diag-Cov)'),
    'shared_full':     (SharedFullCov, {}, 'Full covariance shared by all classes, OAS (Shared-Full-Cov)'),
    'mog_full':        (MixtureFull, {}, 'Per-class mixture of full-covariance Gaussians (MOG4-F)'),
    'mog_shared':      (MixtureShared, {}, 'Per-class mixture of Gaussians with a shared covariance (MOG4-Shared)'),
    'oas':             (OASCov, {}, 'Per-class OAS shrinkage covariance'),
    'lw_identity':     (LedoitWolfCov, {'target': 'identity'}, 'Per-class Ledoit-Wolf shrinkage towards a scaled identity'),
    'lw_diag':         (LedoitWolfCov, {'target': 'diag'}, 'Per-class Ledoit-Wolf shrinkage towards the diagonal'),
}


def options_of(method: str) -> set:
    """Constructor options accepted by `method` (e.g. eps, max_components, use_norm_term)."""
    cls = METHODS[method][0]
    params = set()
    for klass in cls.__mro__[:-1]:   # skip object
        if '__init__' in vars(klass):
            params |= set(inspect.signature(klass.__init__).parameters)
    if not cls.uses_norm_term:
        params.discard('use_norm_term')
    return params - {'self', 'kw'} - set(METHODS[method][1])


def build(method: str, **options):
    """Instantiate `method`. Options it does not accept, or set to None, are ignored."""
    cls, fixed, _ = METHODS[method]
    accepted = options_of(method)
    return cls(**fixed, **{k: v for k, v in options.items() if k in accepted and v is not None})
