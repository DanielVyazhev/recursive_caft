"""Token-level loss weighting from a frozen proxy model.

Per-question entropy gain EG(x) = max{H_S - H_P, 0} (distillation_by_metrics) carried to the
token level and turned into a reducible-fraction weight:

    KL_t  = KL(p_P(. | x_<t) || p_S(. | x_<t))        reducible (epistemic) part relative to the proxy
    H_P(t)                                             irreducible (aleatoric) part relative to the proxy
    rho_t = KL_t / (H_P(t) + KL_t)  in [0, 1]          share of the student's excess that is reducible

The proxy is run once, teacher-forced over the traces (proxy_cache); the student side is
recomputed from the training forward pass at every step (trainer). Modes `kl`, `eg` and
`discord` are ablations / the external baseline on the same machinery.
"""

from core.training.token_weighting.config import TokenWeightingConfig, WeightingMode

__all__ = ["TokenWeightingConfig", "WeightingMode"]
