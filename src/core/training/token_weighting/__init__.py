"""Token-level loss weighting from a frozen proxy model.

Per-question entropy gain EG(x) = max{H_S - H_P, 0} (distillation_by_metrics) carried to the
token level and turned into a reducible-fraction weight:

    KL_t  = KL(p_P(. | x_<t) || p_S(. | x_<t))        reducible (epistemic) part relative to the proxy
    H_P(t)                                             irreducible (aleatoric) part relative to the proxy
    rho_t = KL_t / (H_P(t) + KL_t)  in [0, 1]          share of the student's excess that is reducible

The proxy is run once, teacher-forced over the traces (proxy_cache); the student side is
recomputed from the training forward pass at every step (trainer). Modes `kl`, `eg` and
`discord` are ablations / the external baseline on the same machinery.

Mode `shrink` changes the target instead of the weight: on every content token the one-hot
trace token is mixed with the proxy's distribution, q_t = (1 - a_t) onehot(y_t) + a_t p~_P,
a_t = H_P / (H_P + b_t), where b_t is the proxy's estimated bias at the token (its excess
surprise on the trace, per trace and, gated, per token). Where the proxy is certain the
target is the trace token; where the position is open and the proxy expected the trace token
the student learns the spread; where the trace did something the proxy did not expect the
token is learned from the trace.
"""

from core.training.token_weighting.config import (
    ShrinkAlphaRule,
    TokenWeightingConfig,
    WeightingMode,
)

__all__ = ["ShrinkAlphaRule", "TokenWeightingConfig", "WeightingMode"]
