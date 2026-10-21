from typing import Literal

from pydantic import BaseModel

# uniform : w_t = 1 (the plain HF token-mean loss, kept for equivalence tests / ablation)
# rho     : w_t = KL_t / (H_P(t) + KL_t)              reducible fraction (the method)
# kl      : w_t = KL_t                                 ablation: reducible part only
# eg      : w_t = max(H_S(t) - H_P(t), 0)              ablation: question-level EG applied per token
# discord : w_t = sigmoid(tau * (log p_S(y_t) - log p_P(y_t)))   external baseline (DISCORD)
WeightingMode = Literal["uniform", "rho", "kl", "eg", "discord"]


class TokenWeightingConfig(BaseModel):
    mode: WeightingMode
    # Directory written by core.training.token_weighting.proxy_cache.build_proxy_cache.
    # Only read for the modes that need proxy statistics (everything but `uniform`).
    proxy_cache_path: str | None = None
    # rho_t = KL / (H_P + KL + rho_eps); the eps only matters where both terms vanish.
    rho_eps: float = 1e-3
    discord_tau: float = 1.0
    # Rescale the raw weights of the content tokens so that, per packed block, they sum to
    # the number of content tokens. Keeps the loss scale (and the effective learning rate)
    # equal to the uniform run: weighting redistributes the loss mass between tokens instead
    # of shrinking it.
    normalize_per_pack: bool = True
    # Structural tokens (<think>, </think> and everything after it: the answer letter and the
    # end-of-turn) always get this weight in every mode and stay out of the normalization.
    structural_weight: float = 1.0
    # Positions processed per chunk in the checkpointed loss; bounds the fp32 [chunk, vocab]
    # intermediates (2048 x 151936 x 4 B = 1.2 GB).
    chunk_size: int = 2048
    # Fail when a packed member has no cache entry (default). With False such members fall
    # back to weight 1 and the fraction is logged as tw/missing_frac.
    strict_cache: bool = True

    def needs_proxy(self) -> bool:
        return self.mode != "uniform"
