from typing import Literal

from pydantic import BaseModel, model_validator

# uniform : w_t = 1 (the plain HF token-mean loss, kept for equivalence tests / ablation)
# rho     : w_t = KL_t / (H_P(t) + KL_t)              reducible fraction (the method)
# kl      : w_t = KL_t                                 ablation: reducible part only
# eg      : w_t = max(H_S(t) - H_P(t), 0)              ablation: question-level EG applied per token
# discord : w_t = sigmoid(tau * (log p_S(y_t) - log p_P(y_t)))   external baseline (DISCORD)
# excess_loss : score l_S - l_P (can be negative) -- Rho-1 / Token Cleaning; only meaningful as a
#           selection score, i.e. together with select_top_frac
# student_entropy : score H_S(t) -- SE-KD / the "80/20 rule"; selection score (top 20%)
WeightingMode = Literal["uniform", "rho", "kl", "eg", "discord", "excess_loss", "student_entropy"]


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
    # Hard selection instead of soft weights: per batch row, keep the content tokens with the
    # top `select_top_frac` share of the mode's raw score (weight 1, the rest 0). With the
    # per-pack normalisation this is the mean loss over the selected tokens, the Rho-1 /
    # Token Cleaning objective (Rho-1 keeps 60%).
    select_top_frac: float | None = None
    # Proxy-KD term: adds kd_lambda * KL(p~_P || p~_S) per content token, both distributions
    # renormalised on the proxy's cached top-K ids (DistillKit / EOPD style top-k forward KL).
    # With mode="uniform" this is plain SFT + KD towards the proxy, the "why not distil from
    # the proxy" baseline; it combines with any weighting mode.
    kd_lambda: float = 0.0
    # Control: shuffle the raw weights among the content tokens of every batch row (seeded by the
    # global step) before normalisation. Keeps the weight distribution, destroys the assignment
    # to tokens; a weighting that only helps through its spread would survive this.
    permute_within_row: bool = False

    @model_validator(mode="after")
    def _validate_selection(self):
        assert self.kd_lambda >= 0.0, "kd_lambda must be non-negative"
        if self.select_top_frac is not None:
            assert 0.0 < self.select_top_frac <= 1.0, "select_top_frac must be in (0, 1]"
        if self.mode in ("excess_loss", "student_entropy"):
            assert self.select_top_frac is not None, f"{self.mode} is a selection score; set select_top_frac"
        return self

    def needs_proxy(self) -> bool:
        return self.mode != "uniform" or self.kd_lambda > 0.0
