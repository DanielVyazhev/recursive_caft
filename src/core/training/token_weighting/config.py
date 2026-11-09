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
# shrink  : not a weight but a per-token target, q_t = (1 - a_t) onehot(y_t) + a_t p~_P: CE weight
#           1 - a_t on the trace token plus KD weight a_t towards the proxy's top-K distribution,
#           a_t = H_P / (H_P + b_t) (see the shrink_* fields)
WeightingMode = Literal[
    "uniform", "rho", "kl", "eg", "discord", "excess_loss", "student_entropy", "shrink"
]

# calibrated : a_t = H_P / (H_P + b_t)                        the method
# student    : a_t = H_S / (H_S + b_t)                        control: the student's own entropy
# const      : a_t = shrink_alpha_const                        control: the same KD mass everywhere
# bg_entropy : a_t = H_P / log |V|                             Bridge-Garden, entropy variant
# bg_conf    : a_t = 1 - max_v p_P(v)                          Bridge-Garden, confidence variant
# eopd       : a_t = 1[H_P > shrink_eopd_threshold]            EOPD's hard entropy switch, off-policy
ShrinkAlphaRule = Literal[
    "calibrated", "student", "const", "bg_entropy", "bg_conf", "eopd"
]


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
    # Multiplier on the (normalised) content-token CE weights; structural tokens keep weight 1.
    # 0 with kd_lambda > 0 is pure proxy-KD: no one-hot term on the trace tokens, CE only on the
    # structural tokens so the format is still learned.
    ce_scale: float = 1.0
    # Excess-sharpness gate: EG's one-sided condition applied to any mode. A content token keeps
    # its raw weight where the student is at least as uncertain as the proxy (H_S >= H_P); where
    # the student is already sharper than the proxy the weight is multiplied by exp(-H_P / h0),
    # i.e. kept only to the extent the proxy itself is decisive (h0 = 0: the hard EG sign gate).
    # Removes the "student confident, proxy open" corner, where KL is large but a one-hot push
    # can only sharpen the student further.
    eg_gate: bool = False
    eg_gate_h0: float = 1.0
    # Cap on the per-row normalisation factor n_content / sum(w). Once a gate or a sparse mode
    # leaves few weighted tokens, their weights stop growing past the cap and the row's loss mass
    # shrinks instead (EG's collapse came from this factor reaching the hundreds).
    max_norm_scale: float | None = None
    # Control: shuffle the raw weights among the content tokens of every batch row (seeded by the
    # global step) before normalisation. Keeps the weight distribution, destroys the assignment
    # to tokens; a weighting that only helps through its spread would survive this.
    permute_within_row: bool = False
    # mode="shrink" only. b_t is the proxy's estimated bias at the token, in nats, i.e. the price
    # of replacing the one-hot trace token by the proxy's distribution. Its base is the trace's mean
    # excess surprise of the proxy on the trace tokens, mean_t [l_P(y_t) - H_P(t)] over the content
    # tokens (floored at shrink_b_floor; shrink_b replaces it by a constant). With shrink_gate the
    # token's own excess surprise (l_P(y_t) - H_P(t) - shrink_tau)^+ is taken when larger: a token
    # the proxy did not see coming is learned from the trace, not from the proxy. The CE/KD split is
    # per token, so normalize_per_pack must be off; kd_lambda, ce_scale, eg_gate and selection do
    # not apply. permute_within_row shuffles a_t among the content tokens of a row (control).
    shrink_alpha_rule: ShrinkAlphaRule = "calibrated"
    shrink_gate: bool = True
    shrink_b: float | None = None
    shrink_b_floor: float = 0.05
    shrink_tau: float = 0.0
    shrink_alpha_const: float | None = None
    shrink_eopd_threshold: float = 0.8

    @model_validator(mode="after")
    def _validate_selection(self):
        if self.mode == "shrink":
            assert not self.normalize_per_pack, (
                "shrink sets the CE/KD split per token; set normalize_per_pack=False"
            )
            assert self.kd_lambda == 0.0 and self.ce_scale == 1.0, (
                "shrink owns the CE and KD weights; leave kd_lambda=0 and ce_scale=1"
            )
            assert not self.eg_gate and self.select_top_frac is None, (
                "eg_gate and select_top_frac do not combine with shrink"
            )
            assert self.shrink_b_floor > 0.0, "shrink_b_floor must be positive"
            assert self.shrink_b is None or self.shrink_b > 0.0, (
                "shrink_b must be positive"
            )
            assert self.shrink_tau >= 0.0, "shrink_tau must be non-negative"
            if self.shrink_alpha_rule == "const":
                assert (
                    self.shrink_alpha_const is not None
                    and 0.0 <= self.shrink_alpha_const <= 1.0
                ), "shrink_alpha_rule='const' needs shrink_alpha_const in [0, 1]"
        assert self.kd_lambda >= 0.0, "kd_lambda must be non-negative"
        assert self.ce_scale >= 0.0, "ce_scale must be non-negative"
        assert self.ce_scale > 0.0 or self.kd_lambda > 0.0, (
            "ce_scale=0 leaves no loss on content tokens; set kd_lambda"
        )
        assert self.eg_gate_h0 >= 0.0, "eg_gate_h0 must be non-negative"
        if self.max_norm_scale is not None:
            assert self.max_norm_scale >= 1.0, (
                "max_norm_scale below 1 would shrink every row"
            )
        if self.select_top_frac is not None:
            assert 0.0 < self.select_top_frac <= 1.0, (
                "select_top_frac must be in (0, 1]"
            )
        if self.mode in ("excess_loss", "student_entropy"):
            assert self.select_top_frac is not None, (
                f"{self.mode} is a selection score; set select_top_frac"
            )
        return self

    def needs_proxy(self) -> bool:
        return self.mode != "uniform" or self.kd_lambda > 0.0
