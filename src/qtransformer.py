"""Q-Transformer architecture (phase 7).

Causal transformer over an interleaved sequence of pre-action and post-action
tokens per PA, with factored autoregressive Q-heads ``(type → x → z)`` and a
single value head ``V(s)`` for IQL.

No training loop here — that's phase 8. This module is everything needed to
go from a :class:`PABatch` through the model to ``Q``, ``V``, action logits,
and a deterministic ``policy()`` for inference.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset import PABatch
from src.encoder import EncoderConfig, PostActionEncoder, PreActionEncoder, _StandardizeFloat


@dataclass
class QTransformerConfig:
    """Hyperparameters for the transformer body + heads."""
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1536
    dropout: float = 0.1
    n_x_bins: int = 20
    n_z_bins: int = 20
    # Max interleaved sequence length per PA = 2 * max_PA_pitches. The Statcast
    # 2024 max PA was 16 pitches, so 32 covers it with margin. Used to size the
    # learned positional embedding table.
    max_seq_len: int = 64


# --------------------------------------------------------------------------- #
# Q-Transformer
# --------------------------------------------------------------------------- #


class QTransformer(nn.Module):
    """The model.

    Sequence layout (per PA, length 2T):
        [pre_0, post_0, pre_1, post_1, ..., pre_{T-1}, post_{T-1}]

    With a causal mask, the encoded vector at the ``pre_i`` position attends to
    every prior pitch's full (pre, post) tokens but to nothing of pitch i's own
    action/outcome — exactly what we want for Q(s_i, a_i) prediction.
    """

    def __init__(
        self,
        vocab_sizes: dict[str, int],
        cfg: QTransformerConfig | None = None,
        encoder_cfg: EncoderConfig | None = None,
        pre_cont_stats: tuple[torch.Tensor, torch.Tensor] | None = None,
        profile_stats: tuple[torch.Tensor, torch.Tensor] | None = None,
        post_cont_stats: tuple[torch.Tensor, torch.Tensor] | None = None,
        arsenal_head_stats: tuple[torch.Tensor, torch.Tensor] | None = None,
        batter_per_type_head_stats: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        super().__init__()
        self.cfg = cfg or QTransformerConfig()
        self.enc_cfg = encoder_cfg or EncoderConfig(
            d_model=self.cfg.d_model, dropout=self.cfg.dropout
        )

        # Action-vocab embedding tables — single source of truth, shared between the
        # post-action encoder (so past pitches' actions look the same as Q-head action
        # conditioning) and the Q heads themselves.
        c = self.enc_cfg
        self.emb_pitch_type_action = nn.Embedding(vocab_sizes["pitch_type"], c.d_pitch_type_emb)
        self.emb_x_bin_action = nn.Embedding(self.cfg.n_x_bins, c.d_action_loc_emb)
        self.emb_z_bin_action = nn.Embedding(self.cfg.n_z_bins, c.d_action_loc_emb)

        action_emb_modules = {
            "pitch_type": self.emb_pitch_type_action,
            "x_bin": self.emb_x_bin_action,
            "z_bin": self.emb_z_bin_action,
        }

        self.pre_encoder = PreActionEncoder(
            vocab_sizes, cfg=self.enc_cfg,
            cont_stats=pre_cont_stats,
            profile_stats=profile_stats,
        )
        self.post_encoder = PostActionEncoder(
            vocab_sizes,
            n_x_bins=self.cfg.n_x_bins,
            n_z_bins=self.cfg.n_z_bins,
            cfg=self.enc_cfg,
            action_emb_modules=action_emb_modules,
            cont_stats=post_cont_stats,
        )

        # Learned position embeddings for pre vs post role within each step pair.
        self.role_emb = nn.Embedding(2, self.cfg.d_model)  # 0=pre, 1=post
        # Learned absolute position embeddings over the interleaved sequence.
        # Without this, the transformer can only infer step number from content
        # (pitch_idx_in_pa scalar) which is much weaker.
        self.pos_emb = nn.Embedding(self.cfg.max_seq_len, self.cfg.d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.d_model,
            nhead=self.cfg.n_heads,
            dim_feedforward=self.cfg.d_ff,
            dropout=self.cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=self.cfg.n_layers)
        self.norm_out = nn.LayerNorm(self.cfg.d_model)

        # Per-(pitcher, pitch_type) and per-(batter, pitch_type) feature
        # standardization buffers — applied at head time before concatenation.
        from src.tokenize import ARSENAL_HEAD_FIELDS, BATTER_PER_TYPE_HEAD_FIELDS
        n_arsenal = len(ARSENAL_HEAD_FIELDS)
        n_batter_pt = len(BATTER_PER_TYPE_HEAD_FIELDS)
        self.arsenal_norm = _StandardizeFloat(
            *(arsenal_head_stats or (torch.zeros(n_arsenal), torch.ones(n_arsenal)))
        )
        self.batter_pt_norm = _StandardizeFloat(
            *(batter_per_type_head_stats or (torch.zeros(n_batter_pt), torch.ones(n_batter_pt)))
        )
        self.n_arsenal = n_arsenal
        self.n_batter_pt = n_batter_pt

        # Factored autoregressive Q heads.
        d_h = self.cfg.d_model
        d_pt = c.d_pitch_type_emb
        d_loc = c.d_action_loc_emb

        def _mlp_head(in_dim: int, out_dim: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(in_dim, d_h),
                nn.GELU(),
                nn.Dropout(self.cfg.dropout),
                nn.Linear(d_h, out_dim),
            )

        # Type head is now per-type aware: input includes the (pitcher, type) arsenal
        # features and the (batter, type) profile features for each candidate type.
        # We broadcast h_pre over the type dim and produce one scalar per type.
        self.q_head_type = _mlp_head(d_h + n_arsenal + n_batter_pt, 1)
        # X / Z heads condition on the chosen type, so they get arsenal + batter for that type only.
        self.q_head_x = _mlp_head(d_h + d_pt + n_arsenal + n_batter_pt, self.cfg.n_x_bins)
        self.q_head_z = _mlp_head(d_h + d_pt + d_loc + n_arsenal + n_batter_pt, self.cfg.n_z_bins)

        # IQL value head — state only, no per-type conditioning.
        self.v_head = _mlp_head(d_h, 1)

    # --------------------------------------------------------------------- #
    # Sequence build + transformer
    # --------------------------------------------------------------------- #

    def _interleave(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        """Stack [pre_0, post_0, pre_1, post_1, ...] along time dim."""
        B, T, D = pre.shape
        out = torch.empty(B, 2 * T, D, dtype=pre.dtype, device=pre.device)
        out[:, 0::2] = pre
        out[:, 1::2] = post
        return out

    def _build_key_padding_mask(self, valid_mask: torch.Tensor) -> torch.Tensor:
        """``True`` means *masked out* (PyTorch convention). Pad positions in PA
        sequence are masked; valid positions are not."""
        B, T = valid_mask.shape
        # Replicate each step's valid bit across (pre, post) → length 2T.
        rep = valid_mask.repeat_interleave(2, dim=1)
        return ~rep  # True = pad

    @staticmethod
    def _causal_mask(L: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)

    def encode(self, batch: PABatch) -> torch.Tensor:
        """Run encoders + transformer. Returns interleaved encoded sequence
        ``(B, 2T, d_model)``."""
        pre = self.pre_encoder(batch.pre_cat, batch.pre_cont, batch.profile)
        post = self.post_encoder(batch.post_cat, batch.post_cont)

        seq = self._interleave(pre, post)  # (B, 2T, d_model)
        B, L, D = seq.shape
        if L > self.cfg.max_seq_len:
            raise ValueError(
                f"interleaved sequence length {L} exceeds max_seq_len={self.cfg.max_seq_len}; "
                "increase QTransformerConfig.max_seq_len"
            )
        # Add role embeddings (0=pre, 1=post) and absolute position embeddings.
        role_idx = torch.arange(L, device=seq.device) % 2
        pos_idx = torch.arange(L, device=seq.device)
        seq = seq + self.role_emb(role_idx).unsqueeze(0) + self.pos_emb(pos_idx).unsqueeze(0)

        attn_mask = self._causal_mask(L, seq.device)
        kpm = self._build_key_padding_mask(batch.valid_mask)
        h = self.transformer(seq, mask=attn_mask, src_key_padding_mask=kpm)
        h = self.norm_out(h)
        return h

    # --------------------------------------------------------------------- #
    # Heads — both training (chosen-action Q) and inference (full logits)
    # --------------------------------------------------------------------- #

    def _q_type_logits(
        self,
        h_pre: torch.Tensor,                 # (B, T, d_model)
        arsenal_per_type: torch.Tensor,      # (B, T, n_pitch_types, n_arsenal)  raw
        batter_per_type: torch.Tensor,       # (B, T, n_pitch_types, n_batter_pt) raw
    ) -> torch.Tensor:
        """Type-head logits over all candidate pitch types. Broadcasts h_pre over
        the type dim and concatenates per-(pitcher, type) and per-(batter, type)
        features so the head can value each candidate using pitcher stuff and
        batter tendencies for that specific pitch type."""
        ars = self.arsenal_norm(arsenal_per_type)       # standardized
        bpt = self.batter_pt_norm(batter_per_type)
        B, T, n_t, _ = ars.shape
        h_exp = h_pre.unsqueeze(2).expand(B, T, n_t, -1)
        x = torch.cat([h_exp, ars, bpt], dim=-1)         # (B, T, n_t, d_h + n_a + n_b)
        return self.q_head_type(x).squeeze(-1)            # (B, T, n_t)

    def _gather_per_type_features(
        self,
        per_type: torch.Tensor,             # (B, T, n_pitch_types, K)
        chosen_pitch_type: torch.Tensor,    # (B, T) int64
    ) -> torch.Tensor:
        """Gather features for the chosen pitch type — returns (B, T, K)."""
        idx = chosen_pitch_type.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, per_type.size(-1))
        return per_type.gather(2, idx).squeeze(2)

    def heads_chosen(
        self,
        h_pre: torch.Tensor,                # (B, T, d_model) — encoded state at pre_i positions
        chosen_pitch_type: torch.Tensor,    # (B, T) int64
        chosen_x_bin: torch.Tensor,         # (B, T) int64
        chosen_z_bin: torch.Tensor,         # (B, T) int64
        arsenal_per_type: torch.Tensor,     # (B, T, n_pitch_types, n_arsenal) raw
        batter_per_type: torch.Tensor,      # (B, T, n_pitch_types, n_batter_pt) raw
    ) -> dict[str, torch.Tensor]:
        """Evaluate the autoregressive Q-heads on the *chosen* action tuple at
        each timestep. Returns the per-head Q-values for the chosen indices and
        the V-value."""
        e_pt = self.emb_pitch_type_action(chosen_pitch_type)
        e_x = self.emb_x_bin_action(chosen_x_bin)

        # Per-type type-head logits (full vocab) — used both for the chosen-type
        # gather and at inference for argmax.
        q_type_logits = self._q_type_logits(h_pre, arsenal_per_type, batter_per_type)

        # Gather chosen-type arsenal/profile features for the x/z heads.
        ars_chosen = self.arsenal_norm(self._gather_per_type_features(arsenal_per_type, chosen_pitch_type))
        bpt_chosen = self.batter_pt_norm(self._gather_per_type_features(batter_per_type, chosen_pitch_type))

        q_x_logits = self.q_head_x(torch.cat([h_pre, e_pt, ars_chosen, bpt_chosen], dim=-1))
        q_z_logits = self.q_head_z(torch.cat([h_pre, e_pt, e_x, ars_chosen, bpt_chosen], dim=-1))

        q_type = q_type_logits.gather(-1, chosen_pitch_type.unsqueeze(-1)).squeeze(-1)
        q_x = q_x_logits.gather(-1, chosen_x_bin.unsqueeze(-1)).squeeze(-1)
        q_z = q_z_logits.gather(-1, chosen_z_bin.unsqueeze(-1)).squeeze(-1)
        v = self.v_head(h_pre).squeeze(-1)
        return {
            "q_type": q_type, "q_x": q_x, "q_z": q_z, "q_chosen": q_z,
            "v": v,
            "q_type_logits": q_type_logits, "q_x_logits": q_x_logits, "q_z_logits": q_z_logits,
        }

    # --------------------------------------------------------------------- #
    # Top-level forward
    # --------------------------------------------------------------------- #

    def forward(self, batch: PABatch) -> dict[str, torch.Tensor]:
        h = self.encode(batch)               # (B, 2T, d_model)
        h_pre = h[:, 0::2]                   # (B, T, d_model) — state representations
        out = self.heads_chosen(
            h_pre,
            batch.post_cat["pitch_type_id"],
            batch.post_cat["x_bin"],
            batch.post_cat["z_bin"],
            batch.arsenal_per_type,
            batch.batter_per_type,
        )
        out["h_pre"] = h_pre
        out["valid_mask"] = batch.valid_mask
        return out

    # --------------------------------------------------------------------- #
    # Inference policy: autoregressive argmax with optional repertoire mask
    # --------------------------------------------------------------------- #

    @torch.no_grad()
    def policy(
        self,
        batch: PABatch,
        repertoire_mask: torch.Tensor | None = None,  # (B, T, n_pitch_type) bool, True = allowed
    ) -> dict[str, torch.Tensor]:
        """Greedy policy: argmax type → conditional argmax x → conditional argmax z."""
        h = self.encode(batch)
        h_pre = h[:, 0::2]

        q_type_logits = self._q_type_logits(h_pre, batch.arsenal_per_type, batch.batter_per_type)
        if repertoire_mask is not None:
            q_type_logits = q_type_logits.masked_fill(~repertoire_mask, float("-inf"))
        chosen_type = q_type_logits.argmax(dim=-1)               # (B, T)

        e_pt = self.emb_pitch_type_action(chosen_type)
        ars_chosen = self.arsenal_norm(self._gather_per_type_features(batch.arsenal_per_type, chosen_type))
        bpt_chosen = self.batter_pt_norm(self._gather_per_type_features(batch.batter_per_type, chosen_type))
        q_x_logits = self.q_head_x(torch.cat([h_pre, e_pt, ars_chosen, bpt_chosen], dim=-1))
        chosen_x = q_x_logits.argmax(dim=-1)

        e_x = self.emb_x_bin_action(chosen_x)
        q_z_logits = self.q_head_z(torch.cat([h_pre, e_pt, e_x, ars_chosen, bpt_chosen], dim=-1))
        chosen_z = q_z_logits.argmax(dim=-1)

        return {
            "pitch_type": chosen_type,
            "x_bin": chosen_x,
            "z_bin": chosen_z,
            "valid_mask": batch.valid_mask,
        }


# --------------------------------------------------------------------------- #
# IQL loss helper
# --------------------------------------------------------------------------- #


def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    """Asymmetric L2 used by IQL's value learner. ``diff = Q - V``; for diff > 0
    we weight by ``tau``, for diff <= 0 we weight by ``1 - tau``."""
    weight = torch.where(diff > 0, tau, 1.0 - tau)
    return weight * diff.pow(2)


def iql_losses(
    q_chosen: torch.Tensor,        # (B, T) Q for the action actually taken
    v_current: torch.Tensor,       # (B, T) V at current state
    v_next: torch.Tensor,          # (B, T) V at next state (s_{t+1})
    reward: torch.Tensor,          # (B, T)
    is_terminal: torch.Tensor,     # (B, T) bool — True on the final pitch of PA
    valid_mask: torch.Tensor,      # (B, T) bool — True where position is real (not padding)
    *,
    gamma: float = 1.0,
    tau: float = 0.7,
) -> dict[str, torch.Tensor]:
    """Returns ``{q_loss, v_loss}`` averaged over valid positions.

    Q-loss: TD with V (no max over actions — IQL's key trick).
        target = r + γ · V(s') · (1 − terminal)
        L_Q   = MSE(Q(s, a), target)

    V-loss: expectile regression of Q under the data distribution.
        L_V   = expectile_loss(Q(s, a) − V(s), tau)
    """
    target = reward + gamma * v_next * (~is_terminal).float()
    q_err = q_chosen - target
    q_loss_per = q_err.pow(2)

    v_diff = q_chosen.detach() - v_current
    v_loss_per = expectile_loss(v_diff, tau)

    mask = valid_mask.float()
    n = mask.sum().clamp_min(1.0)
    return {
        "q_loss": (q_loss_per * mask).sum() / n,
        "v_loss": (v_loss_per * mask).sum() / n,
    }


def shift_v_for_next_state(
    v: torch.Tensor,           # (B, T) V at each pre_i position
    valid_mask: torch.Tensor,  # (B, T)
) -> torch.Tensor:
    """``v[:, t]`` is V(s_t); we need V(s_{t+1}) per timestep. Shift left by one,
    pad the trailing position with zero (which is fine because ``is_terminal[t] = True``
    on the last valid pitch makes the bootstrap term irrelevant)."""
    v_next = torch.zeros_like(v)
    v_next[:, :-1] = v[:, 1:]
    return v_next


# --------------------------------------------------------------------------- #
# Repertoire mask helper
# --------------------------------------------------------------------------- #


def build_repertoire_mask(
    pitcher_id: torch.Tensor,           # (B, T) int64
    arsenal_lookup: dict[tuple[int, int], int],  # {(pitcher_id, pitch_type_id): count}
    n_pitch_types: int,
    n_min: int,
) -> torch.Tensor:
    """Build a (B, T, n_pitch_types) bool mask. ``True`` means the pitch type is
    in the pitcher's repertoire (count >= ``n_min``); ``False`` means mask out.

    Pitchers with no arsenal entries (UNK / out-of-train-vocab) get an all-True
    mask — fall back to no masking rather than blocking every action.
    """
    B, T = pitcher_id.shape
    mask = torch.zeros(B, T, n_pitch_types, dtype=torch.bool)
    seen_pitchers: dict[int, set[int]] = {}
    for (pid, pt), cnt in arsenal_lookup.items():
        if cnt >= n_min:
            seen_pitchers.setdefault(int(pid), set()).add(int(pt))

    pid_np = pitcher_id.cpu().numpy()
    for b in range(B):
        for t in range(T):
            pid = int(pid_np[b, t])
            allowed = seen_pitchers.get(pid)
            if allowed is None or len(allowed) == 0:
                mask[b, t, :] = True  # fallback
            else:
                for pt in allowed:
                    if 0 <= pt < n_pitch_types:
                        mask[b, t, pt] = True
    return mask
