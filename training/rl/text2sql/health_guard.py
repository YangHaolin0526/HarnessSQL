"""Fail fast when RL rollouts have collapsed, without confusing length units.

The previous version compared ``sample.effective_response_length`` against
``args.rollout_max_response_len``.  Those are different quantities:

  * ``rollout_max_response_len`` is the per-CALL ``max_new_tokens``; each turn
    gets ``min(that, context_len - len(prompt))``.
  * ``effective_response_length`` is the per-TRAJECTORY total over every turn.

In one measured run, trajectories averaged 10,596 response tokens against a
4,096 per-call cap, so ``all(length >= cap)`` was vacuously true and the guard
degenerated into "abort whenever every reward is zero" -- which is also what a
genuinely hard batch looks like early in training.

The signal we actually want is "it was learning and then stopped".  Track it
directly: fire only after ``PATIENCE`` consecutive all-zero rollouts *following*
a rollout that scored above zero.  No length arithmetic, so no unit bug.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

PATIENCE = 2
INITIAL_ZERO_PATIENCE = 5

_state = {"seen_nonzero": False, "zero_streak": 0}


def abort_on_generation_collapse(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    if not samples:
        return False

    rewards = [float(sample.get_reward_value(args)) for sample in samples]
    lengths = [int(sample.effective_response_length) for sample in samples]
    mean_reward = sum(rewards) / len(rewards)
    all_zero = all(abs(r) < 1e-12 for r in rewards)

    if all_zero:
        _state["zero_streak"] += 1
    else:
        _state["seen_nonzero"] = True
        _state["zero_streak"] = 0

    # Length stats are logged, never used as an abort condition -- they are the
    # thing that was misread before. A trajectory whose total is an exact
    # multiple of the per-call cap is the fingerprint of "ran to max_new_tokens
    # without emitting a parseable tool call"; surfacing the count makes that
    # visible without letting it gate the run.
    cap = int(args.rollout_max_response_len)
    at_cap = sum(1 for L in lengths if cap and L % cap == 0)
    logger.info(
        "[RL_HEALTH] rollout=%s reward_mean=%.4f zero_streak=%d "
        "resp_min=%d resp_med=%d resp_max=%d exact_cap_multiples=%d/%d",
        rollout_id, mean_reward, _state["zero_streak"],
        min(lengths), sorted(lengths)[len(lengths) // 2], max(lengths),
        at_cap, len(lengths),
    )

    if (
        not _state["seen_nonzero"]
        and _state["zero_streak"] >= INITIAL_ZERO_PATIENCE
    ):
        raise RuntimeError(
            f"INITIAL_ROLLOUTS_ZERO_REWARD: First {_state['zero_streak']} rollouts all scored zero reward. "
            f"Likely harness / tool-call parser / environment failure (rollout_id={rollout_id})."
        )

    if _state["seen_nonzero"] and _state["zero_streak"] >= PATIENCE:
        raise RuntimeError(
            f"GENERATION_COLLAPSE: reward was non-zero earlier but the last "
            f"{_state['zero_streak']} rollouts are all zero (rollout_id={rollout_id}); "
            f"aborting instead of burning hours on a dead policy. "
            f"resp_med={sorted(lengths)[len(lengths) // 2]} "
            f"exact_cap_multiples={at_cap}/{len(lengths)} cap={cap}"
        )
    return False
