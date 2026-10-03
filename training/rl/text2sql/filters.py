"""Dynamic-sampling filters for the fan-out Text-to-SQL rollout.

slime ships `filter_hub.dynamic_sampling_filters.check_no_aborted`, and it is
exactly the right policy -- "ABORTED samples typically have empty tokens / zero
response_length / None reward, which would crash or hang the training step". But
it assumes each element of a group is a `Sample`, and does
`any(s.status == ... for s in samples)`.

Our `generate()` fans one trajectory out into one sample per merged segment, so
a group is `list[list[Sample]]` and the built-in filter dies with
`AttributeError: 'list' object has no attribute 'status'`. slime's
own rollout loop already handles both shapes -- it tests
`isinstance(group[0], list)` in four places in `sglang_rollout.py` -- the filter
hub simply never got the same treatment.

These flatten first and then apply the identical policy.
"""

from __future__ import annotations

from collections.abc import Iterable

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample


def _flatten(group: Iterable) -> list[Sample]:
    out: list[Sample] = []
    for entry in group:
        if isinstance(entry, list):
            out.extend(entry)
        else:
            out.append(entry)
    return out


def check_no_aborted(args, samples, **kwargs) -> DynamicFilterOutput:
    """Drop the whole group if any segment of any trajectory aborted."""
    flat = _flatten(samples)
    has_aborted = any(s.status == Sample.Status.ABORTED for s in flat)
    return DynamicFilterOutput(
        keep=not has_aborted,
        reason=None if not has_aborted else "aborted",
    )


def check_no_aborted_and_reward_nonzero_std(args, samples, **kwargs) -> DynamicFilterOutput:
    """`check_no_aborted`, then DAPO-style zero-std rejection.

    This is selected by ``ALGORITHM=dapo``. At the pass rate observed in older
    runs it may need substantially more generated groups to fill one batch;
    that is intentional DAPO dynamic sampling, not a stalled rollout.

    Std is taken over trajectories, not segments: every segment of one
    trajectory carries that trajectory's reward, so flattening first would
    weight a long trajectory more heavily and understate the spread.
    """
    result = check_no_aborted(args, samples, **kwargs)
    if not result.keep:
        return result

    import torch

    rewards = []
    for entry in samples:
        sample = entry[0] if isinstance(entry, list) else entry
        rewards.append(sample.get_reward_value(args))
    keep = bool(torch.tensor(rewards, dtype=torch.float64).std() > 1e-6)
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )
