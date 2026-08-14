"""Training-only implied-logZ calibration state and robust summaries."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from .complexity_scheduler import BalancedNodeCountScheduler
from .calibration_stability import (
    CalibrationStabilityConfig,
    CalibrationStabilityResult,
    assess_calibration_stability,
)


CALIBRATION_SCHEMA = "factor_gfn.normalizer_calibration.v1"


def require_training_only_provider_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("data_scope") != "training_only":
        raise ValueError("calibration provider must declare data_scope=training_only")
    if manifest.get("validation_oos_loaded") is not False:
        raise ValueError("calibration provider must declare validation_oos_loaded=False")
    if not manifest.get("context_fingerprint"):
        raise ValueError("calibration provider must declare a context_fingerprint")


@dataclass(frozen=True, slots=True)
class CalibrationStatistics:
    node_count: int
    calibration_requested: int
    calibration_valid: int
    calibration_sampled_attempts: int
    median: float
    logmeanexp: float
    p10: float
    p25: float
    p75: float
    p90: float
    iqr: float
    median_implied_minus_exact_tb_log_z: float | None
    logmeanexp_implied_minus_exact_tb_log_z: float | None


class NormalizerCalibration:
    """Independent balanced scheduler and resumable implied-logZ observations."""

    def __init__(
        self,
        *,
        node_counts: tuple[int, ...],
        exhaustive_node_counts: tuple[int, ...],
        minimum_valid_samples: int,
        maximum_requested_slots_per_N: int,
        seed: int,
        stability_config: CalibrationStabilityConfig | None = None,
    ) -> None:
        self.node_counts = tuple(node_counts)
        self.exhaustive_node_counts = tuple(exhaustive_node_counts)
        if not self.node_counts:
            raise ValueError("calibration node_counts must not be empty")
        if not set(self.exhaustive_node_counts).issubset(self.node_counts):
            raise ValueError("calibration exhaustive strata must be a subset of F")
        if minimum_valid_samples < 1:
            raise ValueError("minimum_valid_samples must be positive")
        if maximum_requested_slots_per_N < minimum_valid_samples:
            raise ValueError("maximum requested slots must be at least minimum valid samples")
        self.minimum_valid_samples = int(minimum_valid_samples)
        self.maximum_requested_slots_per_N = int(maximum_requested_slots_per_N)
        self.stability_config = stability_config
        if stability_config is not None and (
            stability_config.minimum_valid_samples != self.minimum_valid_samples
            or stability_config.maximum_requested_slots
            != self.maximum_requested_slots_per_N
        ):
            raise ValueError("calibration stability config does not match sample budget")
        self.scheduler = BalancedNodeCountScheduler(self.node_counts, seed=seed)
        self.requested_by_N = {node_count: 0 for node_count in self.node_counts}
        self.valid_by_N = {node_count: 0 for node_count in self.node_counts}
        self.sampled_attempts_by_N = {
            node_count: 0 for node_count in self.node_counts
        }
        self.implied_log_z_by_N: dict[int, list[float]] = {
            node_count: [] for node_count in self.node_counts
        }
        self.status = "collecting"
        self.failure_reason: str | None = None
        self.statistics_by_N: dict[int, CalibrationStatistics] = {}
        self.stability_by_N: dict[int, CalibrationStabilityResult] = {}

    @property
    def ready_to_finalize(self) -> bool:
        if self.stability_config is not None:
            return all(
                self.stability_by_N.get(node_count) is not None
                and self.stability_by_N[node_count].status == "stable"
                for node_count in self.node_counts
            )
        return all(
            self.valid_by_N[node_count] >= self.minimum_valid_samples
            for node_count in self.node_counts
        )

    def next_node_count(self) -> int:
        if self.status != "collecting":
            raise RuntimeError(f"calibration is not collecting: {self.status}")
        for _ in range(len(self.node_counts)):
            node_count = self.scheduler.next_node_count()
            stability = self.stability_by_N.get(node_count)
            if stability is None or stability.status != "stable":
                return node_count
        raise RuntimeError("all calibration strata are stable and ready to finalize")

    def record_slot(
        self,
        node_count: int,
        *,
        sampled_attempts: int,
        implied_log_z: float | None,
    ) -> None:
        if self.status != "collecting":
            raise RuntimeError(f"calibration is not collecting: {self.status}")
        if node_count not in self.requested_by_N:
            raise ValueError("calibration node_count is outside resolved F")
        if sampled_attempts < 1:
            raise ValueError("calibration sampled_attempts must be positive")
        self.requested_by_N[node_count] += 1
        self.sampled_attempts_by_N[node_count] += int(sampled_attempts)
        if implied_log_z is not None:
            value = float(implied_log_z)
            if not math.isfinite(value):
                raise ValueError("implied logZ must be finite")
            self.valid_by_N[node_count] += 1
            self.implied_log_z_by_N[node_count].append(value)
        if self.stability_config is not None:
            stability = assess_calibration_stability(
                self.implied_log_z_by_N[node_count],
                requested_slots=self.requested_by_N[node_count],
                config=self.stability_config,
            )
            self.stability_by_N[node_count] = stability
            if stability.status == "fail_closed":
                self.status = "failed"
                self.failure_reason = f"N={node_count}: {stability.reason}"
                raise RuntimeError(self.failure_reason)
        if (
            self.requested_by_N[node_count]
            >= self.maximum_requested_slots_per_N
            and self.valid_by_N[node_count] < self.minimum_valid_samples
        ):
            self.status = "failed"
            self.failure_reason = (
                f"N={node_count} reached maximum requested calibration slots "
                f"({self.maximum_requested_slots_per_N}) with only "
                f"{self.valid_by_N[node_count]} valid samples; required "
                f"{self.minimum_valid_samples}"
            )
            raise RuntimeError(self.failure_reason)

    @staticmethod
    def _summarize(
        node_count: int,
        values: list[float],
        *,
        requested: int,
        sampled_attempts: int,
        exact_tb_log_z: float | None,
    ) -> CalibrationStatistics:
        array = np.asarray(values, dtype=np.float64)
        maximum = float(np.max(array))
        logmeanexp = maximum + math.log(
            math.fsum(math.exp(float(value) - maximum) for value in array)
            / array.size
        )
        p10, p25, median, p75, p90 = (
            float(value) for value in np.percentile(array, [10, 25, 50, 75, 90])
        )
        return CalibrationStatistics(
            node_count=node_count,
            calibration_requested=requested,
            calibration_valid=int(array.size),
            calibration_sampled_attempts=sampled_attempts,
            median=median,
            logmeanexp=logmeanexp,
            p10=p10,
            p25=p25,
            p75=p75,
            p90=p90,
            iqr=p75 - p25,
            median_implied_minus_exact_tb_log_z=(
                None if exact_tb_log_z is None else median - exact_tb_log_z
            ),
            logmeanexp_implied_minus_exact_tb_log_z=(
                None if exact_tb_log_z is None else logmeanexp - exact_tb_log_z
            ),
        )

    def finalize(
        self,
        *,
        exact_tb_log_z_by_N: Mapping[int, float],
    ) -> dict[int, CalibrationStatistics]:
        if self.status != "collecting":
            raise RuntimeError(f"calibration cannot finalize from status={self.status}")
        if not self.ready_to_finalize:
            raise RuntimeError("calibration has not reached minimum valid samples for every N")
        missing_exact = sorted(
            set(self.exhaustive_node_counts) - set(exact_tb_log_z_by_N)
        )
        if missing_exact:
            raise RuntimeError(
                f"exhaustive calibration lacks fixed exact TB logZ for {missing_exact}"
            )
        statistics: dict[int, CalibrationStatistics] = {}
        exhaustive = set(self.exhaustive_node_counts)
        for node_count in self.node_counts:
            exact = (
                float(exact_tb_log_z_by_N[node_count])
                if node_count in exhaustive
                else None
            )
            statistics[node_count] = self._summarize(
                node_count,
                self.implied_log_z_by_N[node_count],
                requested=self.requested_by_N[node_count],
                sampled_attempts=self.sampled_attempts_by_N[node_count],
                exact_tb_log_z=exact,
            )
        self.statistics_by_N = statistics
        self.status = "complete"
        return dict(statistics)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": CALIBRATION_SCHEMA,
            "node_counts": self.node_counts,
            "exhaustive_node_counts": self.exhaustive_node_counts,
            "minimum_valid_samples": self.minimum_valid_samples,
            "maximum_requested_slots_per_N": self.maximum_requested_slots_per_N,
            "scheduler": self.scheduler.state_dict(),
            "requested_by_N": dict(self.requested_by_N),
            "valid_by_N": dict(self.valid_by_N),
            "sampled_attempts_by_N": dict(self.sampled_attempts_by_N),
            "implied_log_z_by_N": {
                key: list(values) for key, values in self.implied_log_z_by_N.items()
            },
            "status": self.status,
            "failure_reason": self.failure_reason,
            "statistics_by_N": {
                key: asdict(value) for key, value in self.statistics_by_N.items()
            },
            "stability_by_N": {
                key: asdict(value) for key, value in self.stability_by_N.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict) or state.get("schema") != CALIBRATION_SCHEMA:
            raise ValueError("calibration state schema is incompatible")
        expected = (
            self.node_counts,
            self.exhaustive_node_counts,
            self.minimum_valid_samples,
            self.maximum_requested_slots_per_N,
        )
        actual = (
            tuple(state["node_counts"]),
            tuple(state["exhaustive_node_counts"]),
            int(state["minimum_valid_samples"]),
            int(state["maximum_requested_slots_per_N"]),
        )
        if actual != expected:
            raise ValueError("calibration state does not match current config/F/E")
        keys = set(self.node_counts)
        for name, target in (
            ("requested_by_N", self.requested_by_N),
            ("valid_by_N", self.valid_by_N),
            ("sampled_attempts_by_N", self.sampled_attempts_by_N),
        ):
            values = {int(key): int(value) for key, value in state[name].items()}
            if set(values) != keys or any(value < 0 for value in values.values()):
                raise ValueError(f"invalid calibration {name}")
            target.clear()
            target.update(values)
        observations = {
            int(key): [float(value) for value in values]
            for key, values in state["implied_log_z_by_N"].items()
        }
        if set(observations) != keys:
            raise ValueError("invalid calibration observations strata")
        for node_count in keys:
            if (
                len(observations[node_count]) != self.valid_by_N[node_count]
                or any(not math.isfinite(value) for value in observations[node_count])
            ):
                raise ValueError("invalid calibration observation count/value")
        self.implied_log_z_by_N = observations
        status = state["status"]
        if status not in ("collecting", "complete", "failed"):
            raise ValueError("invalid calibration status")
        self.status = status
        self.failure_reason = state.get("failure_reason")
        self.statistics_by_N = {
            int(key): CalibrationStatistics(**value)
            for key, value in state.get("statistics_by_N", {}).items()
        }
        self.stability_by_N = {
            int(key): CalibrationStabilityResult(**value)
            for key, value in state.get("stability_by_N", {}).items()
        }
        if self.stability_config is not None and not set(
            self.stability_by_N
        ).issubset(keys):
            raise ValueError("invalid calibration stability strata")
        if self.status == "complete" and set(self.statistics_by_N) != keys:
            raise ValueError("complete calibration lacks per-N statistics")
        self.scheduler.load_state_dict(state["scheduler"])


__all__ = [
    "CALIBRATION_SCHEMA",
    "CalibrationStatistics",
    "NormalizerCalibration",
    "require_training_only_provider_manifest",
]
