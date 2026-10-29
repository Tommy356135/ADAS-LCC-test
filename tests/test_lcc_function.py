
"""LCC baseline functional tests with evidence archiving.

Execution ID: LCC-EXEC-002

Learning-only HighwayEnv test project.
All thresholds are project-specific and are not production ADAS requirements.
"""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import highway_env
import numpy as np
import pytest

from src.controller import LCCController, LCCGains, wrap_angle


gym.register_envs(highway_env)


# ============================================================
# 1. Test configuration and execution metadata
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXECUTION_ID = "LCC-EXEC-002"
EXECUTION_OUTPUT = PROJECT_ROOT / "outputs" / EXECUTION_ID
RAW_LOG_DIR = EXECUTION_OUTPUT / "raw_logs"
SUMMARY_DIR = EXECUTION_OUTPUT / "run_summaries"

SEED = 42
MAX_STEPS = 60
INITIAL_LONGITUDINAL_M = 5.0

STATE_NOISE = 0.0
DERIVATIVE_NOISE = 0.0

# Learning-project acceptance criteria.
# These are not universal production-vehicle thresholds.
LATERAL_TOLERANCE_M = 0.25
MAX_SETTLING_STEP = 50
STABLE_TAIL_SAMPLES = 10

CONTROLLER_VERSION = "learning-baseline-v1"

TEST_ID_BY_OFFSET = {
    -4.0: "LCC-FUN-001A",
    4.0: "LCC-FUN-001B",
}


# ============================================================
# 2. Utility functions
# ============================================================

def utc_now():
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def sha256_file(path: Path) -> str:
    """Calculate a source file's SHA-256 hash."""
    if not path.is_file():
        return "FILE_NOT_FOUND"

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def flatten_observation(observation, prefix="controller_input"):
    """
    Flatten the observation used by the controller into CSV columns.

    These values represent controller inputs.
    They are distinct from ground-truth evaluation measurements.
    """
    fields = {}

    for observation_name in (
        "state",
        "derivative",
        "reference_state",
    ):
        if observation_name not in observation:
            raise ValueError(
                f"Observation missing '{observation_name}'."
            )

        values = np.asarray(
            observation[observation_name],
            dtype=float,
        ).reshape(-1)

        for index, value in enumerate(values):
            fields[
                f"{prefix}_{observation_name}_{index}"
            ] = float(value)

    return fields


def measure_true_state(base_env):
    """
    Read ground-truth state for evaluation only.

    The controller does not receive these measurements directly.
    """
    vehicle = base_env.vehicle
    lane = base_env.lane

    longitudinal, lateral_error = lane.local_coordinates(
        vehicle.position
    )

    lane_heading = lane.heading_at(longitudinal)

    heading_error = wrap_angle(
        vehicle.heading - lane_heading
    )

    return {
        "eval_longitudinal_m": float(longitudinal),
        "eval_lateral_error_m": float(lateral_error),
        "eval_heading_error_rad": float(heading_error),
        "eval_speed_mps": float(vehicle.speed),
        "eval_within_lane": bool(
            lane.on_lane(vehicle.position)
        ),
        "eval_vehicle_crashed": bool(
            getattr(vehicle, "crashed", False)
        ),
    }


def find_settling_step(errors, tolerance):
    """
    Return the first step after the last sample outside the band.

    Return None if the final sample remains outside the band.
    """
    errors = np.asarray(errors, dtype=float)

    outside_indices = np.flatnonzero(
        np.abs(errors) > tolerance
    )

    if len(outside_indices) == 0:
        return 0

    last_outside_index = int(outside_indices[-1])

    if last_outside_index == len(errors) - 1:
        return None

    return last_outside_index + 1


def calculate_run_metrics(
    records,
    steps_completed,
    ended_early,
):
    """Calculate the diagnostic metrics for a recorded run."""

    lateral_errors = np.asarray(
        [
            row["eval_lateral_error_m"]
            for row in records
        ],
        dtype=float,
    )

    heading_errors = np.asarray(
        [
            row["eval_heading_error_rad"]
            for row in records
        ],
        dtype=float,
    )

    # The initial row has no control action.
    actions = np.asarray(
        [
            row["steering_action"]
            for row in records[1:]
        ],
        dtype=float,
    )

    numeric_data = np.concatenate(
        [
            lateral_errors,
            heading_errors,
            actions,
        ]
    )

    finite_data = bool(
        np.isfinite(numeric_data).all()
    )

    settling_step = find_settling_step(
        lateral_errors,
        LATERAL_TOLERANCE_M,
    )

    stable_tail = lateral_errors[
        -STABLE_TAIL_SAMPLES:
    ]

    action_validity = [
        row["action_valid"]
        for row in records
    ]

    lane_violations = sum(
        not row["eval_within_lane"]
        for row in records
    )

    action_changes = (
        np.abs(np.diff(actions))
        if len(actions) >= 2
        else np.asarray([], dtype=float)
    )

    if len(actions) > 0:
        steering_min = float(np.min(actions))
        steering_max = float(np.max(actions))
    else:
        steering_min = None
        steering_max = None

    if finite_data and len(lateral_errors) > 0:
        initial_error = float(lateral_errors[0])
        final_error = float(lateral_errors[-1])

        lateral_rmse = float(
            np.sqrt(np.mean(lateral_errors ** 2))
        )

        mean_absolute_error = float(
            np.mean(np.abs(lateral_errors))
        )

        max_absolute_error = float(
            np.max(np.abs(lateral_errors))
        )

        max_heading_error = float(
            np.max(np.abs(heading_errors))
        )
    else:
        initial_error = None
        final_error = None
        lateral_rmse = None
        mean_absolute_error = None
        max_absolute_error = None
        max_heading_error = None

    return {
        "steps_completed": int(steps_completed),
        "ended_early": bool(ended_early),
        "record_count": len(records),
        "finite_data": finite_data,
        "all_actions_valid": bool(
            all(action_validity)
        ),
        "initial_lateral_error_m": initial_error,
        "final_lateral_error_m": final_error,
        "final_abs_lateral_error_m": (
            abs(final_error)
            if final_error is not None
            else None
        ),
        "lateral_rmse_m": lateral_rmse,
        "lateral_mae_m": mean_absolute_error,
        "max_abs_lateral_error_m": max_absolute_error,
        "max_abs_heading_error_rad": max_heading_error,
        "steering_min": steering_min,
        "steering_max": steering_max,
        "max_adjacent_action_change": (
            float(np.max(action_changes))
            if len(action_changes)
            else 0.0
        ),
        "lane_check_violations": int(lane_violations),
        "settling_step_025m": settling_step,
        "final_10_within_025m": bool(
            len(stable_tail) == STABLE_TAIL_SAMPLES
            and np.all(
                np.abs(stable_tail)
                <= LATERAL_TOLERANCE_M
            )
        ),
    }


def save_csv(path, records):
    """Save per-step measurements to CSV."""
    if not records:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(records[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(records)


def save_json(path, data):
    """Save execution metadata and metric summary to JSON."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )


# ============================================================
# 3. Simulation fixture and automatic evidence collection
# ============================================================

@pytest.fixture(
    scope="module",
    params=[-4.0, 4.0],
    ids=[
        "offset_negative_4m",
        "offset_positive_4m",
    ],
)
def baseline_run(request):
    """
    Execute one baseline simulation for each initial offset.

    Each fixture run automatically saves:
    1. Raw per-step measurement CSV.
    2. A JSON execution summary and metadata.
    """

    initial_lateral_offset_m = float(request.param)

    test_id = TEST_ID_BY_OFFSET[
        initial_lateral_offset_m
    ]

    timestamp = utc_now()

    timestamp_for_filename = timestamp.strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    run_id = (
        f"{EXECUTION_ID}_{test_id}_"
        f"{timestamp_for_filename}"
    )

    raw_csv_path = RAW_LOG_DIR / f"{run_id}.csv"
    summary_json_path = (
        SUMMARY_DIR / f"{run_id}.summary.json"
    )

    controller_source = (
        PROJECT_ROOT / "src" / "controller.py"
    )

    test_source = Path(__file__).resolve()

    metadata = {
        "execution_id": EXECUTION_ID,
        "run_id": run_id,
        "test_id": test_id,
        "scenario_id": "LCC_STRAIGHT_OFFSET_CORRECTION",
        "executed_at_utc": timestamp.isoformat(),
        "seed": SEED,
        "controller_name": "LCCController",
        "controller_version": CONTROLLER_VERSION,
        "controller_k_lateral": 0.025,
        "controller_k_heading": 0.45,
        "controller_max_action": 1.0,
        "initial_longitudinal_m": INITIAL_LONGITUDINAL_M,
        "initial_lateral_offset_m": initial_lateral_offset_m,
        "state_noise": STATE_NOISE,
        "derivative_noise": DERIVATIVE_NOISE,
        "planned_steps": MAX_STEPS,
        "lateral_tolerance_m": LATERAL_TOLERANCE_M,
        "max_settling_step": MAX_SETTLING_STEP,
        "stable_tail_samples": STABLE_TAIL_SAMPLES,
        "controller_sha256": sha256_file(
            controller_source
        ),
        "test_script_sha256": sha256_file(
            test_source
        ),
    }

    env = None
    records = []
    steps_completed = 0
    ended_early = False
    execution_exception = None

    try:
        env = gym.make(
            "lane-keeping-v0",
            config={
                "state_noise": STATE_NOISE,
                "derivative_noise": DERIVATIVE_NOISE,
            },
        )

        env.reset(seed=SEED)

        base = env.unwrapped
        lane = base.lane

        # Configure a repeatable initial condition.
        base.vehicle.position = lane.position(
            INITIAL_LONGITUDINAL_M,
            initial_lateral_offset_m,
        )

        base.vehicle.heading = lane.heading_at(
            INITIAL_LONGITUDINAL_M
        )

        observation = (
            base.observation_type.observe()
        )

        controller = LCCController(
            LCCGains(
                k_lateral=0.025,
                k_heading=0.45,
                max_action=1.0,
            )
        )

        initial_measurement = measure_true_state(base)

        # Step 0 captures the initial condition.
        # No control action has been applied yet.
        initial_row = {
            **metadata,
            "step": 0,
            **flatten_observation(observation),
            **initial_measurement,
            "steering_action": None,
            "action_valid": True,
            "reward_from_step": None,
            "step_terminated": False,
            "step_truncated": False,
        }

        records.append(initial_row)

        for step in range(1, MAX_STEPS + 1):
            # Preserve the exact observation used to compute this action.
            observation_used = observation

            action = controller.compute_action(
                observation_used
            )

            action_valid = bool(
                env.action_space.contains(action)
            )

            if not action_valid:
                raise ValueError(
                    f"Invalid action at step {step}: {action}"
                )

            (
                _,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(action)

            # The lane-keeping environment needs a fresh observation
            # after the simulation update for the next control cycle.
            observation = (
                base.observation_type.observe()
            )

            current_measurement = measure_true_state(base)

            numeric_values = [
                current_measurement["eval_longitudinal_m"],
                current_measurement["eval_lateral_error_m"],
                current_measurement["eval_heading_error_rad"],
                current_measurement["eval_speed_mps"],
                float(action[0]),
                float(reward),
            ]

            if not np.isfinite(numeric_values).all():
                raise ValueError(
                    f"Non-finite measurement at step {step}"
                )

            # Each row records:
            # - the observation used to compute the action;
            # - the action and reward;
            # - the resulting ground-truth evaluation state.
            row = {
                **metadata,
                "step": step,
                **flatten_observation(
                    observation_used
                ),
                **current_measurement,
                "steering_action": float(action[0]),
                "action_valid": action_valid,
                "reward_from_step": float(reward),
                "step_terminated": bool(terminated),
                "step_truncated": bool(truncated),
            }

            records.append(row)
            steps_completed += 1

            if terminated or truncated:
                ended_early = (
                    step < MAX_STEPS
                )
                break

    except Exception as exc:
        execution_exception = exc

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    # Archive partial measurements too, if an execution error occurred.
    save_csv(
        raw_csv_path,
        records,
    )

    if records:
        metrics = calculate_run_metrics(
            records,
            steps_completed,
            ended_early,
        )
    else:
        metrics = {
            "steps_completed": steps_completed,
            "ended_early": ended_early,
            "record_count": 0,
            "finite_data": False,
        }

    summary = {
        "metadata": metadata,
        "execution": {
            "execution_error": (
                f"{type(execution_exception).__name__}: "
                f"{execution_exception}"
                if execution_exception is not None
                else None
            ),
            "steps_completed": steps_completed,
            "ended_early": ended_early,
            "record_count": len(records),
        },
        "metrics": metrics,
        "artifacts": {
            "raw_csv": (
                str(raw_csv_path.resolve())
                if records
                else None
            ),
            "summary_json": str(
                summary_json_path.resolve()
            ),
        },
        "acceptance": {
            "status": "NOT_EVALUATED_IN_FIXTURE",
            "note": (
                "The pytest assertions provide the test outcomes. "
                "This JSON stores run evidence and diagnostic metrics."
            ),
        },
    }

    save_json(
        summary_json_path,
        summary,
    )

    if execution_exception is not None:
        raise RuntimeError(
            f"Simulation failed for {test_id}. "
            f"Execution summary: {summary_json_path.resolve()}"
        ) from execution_exception

    # Returned data is used by the assertions below.
    lateral_errors = np.asarray(
        [
            row["eval_lateral_error_m"]
            for row in records
        ],
        dtype=float,
    )

    heading_errors = np.asarray(
        [
            row["eval_heading_error_rad"]
            for row in records
        ],
        dtype=float,
    )

    actions = np.asarray(
        [
            row["steering_action"]
            for row in records[1:]
        ],
        dtype=float,
    )

    action_validity = np.asarray(
        [
            row["action_valid"]
            for row in records[1:]
        ],
        dtype=bool,
    )

    lane_flags = np.asarray(
        [
            row["eval_within_lane"]
            for row in records
        ],
        dtype=bool,
    )

    return {
        "execution_id": EXECUTION_ID,
        "run_id": run_id,
        "test_id": test_id,
        "metadata": metadata,
        "initial_lateral_offset_m": initial_lateral_offset_m,
        "records": records,
        "lateral_errors": lateral_errors,
        "heading_errors": heading_errors,
        "actions": actions,
        "action_validity": action_validity,
        "lane_flags": lane_flags,
        "steps_completed": steps_completed,
        "ended_early": ended_early,
        "raw_csv_path": raw_csv_path,
        "summary_json_path": summary_json_path,
    }


# ============================================================
# 4. Automated assertions
# ============================================================

def test_lcc_executes_all_steps(baseline_run):
    """TC-01: The simulation completes all planned steps."""

    assert (
        baseline_run["steps_completed"]
        == MAX_STEPS
    ), (
        f"{baseline_run['test_id']}: completed "
        f"{baseline_run['steps_completed']} steps; "
        f"expected {MAX_STEPS}."
    )

    assert not baseline_run["ended_early"], (
        f"{baseline_run['test_id']}: simulation ended early."
    )


def test_lcc_data_is_finite(baseline_run):
    """TC-02: Evaluation measurements contain finite values."""

    assert np.isfinite(
        baseline_run["lateral_errors"]
    ).all(), "Lateral error contains NaN or Inf."

    assert np.isfinite(
        baseline_run["heading_errors"]
    ).all(), "Heading error contains NaN or Inf."

    assert np.isfinite(
        baseline_run["actions"]
    ).all(), "Steering actions contain NaN or Inf."


def test_lcc_actions_are_valid(baseline_run):
    """TC-03: All steering actions respect the action space."""

    assert baseline_run["action_validity"].all(), (
        "One or more actions were rejected by the action space."
    )

    actions = baseline_run["actions"]

    assert np.all(actions >= -1.0), (
        "A steering action is below -1.0."
    )

    assert np.all(actions <= 1.0), (
        "A steering action is above +1.0."
    )


def test_lcc_reduces_lateral_error(baseline_run):
    """TC-04: The final absolute lateral error is smaller."""

    errors = baseline_run["lateral_errors"]

    initial_error = abs(errors[0])
    final_error = abs(errors[-1])

    assert final_error < initial_error, (
        f"{baseline_run['test_id']}: lateral error did not improve. "
        f"Initial={initial_error:.4f} m; "
        f"final={final_error:.4f} m."
    )


def test_lcc_final_lateral_error_within_tolerance(
    baseline_run,
):
    """TC-05: The final lateral error meets the project criterion."""

    final_error = baseline_run["lateral_errors"][-1]

    assert abs(final_error) <= LATERAL_TOLERANCE_M, (
        f"{baseline_run['test_id']}: final lateral error "
        f"{final_error:.4f} m exceeds "
        f"{LATERAL_TOLERANCE_M:.2f} m."
    )


def test_lcc_settles_and_remains_within_tolerance(
    baseline_run,
):
    """TC-06: The trajectory settles and remains in the band."""

    errors = baseline_run["lateral_errors"]

    settling_step = find_settling_step(
        errors,
        LATERAL_TOLERANCE_M,
    )

    assert settling_step is not None, (
        "The trajectory did not settle within the tolerance band."
    )

    assert settling_step <= MAX_SETTLING_STEP, (
        f"Settling step {settling_step} exceeds "
        f"{MAX_SETTLING_STEP}."
    )

    stable_tail = errors[-STABLE_TAIL_SAMPLES:]

    assert np.all(
        np.abs(stable_tail) <= LATERAL_TOLERANCE_M
    ), (
        f"The final {STABLE_TAIL_SAMPLES} samples are not all "
        "within the tolerance band."
    )


def test_lcc_vehicle_remains_on_lane(baseline_run):
    """TC-07: All states pass HighwayEnv's lane check."""

    assert baseline_run["lane_flags"].all(), (
        f"{baseline_run['test_id']}: at least one state failed "
        "lane.on_lane()."
    )


def test_lcc_raw_log_is_archived(baseline_run):
    """TC-08: Per-step evidence is written to CSV."""

    raw_path = Path(
        baseline_run["raw_csv_path"]
    )

    assert raw_path.is_file(), (
        f"Raw CSV was not archived: {raw_path}"
    )

    with raw_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == MAX_STEPS + 1, (
        f"Expected {MAX_STEPS + 1} records, "
        f"found {len(rows)}."
    )

    assert int(rows[0]["step"]) == 0
    assert int(rows[-1]["step"]) == MAX_STEPS

    assert all(
        row["test_id"] == baseline_run["test_id"]
        for row in rows
    ), "CSV contains an unexpected test ID."


def test_lcc_execution_summary_is_archived(baseline_run):
    """TC-09: Execution metadata and source hashes are saved."""

    summary_path = Path(
        baseline_run["summary_json_path"]
    )

    assert summary_path.is_file(), (
        f"Execution summary was not archived: {summary_path}"
    )

    with summary_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        summary = json.load(file)

    metadata = summary["metadata"]

    assert metadata["execution_id"] == EXECUTION_ID
    assert metadata["test_id"] == baseline_run["test_id"]

    assert metadata["controller_sha256"] != "FILE_NOT_FOUND"
    assert metadata["test_script_sha256"] != "FILE_NOT_FOUND"

    assert summary["execution"]["record_count"] == (
        MAX_STEPS + 1
    )

    assert summary["execution"]["execution_error"] is None

    assert summary["artifacts"]["raw_csv"] is not None
