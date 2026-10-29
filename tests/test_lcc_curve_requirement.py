"""Formal LCC curve-lane requirement test for execution LCC-EXEC-003."""

from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
import csv
import json
from pathlib import Path

import gymnasium as gym
import highway_env  # 导入并注册 HighwayEnv 环境
import numpy as np
import pytest
from src.controller import LCCController


# ---------- Frozen test configuration ----------
EXECUTION_ID = "LCC-EXEC-003"
TEST_CASE_ID = "TC-LCC-CURVE-001"
REQUIREMENT_ID = "LCC-REQ-CURVE-001"

SEED = 42
STEPS = 60
START_LONGITUDINAL_M = 5.0
START_LATERAL_M = 0.0
EXPECTED_INITIAL_SPEED_MPS = 8.3
STATE_NOISE = 0.0
DERIVATIVE_NOISE = 0.0

MAX_ABS_LATERAL_ERROR_M = 1.00
FINAL_TAIL_LENGTH = 10
FINAL_TAIL_ABS_ERROR_M = 0.25

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / EXECUTION_ID
RAW_LOG_DIR = OUTPUT_ROOT / "raw_logs"
SUMMARY_DIR = OUTPUT_ROOT / "run_summaries"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return f"{EXECUTION_ID}_{stamp}"


def sha256_file(path: Path):
    if not path.is_file():
        return None

    digest = sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "unknown"


def flatten_observation(observation: dict, key: str) -> list[float]:
    values = np.asarray(observation[key], dtype=float).reshape(-1)
    return [float(value) for value in values]


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def measure_vehicle(base_env, curve_lane) -> dict:
    """Measure the simulated vehicle against the actual curve geometry."""
    longitudinal_m, lateral_error_m = curve_lane.local_coordinates(
        base_env.vehicle.position
    )

    reference_heading_rad = float(
        curve_lane.heading_at(float(longitudinal_m))
    )
    heading_error_rad = wrap_angle(
        float(base_env.vehicle.heading) - reference_heading_rad
    )

    speed_mps = float(base_env.vehicle.speed)
    crashed = bool(getattr(base_env.vehicle, "crashed", False))
    within_lane = bool(curve_lane.on_lane(base_env.vehicle.position))

    values = [
        float(longitudinal_m),
        float(lateral_error_m),
        float(heading_error_rad),
        speed_mps,
    ]

    return {
        "eval_longitudinal_m": float(longitudinal_m),
        "eval_lateral_error_m": float(lateral_error_m),
        "eval_heading_error_rad": float(heading_error_rad),
        "eval_speed_mps": speed_mps,
        "eval_within_lane": within_lane,
        "eval_vehicle_crashed": crashed,
        "evaluation_finite": bool(np.all(np.isfinite(values))),
    }


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = [
            "execution_id",
            "run_id",
            "test_case_id",
            "step_index",
            "recorded_at_utc",
        ]

    with path.open("w", newline="", encoding="utf-8-sig") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_curve_scenario(run_id: str) -> dict:
    RAW_LOG_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RAW_LOG_DIR / f"{run_id}.csv"
    json_path = SUMMARY_DIR / f"{run_id}.json"

    controller_path = PROJECT_ROOT / "src" / "controller.py"
    test_path = Path(__file__).resolve()
    design_path = PROJECT_ROOT / "docs" / "LCC_CURVE_TEST_DESIGN.md"

    controller = LCCController()
    rows = []
    execution_error = None
    initial_lateral_error_m = None
    initial_speed_mps = None
    env = None

    # Keep source references and controller parameters with the evidence.
    source_hashes = {
        "controller_sha256": sha256_file(controller_path),
        "test_script_sha256": sha256_file(test_path),
        "test_design_sha256": sha256_file(design_path),
    }

    controller_parameters = {
        "k_lateral": float(controller.gains.k_lateral),
        "k_heading": float(controller.gains.k_heading),
        "max_action": float(controller.gains.max_action),
    }

    try:
        env = gym.make(
            "lane-keeping-v0",
            config={
                "simulation_frequency": 10,
                "policy_frequency": 10,
                "state_noise": STATE_NOISE,
                "derivative_noise": DERIVATIVE_NOISE,
            },
        )

        observation, reset_info = env.reset(seed=SEED)
        base_env = env.unwrapped

        # The default reset starts with a straight connector lane.
        # Select the SineLane explicitly for this test scenario.
        if not getattr(base_env, "lanes", None):
            raise RuntimeError(
                "Could not select curve lane: env.unwrapped.lanes is empty."
            )

        curve_lane = base_env.lanes[-1]
        base_env.lane = curve_lane
        base_env.lanes = []

        base_env.vehicle.position = curve_lane.position(
            START_LONGITUDINAL_M,
            START_LATERAL_M,
        )
        base_env.vehicle.heading = float(
            curve_lane.heading_at(START_LONGITUDINAL_M)
        )

        # Refresh the observation after manually setting the initial state.
        observation = base_env.observation_type.observe()

        _, initial_lateral_error_m = curve_lane.local_coordinates(
            base_env.vehicle.position
        )
        initial_lateral_error_m = float(initial_lateral_error_m)
        initial_speed_mps = float(base_env.vehicle.speed)

        for step_index in range(1, STEPS + 1):
            # Inputs are exactly those presented to the controller.
            state = flatten_observation(observation, "state")
            derivative = flatten_observation(observation, "derivative")
            reference_state = flatten_observation(
                observation, "reference_state"
            )

            controller_input_values = state + derivative + reference_state
            controller_input_finite = bool(
                np.all(np.isfinite(controller_input_values))
            )

            action = np.asarray(
                controller.compute_action(observation),
                dtype=np.float32,
            ).reshape(-1)

            action_valid = bool(
                action.size == 1
                and np.all(np.isfinite(action))
                and env.action_space.contains(action)
            )

            next_observation, reward, terminated, truncated, info = env.step(
                action
            )

            evaluation = measure_vehicle(base_env, curve_lane)

            row = {
                "execution_id": EXECUTION_ID,
                "run_id": run_id,
                "test_case_id": TEST_CASE_ID,
                "requirement_id": REQUIREMENT_ID,
                "step_index": step_index,
                "recorded_at_utc": utc_now(),
                "seed": SEED,
                "start_longitudinal_m": START_LONGITUDINAL_M,
                "start_lateral_m": START_LATERAL_M,
                "initial_speed_mps": initial_speed_mps,
                "state_noise": STATE_NOISE,
                "derivative_noise": DERIVATIVE_NOISE,
                "controller_k_lateral": controller_parameters["k_lateral"],
                "controller_k_heading": controller_parameters["k_heading"],
                "controller_max_action": controller_parameters["max_action"],
                "controller_input_finite": controller_input_finite,
                "action_valid": action_valid,
                "steering_action": (
                    float(action[0]) if action.size >= 1 else None
                ),
                "reward_from_step": float(reward),
                "step_terminated": bool(terminated),
                "step_truncated": bool(truncated),
                **{
                    f"controller_input_state_{i}": (
                        state[i] if i < len(state) else None
                    )
                    for i in range(4)
                },
                **{
                    f"controller_input_derivative_{i}": (
                        derivative[i] if i < len(derivative) else None
                    )
                    for i in range(4)
                },
                **{
                    f"controller_input_reference_state_{i}": (
                        reference_state[i]
                        if i < len(reference_state)
                        else None
                    )
                    for i in range(4)
                },
                **evaluation,
            }

            rows.append(row)
            observation = next_observation

    except Exception as exc:
        execution_error = {
            "type": type(exc).__name__,
            "message": str(exc),
        }

    finally:
        if env is not None:
            try:
                env.close()
            except Exception as close_exc:
                if execution_error is None:
                    execution_error = {
                        "type": type(close_exc).__name__,
                        "message": f"Error closing environment: {close_exc}",
                    }

    # Evaluate the frozen criteria only after the run and evidence capture.
    lateral_errors = [
        float(row["eval_lateral_error_m"])
        for row in rows
        if row.get("eval_lateral_error_m") is not None
    ]

    max_abs_lateral_error_m = (
        max(abs(value) for value in lateral_errors)
        if lateral_errors
        else None
    )

    final_lateral_error_m = lateral_errors[-1] if lateral_errors else None
    final_abs_lateral_error_m = (
        abs(final_lateral_error_m)
        if final_lateral_error_m is not None
        else None
    )

    final_tail_pass = bool(
        len(lateral_errors) >= FINAL_TAIL_LENGTH
        and all(
            abs(value) <= FINAL_TAIL_ABS_ERROR_M
            for value in lateral_errors[-FINAL_TAIL_LENGTH:]
        )
    )

    input_finite_pass = bool(
        len(rows) == STEPS
        and all(row["controller_input_finite"] for row in rows)
        and all(row["evaluation_finite"] for row in rows)
    )

    action_valid_pass = bool(
        len(rows) == STEPS
        and all(row["action_valid"] for row in rows)
    )

    all_within_lane = bool(
        len(rows) == STEPS
        and all(row["eval_within_lane"] for row in rows)
    )

    no_collision = bool(
        len(rows) == STEPS
        and not any(row["eval_vehicle_crashed"] for row in rows)
    )

    criteria = {
        "REQ-CURVE-01_max_abs_lateral_error_le_1m": {
            "pass": bool(
                max_abs_lateral_error_m is not None
                and max_abs_lateral_error_m <= MAX_ABS_LATERAL_ERROR_M
            ),
            "limit_m": MAX_ABS_LATERAL_ERROR_M,
            "observed_max_abs_m": max_abs_lateral_error_m,
        },
        "REQ-CURVE-02_final_10_samples_within_0_25m": {
            "pass": final_tail_pass,
            "tail_length": FINAL_TAIL_LENGTH,
            "limit_m": FINAL_TAIL_ABS_ERROR_M,
            "observed_final_abs_m": final_abs_lateral_error_m,
        },
        "REQ-CURVE-03_within_lane_and_no_collision": {
            "pass": bool(all_within_lane and no_collision),
            "all_samples_within_lane": all_within_lane,
            "no_collision": no_collision,
        },
        "REQ-CURVE-04_finite_data_and_valid_actions": {
            "pass": bool(input_finite_pass and action_valid_pass),
            "finite_inputs_and_evaluations": input_finite_pass,
            "valid_actions": action_valid_pass,
        },
    }

    violations = [
        criterion_id
        for criterion_id, result in criteria.items()
        if not result["pass"]
    ]

    if execution_error is not None:
        status = "ERROR"
    elif len(rows) != STEPS:
        status = "ERROR"
        violations.append(f"expected_{STEPS}_samples_got_{len(rows)}")
    elif violations:
        status = "FAIL"
    else:
        status = "PASS"

    summary = {
        "execution_id": EXECUTION_ID,
        "run_id": run_id,
        "test_case_id": TEST_CASE_ID,
        "requirement_id": REQUIREMENT_ID,
        "status": status,
        "executed_at_utc": utc_now(),
        "scenario": {
            "environment_id": "lane-keeping-v0",
            "lane_type": "SineLane",
            "seed": SEED,
            "steps_planned": STEPS,
            "steps_recorded": len(rows),
            "start_longitudinal_m": START_LONGITUDINAL_M,
            "start_lateral_m": START_LATERAL_M,
            "initial_lateral_error_m": initial_lateral_error_m,
            "expected_initial_speed_mps": EXPECTED_INITIAL_SPEED_MPS,
            "observed_initial_speed_mps": initial_speed_mps,
            "state_noise": STATE_NOISE,
            "derivative_noise": DERIVATIVE_NOISE,
        },
        "controller_parameters": controller_parameters,
        "software_versions": {
            "python": metadata.version("pip") if False else None,
            "gymnasium": package_version("gymnasium"),
            "highway-env": package_version("highway-env"),
            "numpy": package_version("numpy"),
            "pytest": package_version("pytest"),
        },
        "source_hashes": source_hashes,
        "metrics": {
            "max_abs_lateral_error_m": max_abs_lateral_error_m,
            "final_lateral_error_m": final_lateral_error_m,
            "final_abs_lateral_error_m": final_abs_lateral_error_m,
            "lane_departure_samples": sum(
                not row["eval_within_lane"] for row in rows
            ),
            "collision_samples": sum(
                row["eval_vehicle_crashed"] for row in rows
            ),
            "invalid_action_samples": sum(
                not row["action_valid"] for row in rows
            ),
        },
        "criteria": criteria,
        "violations": violations,
        "execution_error": execution_error,
        "evidence": {
            "raw_csv": str(csv_path.relative_to(PROJECT_ROOT)),
            "summary_json": str(json_path.relative_to(PROJECT_ROOT)),
        },
    }

    save_csv(csv_path, rows)
    with json_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)

    return summary


def test_lcc_curve_tracking_meets_frozen_requirements():
    """TC-LCC-CURVE-001: check closed-loop curve tracking against frozen limits."""
    run_id = make_run_id()
    summary = run_curve_scenario(run_id)

    status = summary["status"]
    evidence_paths = (
        f"\nCSV: {summary['evidence']['raw_csv']}"
        f"\nJSON: {summary['evidence']['summary_json']}"
    )

    if status == "ERROR":
        pytest.fail(
            f"{TEST_CASE_ID} execution ERROR: "
            f"{summary['execution_error']}{evidence_paths}"
        )

    if status == "FAIL":
        pytest.fail(
            f"{TEST_CASE_ID} requirement FAIL. "
            f"Violations: {summary['violations']}. "
            f"Metrics: {summary['metrics']}."
            f"{evidence_paths}"
        )

    assert status == "PASS", f"Unexpected test status: {status}"