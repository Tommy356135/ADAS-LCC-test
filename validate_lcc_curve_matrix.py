
"""LCC controller robustness matrix for HighwayEnv.

This is a learning-only diagnostic experiment.
It does not represent production vehicle validation.
"""

import csv
from pathlib import Path

import gymnasium as gym
import highway_env
import numpy as np
import pandas as pd

from src.controller import LCCController, LCCGains, wrap_angle


gym.register_envs(highway_env)

OUTPUT_DIR = Path("outputs")

RUNS_FILE = OUTPUT_DIR / "lcc_robustness_runs.csv"
SUMMARY_FILE = OUTPUT_DIR / "lcc_robustness_summary.csv"

SEEDS = [42, 123, 2026]
INITIAL_OFFSETS_M = [-1.0, 0.0, 1.0]
INITIAL_SPEEDS_MPS = [6.0, 8.3, 10.0]

MAX_STEPS = 60
INITIAL_LONGITUDINAL_M = 5.0

# Diagnostic threshold only; not a production requirement.
REFERENCE_TOLERANCE_M = 0.25
STABLE_TAIL_SAMPLES = 10

# name, lateral gain, heading gain
CONTROLLERS = [
    ("baseline", 0.025, 0.45),
    ("candidate_A", 0.080, 0.45),
    ("candidate_B", 0.080, 0.65),
]

# Both state_noise and derivative_noise use these values.
NOISE_CONFIGS = [
    ("noise_off", 0.0),
    ("noise_005", 0.05),
]


def measure_state(base_env):
    """Read ground-truth state for evaluation only."""

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
        "longitudinal_m": float(longitudinal),
        "lateral_error_m": float(lateral_error),
        "heading_error_rad": float(heading_error),
        "speed_mps": float(vehicle.speed),
        "within_lane": bool(
            lane.on_lane(vehicle.position)
        ),
    }


def calculate_metrics(records):
    """Calculate diagnostic metrics for one run."""

    lateral = np.array(
        [row["lateral_error_m"] for row in records],
        dtype=float,
    )

    heading = np.array(
        [row["heading_error_rad"] for row in records],
        dtype=float,
    )

    actions = np.array(
        [
            row["steering_action"]
            for row in records[1:]
        ],
        dtype=float,
    )

    outside_indices = np.flatnonzero(
        np.abs(lateral) > REFERENCE_TOLERANCE_M
    )

    if len(outside_indices) == 0:
        settling_step = 0
    elif outside_indices[-1] == len(lateral) - 1:
        settling_step = None
    else:
        settling_step = int(outside_indices[-1] + 1)

    tail_size = min(
        STABLE_TAIL_SAMPLES,
        len(lateral),
    )

    final_tail = lateral[-tail_size:]

    if len(actions) >= 2:
        max_action_change = float(
            np.max(np.abs(np.diff(actions)))
        )
    else:
        max_action_change = 0.0

    finite_data = bool(
        np.isfinite(lateral).all()
        and np.isfinite(heading).all()
        and np.isfinite(actions).all()
    )

    return {
        "initial_lateral_error_m": float(lateral[0]),
        "final_lateral_error_m": float(lateral[-1]),
        "final_abs_lateral_error_m": float(
            abs(lateral[-1])
        ),
        "lateral_rmse_m": float(
            np.sqrt(np.mean(lateral ** 2))
        ),
        "lateral_mae_m": float(
            np.mean(np.abs(lateral))
        ),
        "max_abs_lateral_error_m": float(
            np.max(np.abs(lateral))
        ),
        "max_abs_heading_error_rad": float(
            np.max(np.abs(heading))
        ),
        "steering_min": float(np.min(actions)),
        "steering_max": float(np.max(actions)),
        "max_adjacent_action_change": max_action_change,
        "lane_check_violations": int(
            sum(
                not row["within_lane"]
                for row in records
            )
        ),
        "settling_step_025m": settling_step,
        "final_10_within_025m": bool(
            np.all(
                np.abs(final_tail)
                <= REFERENCE_TOLERANCE_M
            )
        ),
        "finite_data": finite_data,
    }


def run_one_case(
    controller_name,
    k_lateral,
    k_heading,
    seed,
    initial_offset_m,
    initial_speed_mps,
    noise_name,
    noise_value,
):
    """Execute one controlled, reproducible scenario."""

    env = gym.make(
        "lane-keeping-v0",
        config={
            "state_noise": noise_value,
            "derivative_noise": noise_value,
        },
    )

    records = []

    try:
        env.reset(seed=seed)
        base = env.unwrapped

        if not base.lanes:
            raise RuntimeError(
                "No subsequent lane found after reset."
            )

        curved_lane = base.lanes[-1]

        if not hasattr(curved_lane, "amplitude"):
            raise RuntimeError(
                "The selected lane is not a sine lane."
            )

        # Deliberately test the sine lane directly.
        # This does not test the connector-to-curve transition.
        base.lane = curved_lane
        base.lanes = []

        lane = base.lane

        base.vehicle.position = lane.position(
            INITIAL_LONGITUDINAL_M,
            initial_offset_m,
        )

        base.vehicle.heading = lane.heading_at(
            INITIAL_LONGITUDINAL_M
        )

        base.vehicle.speed = initial_speed_mps

        base.vehicle.lane_index = (
            base.road.network.get_closest_lane_index(
                base.vehicle.position
            )
        )

        # Refresh observation after modifying initial conditions.
        observation = base.observation_type.observe()

        controller = LCCController(
            LCCGains(
                k_lateral=k_lateral,
                k_heading=k_heading,
                max_action=1.0,
            )
        )

        initial = measure_state(base)

        records.append({
            "step": 0,
            **initial,
            "steering_action": np.nan,
            "reward": np.nan,
        })

        steps_completed = 0
        ended_early = False

        for step in range(1, MAX_STEPS + 1):
            action = controller.compute_action(observation)

            if not env.action_space.contains(action):
                raise ValueError(
                    f"Invalid action at step {step}: {action}"
                )

            _, reward, terminated, truncated, _ = env.step(
                action
            )

            # Refresh observations for the next control cycle.
            observation = base.observation_type.observe()

            current = measure_state(base)

            numeric_values = [
                current["longitudinal_m"],
                current["lateral_error_m"],
                current["heading_error_rad"],
                current["speed_mps"],
                float(action[0]),
            ]

            if not np.isfinite(numeric_values).all():
                raise ValueError(
                    f"Invalid data at step {step}"
                )

            records.append({
                "step": step,
                **current,
                "steering_action": float(action[0]),
                "reward": float(reward),
            })

            steps_completed += 1

            if terminated or truncated:
                ended_early = step < MAX_STEPS
                break

    finally:
        env.close()

    metrics = calculate_metrics(records)

    run_result = {
        "controller": controller_name,
        "k_lateral": k_lateral,
        "k_heading": k_heading,
        "seed": seed,
        "initial_offset_m": initial_offset_m,
        "initial_speed_mps": initial_speed_mps,
        "noise_name": noise_name,
        "noise_value": noise_value,
        "steps_completed": steps_completed,
        "ended_early": ended_early,
        **metrics,
    }

    trajectory = []

    for record in records:
        trajectory.append({
            "controller": controller_name,
            "k_lateral": k_lateral,
            "k_heading": k_heading,
            "seed": seed,
            "initial_offset_m": initial_offset_m,
            "initial_speed_mps": initial_speed_mps,
            "noise_name": noise_name,
            "noise_value": noise_value,
            **record,
        })

    return run_result, trajectory


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    all_trajectories = []

    total_runs = (
        len(CONTROLLERS)
        * len(SEEDS)
        * len(INITIAL_OFFSETS_M)
        * len(INITIAL_SPEEDS_MPS)
        * len(NOISE_CONFIGS)
    )

    print("=" * 78)
    print("LCC 鲁棒性测试矩阵")
    print("=" * 78)
    print(f"计划执行组合数：{total_runs}")
    print("本轮为诊断性测试，不直接给出正式 PASS/FAIL。")
    print("-" * 78)

    run_index = 0

    for controller_name, k_lat, k_heading in CONTROLLERS:
        for noise_name, noise_value in NOISE_CONFIGS:
            for seed in SEEDS:
                for speed in INITIAL_SPEEDS_MPS:
                    for offset in INITIAL_OFFSETS_M:

                        run_index += 1

                        result, trajectory = run_one_case(
                            controller_name=controller_name,
                            k_lateral=k_lat,
                            k_heading=k_heading,
                            seed=seed,
                            initial_offset_m=offset,
                            initial_speed_mps=speed,
                            noise_name=noise_name,
                            noise_value=noise_value,
                        )

                        results.append(result)
                        all_trajectories.extend(trajectory)

                        print(
                            f"[{run_index:03d}/{total_runs}] "
                            f"{controller_name:12s} "
                            f"noise={noise_name:9s} "
                            f"speed={speed:4.1f} "
                            f"offset={offset:+.1f} "
                            f"seed={seed} "
                            f"RMSE={result['lateral_rmse_m']:.3f} "
                            f"max={result['max_abs_lateral_error_m']:.3f}"
                        )

    results_df = pd.DataFrame(results)
    trajectories_df = pd.DataFrame(all_trajectories)

    results_df.to_csv(
        RUNS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    trajectories_df.to_csv(
        OUTPUT_DIR / "lcc_robustness_trajectory.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Group by controller and noise condition for comparison.
    summary_df = (
        results_df
        .groupby(
            ["controller", "noise_name"],
            as_index=False,
        )
        .agg(
            run_count=("controller", "size"),
            mean_rmse_m=("lateral_rmse_m", "mean"),
            mean_final_abs_error_m=(
                "final_abs_lateral_error_m",
                "mean",
            ),
            worst_max_abs_error_m=(
                "max_abs_lateral_error_m",
                "max",
            ),
            mean_max_heading_error_rad=(
                "max_abs_heading_error_rad",
                "mean",
            ),
            stable_tail_runs=(
                "final_10_within_025m",
                "sum",
            ),
            lane_violations=(
                "lane_check_violations",
                "sum",
            ),
            incomplete_runs=("ended_early", "sum"),
            invalid_data_runs=(
                "finite_data",
                lambda values: int((~values).sum()),
            ),
        )
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 78)
    print("汇总结果")
    print("=" * 78)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(f"\n单次运行结果：{RUNS_FILE.resolve()}")
    print(
        "逐步轨迹："
        f"{(OUTPUT_DIR / 'lcc_robustness_trajectory.csv').resolve()}"
    )
    print(f"汇总结果：{SUMMARY_FILE.resolve()}")


if __name__ == "__main__":
    main()
