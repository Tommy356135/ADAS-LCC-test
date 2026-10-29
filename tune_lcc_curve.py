
"""LCC curved-lane controller gain comparison.

Learning-only diagnostic experiment for HighwayEnv.
No formal acceptance thresholds are applied here.
"""

import csv
from pathlib import Path

import gymnasium as gym
import highway_env
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.controller import LCCController, LCCGains, wrap_angle


gym.register_envs(highway_env)

OUTPUT_DIR = Path("outputs")

SUMMARY_FILE = OUTPUT_DIR / "lcc_curve_gain_sweep_summary.csv"
TRAJECTORY_FILE = OUTPUT_DIR / "lcc_curve_gain_sweep_trajectory.csv"
PLOT_FILE = OUTPUT_DIR / "lcc_curve_gain_sweep.png"

SEED = 42
MAX_STEPS = 60
INITIAL_LONGITUDINAL_M = 5.0
INITIAL_LATERAL_OFFSET_M = 0.0

# Format: (case name, lateral gain, heading gain)
GAIN_CASES = [
    ("baseline", 0.025, 0.45),
    ("lateral_x2", 0.050, 0.45),
    ("lateral_x3_2", 0.080, 0.45),
    ("heading_0_65", 0.025, 0.65),
    ("lateral_0_05_heading_0_65", 0.050, 0.65),
    ("lateral_0_08_heading_0_65", 0.080, 0.65),
]

# Diagnostic reference band only; not a formal acceptance threshold.
REFERENCE_TOLERANCE_M = 0.25


def measure_state(base_env):
    """Read ground-truth values for evaluation."""

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
    """Calculate comparable metrics for one controller setting."""

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

    tail_size = min(10, len(lateral))
    final_tail = lateral[-tail_size:]

    return {
        "initial_lateral_error_m": float(lateral[0]),
        "final_lateral_error_m": float(lateral[-1]),
        "final_abs_lateral_error_m": float(abs(lateral[-1])),
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
        "max_adjacent_action_change": float(
            np.max(np.abs(np.diff(actions)))
        ) if len(actions) > 1 else 0.0,
        "lane_check_violations": int(
            sum(not row["within_lane"] for row in records)
        ),
        "settling_step_025m": settling_step,
        "final_10_within_025m": bool(
            np.all(np.abs(final_tail) <= REFERENCE_TOLERANCE_M)
        ),
    }


def run_one_case(case_name, k_lateral, k_heading):
    """Run one repeatable curved-lane experiment."""

    env = gym.make(
        "lane-keeping-v0",
        config={
            "state_noise": 0.0,
            "derivative_noise": 0.0,
        },
    )

    records = []

    try:
        env.reset(seed=SEED)
        base = env.unwrapped

        if not base.lanes:
            raise RuntimeError(
                "No sine lane available after environment reset."
            )

        # The environment initially activates a connector lane.
        # This experiment deliberately tests the sine lane directly.
        curved_lane = base.lanes[-1]

        if not hasattr(curved_lane, "amplitude"):
            raise RuntimeError(
                "Expected a sine lane, but none was found."
            )

        base.lane = curved_lane
        base.lanes = []

        lane = base.lane

        base.vehicle.position = lane.position(
            INITIAL_LONGITUDINAL_M,
            INITIAL_LATERAL_OFFSET_M,
        )

        base.vehicle.heading = lane.heading_at(
            INITIAL_LONGITUDINAL_M
        )

        base.vehicle.lane_index = (
            base.road.network.get_closest_lane_index(
                base.vehicle.position
            )
        )

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
            "case": case_name,
            "step": 0,
            "k_lateral": k_lateral,
            "k_heading": k_heading,
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
                    f"Invalid action in {case_name}, step {step}: "
                    f"{action}"
                )

            _, reward, terminated, truncated, _ = env.step(action)

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
                    f"Non-finite measurement in "
                    f"{case_name}, step {step}"
                )

            records.append({
                "case": case_name,
                "step": step,
                "k_lateral": k_lateral,
                "k_heading": k_heading,
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

    summary = {
        "case": case_name,
        "k_lateral": k_lateral,
        "k_heading": k_heading,
        "steps_completed": steps_completed,
        "ended_early": ended_early,
        **metrics,
    }

    return summary, records


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []
    all_records = []

    print("=" * 78)
    print("LCC 正弦曲线车道控制器参数对比")
    print("=" * 78)
    print("所有用例使用相同随机种子、初始位置和车道配置。")
    print("本次只进行诊断对比，不自动判定 PASS/FAIL。")
    print("-" * 78)

    for case_name, k_lateral, k_heading in GAIN_CASES:
        summary, records = run_one_case(
            case_name,
            k_lateral,
            k_heading,
        )

        summaries.append(summary)
        all_records.extend(records)

        print(
            f"{case_name:32s} "
            f"final_abs={summary['final_abs_lateral_error_m']:.4f} m, "
            f"RMSE={summary['lateral_rmse_m']:.4f} m, "
            f"max_abs={summary['max_abs_lateral_error_m']:.4f} m, "
            f"lane_violations={summary['lane_check_violations']}"
        )

    summary_df = pd.DataFrame(summaries)
    trajectory_df = pd.DataFrame(all_records)

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    trajectory_df.to_csv(
        TRAJECTORY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # Plot lateral-error trajectories for all parameter settings.
    fig, axes = plt.subplots(2, 1, figsize=(11, 9))

    for case_name, group in trajectory_df.groupby("case"):
        axes[0].plot(
            group["step"],
            group["lateral_error_m"],
            label=case_name,
        )

        axes[1].plot(
            group["step"].iloc[1:],
            group["steering_action"].iloc[1:],
            label=case_name,
        )

    axes[0].axhline(0.0, linestyle="--", label="Lane center")
    axes[0].axhline(
        REFERENCE_TOLERANCE_M,
        linestyle=":",
        label="Diagnostic reference band",
    )
    axes[0].axhline(
        -REFERENCE_TOLERANCE_M,
        linestyle=":",
    )
    axes[0].set_xlabel("Simulation step")
    axes[0].set_ylabel("Lateral error (m)")
    axes[0].set_title("Lateral error comparison")
    axes[0].legend(fontsize=8)
    axes[0].grid(True)

    axes[1].axhline(0.0, linestyle="--")
    axes[1].set_xlabel("Simulation step")
    axes[1].set_ylabel("Steering action")
    axes[1].set_title("Steering command comparison")
    axes[1].legend(fontsize=8)
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=150)
    plt.close(fig)

    print("\n汇总表：")
    print(
        summary_df[
            [
                "case",
                "final_abs_lateral_error_m",
                "lateral_rmse_m",
                "max_abs_lateral_error_m",
                "max_abs_heading_error_rad",
                "lane_check_violations",
            ]
        ].to_string(index=False)
    )

    print(f"\nSummary saved: {SUMMARY_FILE.resolve()}")
    print(f"Trajectories saved: {TRAJECTORY_FILE.resolve()}")
    print(f"Comparison plot saved: {PLOT_FILE.resolve()}")


if __name__ == "__main__":
    main()
