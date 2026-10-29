
"""First diagnostic LCC run: no GUI, deterministic configuration."""

import csv
from pathlib import Path

import gymnasium as gym
import highway_env
import numpy as np

from src.controller import LCCController, LCCGains, wrap_angle


gym.register_envs(highway_env)

OUTPUT_DIR = Path("outputs")
OUTPUT_FILE = OUTPUT_DIR / "lcc_baseline.csv"

SEED = 42
MAX_STEPS = 60
INITIAL_LONGITUDINAL = 5.0
INITIAL_LATERAL_OFFSET = -4.0


def measure_true_state(base_env):
    """Measure ground-truth metrics for evaluation only."""
    vehicle = base_env.vehicle
    lane = base_env.lane

    longitudinal, lateral_error = lane.local_coordinates(
        vehicle.position
    )

    lane_heading = lane.heading_at(longitudinal)
    heading_error = wrap_angle(
        vehicle.heading - lane_heading
    )

    vehicle_state = np.asarray(
        vehicle.state, dtype=float
    ).reshape(-1)

    return {
        "longitudinal_m": float(longitudinal),
        "lateral_error_m": float(lateral_error),
        "heading_error_rad": float(heading_error),
        "speed_mps": float(vehicle.speed),
        "steering_angle_rad": float(vehicle_state[4]),
        "yaw_rate_rad_s": float(vehicle_state[5]),
        "within_lane": bool(lane.on_lane(vehicle.position)),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

        # Move the initial position earlier along the same straight
        # lane, so the test has enough road length before the segment
        # transition. This is a test-specific initial condition.
        lane = base.lane
        base.vehicle.position = lane.position(
            INITIAL_LONGITUDINAL,
            INITIAL_LATERAL_OFFSET,
        )
        base.vehicle.heading = lane.heading_at(
            INITIAL_LONGITUDINAL
        )

        controller = LCCController(
            LCCGains(
                k_lateral=0.025,
                k_heading=0.45,
                max_action=1.0,
            )
        )

        # Read the current observation after setting initial conditions.
        observation = base.observation_type.observe()

        initial = measure_true_state(base)
        records.append({
            "step": 0,
            **initial,
            "steering_action": 0.0,
            "reward": np.nan,
            "terminated": False,
            "truncated": False,
        })

        print("LCC 基础控制器试运行")
        print(f"随机种子: {SEED}")
        print(f"初始纵向位置: {INITIAL_LONGITUDINAL:.2f} m")
        print(f"初始横向偏差: {initial['lateral_error_m']:.4f} m")
        print(f"计划执行步数: {MAX_STEPS}")
        print("-" * 55)

        for step in range(1, MAX_STEPS + 1):
            action = controller.compute_action(observation)

            if not env.action_space.contains(action):
                raise ValueError(
                    f"Invalid action at step {step}: {action}"
                )

            _, reward, terminated, truncated, _ = env.step(action)

            # lane-keeping-v0 returns a pre-simulation observation.
            # Refresh it after the simulator has advanced.
            observation = base.observation_type.observe()

            measurement = measure_true_state(base)

            if not all(
                np.isfinite(measurement[k])
                for k in (
                    "longitudinal_m",
                    "lateral_error_m",
                    "heading_error_rad",
                    "speed_mps",
                    "steering_angle_rad",
                    "yaw_rate_rad_s",
                )
            ):
                raise ValueError(
                    f"Non-finite measurement at step {step}"
                )

            records.append({
                "step": step,
                **measurement,
                "steering_action": float(action[0]),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            })

            if terminated or truncated:
                print(
                    f"Environment ended at step {step}: "
                    f"terminated={terminated}, truncated={truncated}"
                )
                break

    finally:
        env.close()

    # Save the raw results.
    fieldnames = list(records[0].keys())

    with OUTPUT_FILE.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # Summarize results. No formal PASS/FAIL threshold is applied yet.
    lateral = np.array([
        row["lateral_error_m"] for row in records
    ], dtype=float)

    heading = np.array([
        row["heading_error_rad"] for row in records
    ], dtype=float)

    actions = np.array([
        row["steering_action"] for row in records[1:]
    ], dtype=float)

    initial_abs = abs(lateral[0])
    final_abs = abs(lateral[-1])

    print("\n试运行结果")
    print("-" * 55)
    print(f"记录行数: {len(records)}")
    print(f"初始横向偏差: {lateral[0]:+.4f} m")
    print(f"最终横向偏差: {lateral[-1]:+.4f} m")
    print(f"横向偏差 RMSE: {np.sqrt(np.mean(lateral**2)):.4f} m")
    print(f"平均绝对横向偏差: {np.mean(np.abs(lateral)):.4f} m")
    print(f"最大绝对航向误差: {np.max(np.abs(heading)):.4f} rad")
    print(f"转向动作范围: [{actions.min():+.4f}, {actions.max():+.4f}]")
    print(
        "车道范围外的记录数:",
        sum(not row["within_lane"] for row in records),
    )

    if final_abs < initial_abs:
        print("初步观察: 最终横向偏差绝对值小于初始值。")
    else:
        print("初步观察: 横向偏差未改善或出现退化，需要检查控制器。")

    print("\n注意：这是诊断性试运行，尚未冻结正式验收阈值。")
    print(f"原始结果已保存: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
