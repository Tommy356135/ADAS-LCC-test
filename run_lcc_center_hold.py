"""LCC-FUN-002: Center-start lane keeping diagnostic.

This is a learning-only HighwayEnv experiment.
It is not a production LCC acceptance test.
"""

import csv
from pathlib import Path

import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt

from src.controller import LCCController, LCCGains, wrap_angle


gym.register_envs(highway_env)

OUTPUT_DIR = Path("outputs")
CSV_FILE = OUTPUT_DIR / "lcc_center_hold.csv"
PLOT_FILE = OUTPUT_DIR / "lcc_center_hold_analysis.png"

SEED = 42
MAX_STEPS = 60
INITIAL_LONGITUDINAL_M = 5.0
INITIAL_LATERAL_OFFSET_M = 0.0

# 仅用于诊断，不在本轮直接判定正式 PASS/FAIL
REFERENCE_TOLERANCE_M = 0.25


def measure_state(base_env):
    """获取真实状态，用于离线评测，不作为控制器输入。"""

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
    steps_completed = 0
    ended_early = False

    try:
        env.reset(seed=SEED)

        base = env.unwrapped
        lane = base.lane

        # 设置车道中心初始位置
        base.vehicle.position = lane.position(
            INITIAL_LONGITUDINAL_M,
            INITIAL_LATERAL_OFFSET_M,
        )

        # 初始航向与该位置的车道方向一致
        base.vehicle.heading = lane.heading_at(
            INITIAL_LONGITUDINAL_M
        )

        # 重新读取观测，确保控制器使用新的初始状态
        observation = base.observation_type.observe()

        controller = LCCController(
            LCCGains(
                k_lateral=0.025,
                k_heading=0.45,
                max_action=1.0,
            )
        )

        initial = measure_state(base)

        records.append({
            "step": 0,
            **initial,
            "steering_action": None,
            "action_valid": True,
            "reward": None,
            "terminated": False,
            "truncated": False,
        })

        print("=" * 58)
        print("LCC-FUN-002：车道中心稳定保持诊断")
        print("=" * 58)
        print(f"随机种子：{SEED}")
        print(f"初始横向偏差：{initial['lateral_error_m']:+.6f} m")
        print(f"初始航向误差：{initial['heading_error_rad']:+.6f} rad")
        print(f"计划执行步数：{MAX_STEPS}")
        print("-" * 58)

        for step in range(1, MAX_STEPS + 1):

            # 控制器只使用约定的环境观测
            action = controller.compute_action(observation)

            action_valid = bool(
                env.action_space.contains(action)
            )

            if not action_valid:
                raise ValueError(
                    f"第 {step} 步动作超出动作空间：{action}"
                )

            _, reward, terminated, truncated, _ = env.step(
                action
            )

            # lane-keeping-v0 在本项目中的处理方式：
            # 仿真更新后重新读取观测供下一个周期使用
            observation = base.observation_type.observe()

            current = measure_state(base)

            if not all(
                np.isfinite(current[key])
                for key in (
                    "longitudinal_m",
                    "lateral_error_m",
                    "heading_error_rad",
                    "speed_mps",
                )
            ):
                raise ValueError(
                    f"第 {step} 步出现无效状态数据"
                )

            records.append({
                "step": step,
                **current,
                "steering_action": float(action[0]),
                "action_valid": action_valid,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            })

            steps_completed += 1

            if terminated or truncated:
                ended_early = step < MAX_STEPS
                break

    finally:
        env.close()

    # 保存逐步测试记录
    fieldnames = list(records[0].keys())

    with CSV_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # 统计指标
    lateral = np.array(
        [row["lateral_error_m"] for row in records],
        dtype=float,
    )

    heading = np.array(
        [row["heading_error_rad"] for row in records],
        dtype=float,
    )

    # 不把第 0 步的初始状态当成真实控制动作
    actions = np.array(
        [
            row["steering_action"]
            for row in records[1:]
        ],
        dtype=float,
    )

    action_changes = (
        np.abs(np.diff(actions))
        if len(actions) >= 2
        else np.array([], dtype=float)
    )

    outside = np.flatnonzero(
        np.abs(lateral) > REFERENCE_TOLERANCE_M
    )

    if len(outside) == 0:
        last_settling_step = 0
    elif outside[-1] == len(lateral) - 1:
        last_settling_step = None
    else:
        last_settling_step = int(outside[-1] + 1)

    lane_violations = sum(
        not row["within_lane"]
        for row in records
    )

    print("\n诊断结果")
    print("-" * 58)
    print(f"实际执行步数：{steps_completed}")
    print(f"是否提前结束：{ended_early}")
    print(f"有效记录数：{len(records)}")
    print(f"初始横向偏差：{lateral[0]:+.6f} m")
    print(f"最终横向偏差：{lateral[-1]:+.6f} m")
    print(
        f"横向误差 RMSE："
        f"{np.sqrt(np.mean(lateral ** 2)):.6f} m"
    )
    print(
        f"平均绝对横向误差："
        f"{np.mean(np.abs(lateral)):.6f} m"
    )
    print(
        f"最大绝对横向误差："
        f"{np.max(np.abs(lateral)):.6f} m"
    )
    print(
        f"最大绝对航向误差："
        f"{np.max(np.abs(heading)):.6f} rad"
    )
    print(
        f"转向动作范围："
        f"[{actions.min():+.6f}, {actions.max():+.6f}]"
    )

    if len(action_changes):
        print(
            f"最大相邻控制动作变化："
            f"{np.max(action_changes):.6f}"
        )

    print(f"车道范围外记录数：{lane_violations}")
    print(
        f"参考误差带：±{REFERENCE_TOLERANCE_M:.2f} m"
    )
    print(f"最后一次进入并保持在参考误差带的步数：{last_settling_step}")

    # 绘制轨迹
    steps = np.array(
        [row["step"] for row in records],
        dtype=int,
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    axes[0].plot(steps, lateral, label="Lateral error")
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
    axes[0].set_ylabel("Lateral error (m)")
    axes[0].set_title("LCC center-hold: lateral error")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(steps, heading)
    axes[1].axhline(0.0, linestyle="--")
    axes[1].set_ylabel("Heading error (rad)")
    axes[1].set_title("Heading error")
    axes[1].grid(True)

    axes[2].plot(steps[1:], actions)
    axes[2].axhline(1.0, linestyle=":")
    axes[2].axhline(-1.0, linestyle=":")
    axes[2].set_xlabel("Simulation step")
    axes[2].set_ylabel("Steering action")
    axes[2].set_title("Steering control output")
    axes[2].grid(True)

    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=150)
    plt.close(fig)

    print(f"\nCSV 已保存：{CSV_FILE.resolve()}")
    print(f"分析图已保存：{PLOT_FILE.resolve()}")
    print("本轮为诊断运行，尚未执行正式 PASS/FAIL 判定。")


if __name__ == "__main__":
    main()