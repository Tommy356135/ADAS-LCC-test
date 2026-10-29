import gymnasium as gym
import highway_env
import numpy as np


gym.register_envs(highway_env)


def wrap_angle(angle: float) -> float:
    """将角度归一化到 [-pi, pi)。"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def measure_state(env):
    """读取用于测试评估的环境真实状态。"""
    base_env = env.unwrapped
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
        "longitudinal": float(longitudinal),
        "lateral_error": float(lateral_error),
        "heading_error": float(heading_error),
        "speed": float(vehicle.speed),
    }


def run_probe(action_value: float, steps: int = 10):
    """运行一次固定转向动作的方向探测。"""
    env = gym.make(
        "lane-keeping-v0",
        config={
            "state_noise": 0.0,
            "derivative_noise": 0.0,
        },
    )

    try:
        env.reset(seed=42)

        before = measure_state(env)

        action = np.array(
            [action_value],
            dtype=np.float32,
        )

        for _ in range(steps):
            _, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                break

        after = measure_state(env)

        print(f"\n转向动作: {action_value:+.3f}")
        print("-" * 42)
        print(
            f"初始横向偏差: "
            f"{before['lateral_error']:+.6f}"
        )
        print(
            f"最终横向偏差: "
            f"{after['lateral_error']:+.6f}"
        )
        print(
            f"横向偏差变化: "
            f"{after['lateral_error'] - before['lateral_error']:+.6f}"
        )
        print(
            f"初始航向误差: "
            f"{before['heading_error']:+.6f}"
        )
        print(
            f"最终航向误差: "
            f"{after['heading_error']:+.6f}"
        )
        print(
            f"航向误差变化: "
            f"{after['heading_error'] - before['heading_error']:+.6f}"
        )
        print(
            f"最终车速: {after['speed']:.3f} m/s"
        )

    finally:
        env.close()


def main():
    print("HighwayEnv LCC 转向方向探测")
    print("每次运行 10 步，随机种子固定为 42。")

    for action_value in [-0.05, 0.0, 0.05]:
        run_probe(action_value)


if __name__ == "__main__":
    main()