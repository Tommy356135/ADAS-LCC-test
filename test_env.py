
import gymnasium as gym
import highway_env

gym.register_envs(highway_env)


def test_lane_keeping_environment_runs():
    """检查车道保持环境是否能够正常初始化和执行。"""
    env = gym.make("lane-keeping-v0")

    try:
        obs, info = env.reset(seed=42)
        assert obs is not None

        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            assert obs is not None

            if terminated or truncated:
                obs, info = env.reset()

    finally:
        env.close()
