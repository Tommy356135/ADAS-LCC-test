
import importlib.metadata
import gymnasium as gym
import highway_env

# 注册 HighwayEnv 提供的驾驶环境
gym.register_envs(highway_env)

def main():
    print("HighwayEnv:", importlib.metadata.version("highway-env"))
    print("Gymnasium:", gym.__version__)

    # 先不启动图形界面，检查环境是否能正常运行
    env = gym.make("lane-keeping-v0")

    try:
        obs, info = env.reset(seed=42)

        print("环境创建成功")
        print("初始观测类型:", type(obs).__name__)
        print("观测形状:", getattr(obs, "shape", None))
        print("动作空间:", env.action_space)

        for step in range(10):
            # 随机动作仅用于安装检查，不是有效的 LCC 控制器
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            print(
                f"step={step + 1}, "
                f"reward={reward:.3f}, "
                f"terminated={terminated}, "
                f"truncated={truncated}"
            )

            if terminated or truncated:
                obs, info = env.reset()

        print("仿真步骤执行成功")

    finally:
        env.close()

if __name__ == "__main__":
    main()
