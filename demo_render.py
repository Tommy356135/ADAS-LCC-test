import time
import traceback
from types import MethodType

import gymnasium as gym
import highway_env
import pygame

from highway_env.envs.common.graphics import EventHandler


def install_close_handlers(env):
    """
    为 HighwayEnv 的窗口增加关闭按钮和 Esc 键退出功能。

    适配 lane-keeping-v0 当前的构造接口：
    通过环境自身的 viewer 处理显示和事件，而不是在外部
    重复读取 Pygame 事件队列。
    """
    viewer = env.viewer

    if viewer is None:
        raise RuntimeError(
            "仿真窗口尚未初始化，请检查环境的渲染配置。"
        )

    def handle_events_with_escape(self):
        """处理窗口事件，并增加 Esc 退出功能。"""
        for event in pygame.event.get():

            # 点击窗口右上角 X
            # 或按下键盘 Esc
            if (
                event.type == pygame.QUIT
                or (
                    event.type == pygame.KEYDOWN
                    and event.key == pygame.K_ESCAPE
                )
            ):
                self.env.close()
                return

            # 其余事件继续交给 HighwayEnv 原有逻辑处理
            self.sim_surface.handle_event(event)

            if self.env.action_type:
                EventHandler.handle_event(
                    self.env.action_type,
                    event,
                )

    # 替换当前 viewer 的事件处理函数
    viewer.handle_events = MethodType(
        handle_events_with_escape,
        viewer,
    )


def main():
    env = None
    error_occurred = False

    try:
        # 注册 HighwayEnv 环境
        gym.register_envs(highway_env)

        # 注意：
        # lane-keeping-v0 不接受 render_mode 构造参数，
        # 因此这里不能写 render_mode="human"。
        env = gym.make("lane-keeping-v0")

        base_env = env.unwrapped

        # 配置窗口渲染
        base_env.render_mode = "human"
        base_env.config["offscreen_rendering"] = False
        base_env.config["real_time_rendering"] = True

        # 重置环境。
        # 此时会根据 render_mode 创建并初始化窗口。
        obs, info = env.reset(seed=42)

        # 确认窗口已创建
        if base_env.viewer is None:
            raise RuntimeError(
                "窗口创建失败，请检查 Pygame 和渲染配置。"
            )

        # 必须在窗口初始化后安装事件处理函数
        install_close_handlers(base_env)

        print("=" * 50)
        print("HighwayEnv 车道保持环境已启动")
        print("点击窗口右上角 X：退出")
        print("按 Esc：退出")
        print("在终端按 Ctrl+C：退出")
        print("当前车辆使用随机动作，仅用于验证窗口功能")
        print("=" * 50)

        step_count = 0

        # 持续运行，直到用户关闭窗口或按 Esc
        while not base_env.done:

            # 随机动作只用于检查环境显示与事件处理。
            # 后续会替换为真正的 LCC 控制器。
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(
                action
            )

            # 主动渲染当前状态并处理窗口事件
            env.render()

            step_count += 1

            # 窗口关闭后，base_env.done 会变为 True
            if base_env.done:
                break

            # 这里只处理仿真 episode 的结束，
            # 不把窗口关闭和 episode 结束混为一谈。
            if terminated or truncated:
                obs, info = env.reset()

                # reset 后可能会重新创建或刷新 viewer，
                # 因此再次安装关闭事件处理函数。
                if base_env.viewer is not None:
                    install_close_handlers(base_env)

        print(f"仿真结束，共执行 {step_count} 步。")

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在退出仿真。")

    except Exception:
        error_occurred = True
        print("\n程序运行出错，完整错误信息如下：")
        traceback.print_exc()

    finally:
        # 无论正常退出还是发生异常，都尝试释放资源
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

        print("仿真环境已清理。")

    # 仅在程序发生异常时暂停，方便查看报错
    if error_occurred:
        try:
            input("请检查以上错误信息，按 Enter 结束程序...")
        except EOFError:
            pass


if __name__ == "__main__":
    main()