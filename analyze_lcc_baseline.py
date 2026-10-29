from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_FILE = Path("outputs/lcc_baseline.csv")
OUTPUT_FILE = Path("outputs/lcc_baseline_analysis.png")

# 仅用于观察曲线的参考带，并非已批准的验收阈值
REFERENCE_TOLERANCE_M = 0.25


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"找不到测试数据：{INPUT_FILE.resolve()}"
        )

    df = pd.read_csv(INPUT_FILE)
    df = df.sort_values("step").reset_index(drop=True)

    required_columns = [
        "step",
        "lateral_error_m",
        "heading_error_rad",
        "steering_action",
        "within_lane",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"缺少必要字段：{missing}")

    step = df["step"].to_numpy(dtype=int)
    lateral = df["lateral_error_m"].to_numpy(dtype=float)
    heading = df["heading_error_rad"].to_numpy(dtype=float)
    action = df["steering_action"].to_numpy(dtype=float)

    # 检查数据有效性
    numeric_data = np.column_stack([lateral, heading, action])
    finite_mask = np.isfinite(numeric_data).all(axis=1)

    print("=" * 58)
    print("LCC 基线测试轨迹分析")
    print("=" * 58)
    print(f"输入文件：{INPUT_FILE.resolve()}")
    print(f"记录行数：{len(df)}")
    print(f"有效数值记录：{finite_mask.sum()}")
    print(f"无效数值记录：{(~finite_mask).sum()}")

    if not finite_mask.all():
        print("警告：存在无效数值，请先检查对应记录。")

    if len(df) < 2:
        raise ValueError("记录数量不足，无法分析轨迹。")

    # 横向误差指标
    initial_error = lateral[0]
    final_error = lateral[-1]

    rmse = np.sqrt(np.mean(lateral ** 2))
    mae = np.mean(np.abs(lateral))
    max_abs_lateral = np.max(np.abs(lateral))

    if abs(initial_error) > 1e-12:
        reduction_pct = (
            1 - abs(final_error) / abs(initial_error)
        ) * 100
    else:
        reduction_pct = np.nan

    # 航向误差
    max_abs_heading = np.max(np.abs(heading))

    # 控制动作
    min_action = np.min(action)
    max_action = np.max(action)
    max_action_change = np.max(np.abs(np.diff(action)))

    saturation_count = np.sum(np.abs(action) >= 0.999)

    # 横向误差符号变化次数，用于检查是否越过中心
    nonzero_signs = np.sign(lateral)
    nonzero_signs = nonzero_signs[nonzero_signs != 0]

    zero_crossings = (
        np.sum(nonzero_signs[1:] != nonzero_signs[:-1])
        if len(nonzero_signs) >= 2
        else 0
    )

    # 观察最后一次离开参考误差带后的时间步
    outside = np.flatnonzero(
        np.abs(lateral) > REFERENCE_TOLERANCE_M
    )

    if len(outside) == 0:
        settling_step = int(step[0])
    elif outside[-1] + 1 < len(step):
        settling_step = int(step[outside[-1] + 1])
    else:
        settling_step = None

    lane_out_count = int(
        (~df["within_lane"].astype(bool)).sum()
    )

    print("\n横向误差")
    print("-" * 58)
    print(f"初始横向误差：{initial_error:+.4f} m")
    print(f"最终横向误差：{final_error:+.4f} m")
    print(f"RMSE：{rmse:.4f} m")
    print(f"平均绝对误差：{mae:.4f} m")
    print(f"最大绝对横向误差：{max_abs_lateral:.4f} m")
    print(f"最终误差绝对值相对初始值的降低比例：{reduction_pct:.2f}%")

    print("\n稳定性观察")
    print("-" * 58)
    print(f"最大绝对航向误差：{max_abs_heading:.4f} rad")
    print(f"横向误差正负号变化次数：{zero_crossings}")
    print(f"参考误差带：±{REFERENCE_TOLERANCE_M:.2f} m")
    print(f"最后一次进入并保持在参考误差带的步数：{settling_step}")
    print(f"车道范围外记录数：{lane_out_count}")

    print("\n控制动作")
    print("-" * 58)
    print(f"最小动作：{min_action:+.4f}")
    print(f"最大动作：{max_action:+.4f}")
    print(f"相邻记录间最大动作变化：{max_action_change:.4f}")
    print(f"接近动作饱和的记录数：{saturation_count}")

    # 绘制轨迹，帮助识别收敛、过冲和振荡
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    axes[0].plot(step, lateral, label="Lateral error")
    axes[0].axhline(0, linestyle="--", label="Lane center")
    axes[0].axhline(
        REFERENCE_TOLERANCE_M,
        linestyle=":",
        label="Reference band (not acceptance limit)",
    )
    axes[0].axhline(
        -REFERENCE_TOLERANCE_M,
        linestyle=":",
    )
    axes[0].set_xlabel("Simulation step")
    axes[0].set_ylabel("Lateral error (m)")
    axes[0].set_title("LCC lateral error")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(step, heading)
    axes[1].axhline(0, linestyle="--")
    axes[1].set_xlabel("Simulation step")
    axes[1].set_ylabel("Heading error (rad)")
    axes[1].set_title("LCC heading error")
    axes[1].grid(True)

    axes[2].plot(step, action)
    axes[2].axhline(1, linestyle=":")
    axes[2].axhline(-1, linestyle=":")
    axes[2].set_xlabel("Simulation step")
    axes[2].set_ylabel("Steering action")
    axes[2].set_title("LCC control output")
    axes[2].grid(True)

    fig.tight_layout()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=150)
    plt.close(fig)

    print(f"\n分析图已保存：{OUTPUT_FILE.resolve()}")
    print("提示：本次分析用于诊断，不直接判定正式测试 PASS/FAIL。")


if __name__ == "__main__":
    main()