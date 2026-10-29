
from pathlib import Path

import pandas as pd

INPUT_FILE = Path("outputs/lcc_robustness_runs.csv")


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"找不到逐次运行记录：{INPUT_FILE.resolve()}\n"
            "请确认鲁棒性测试脚本已经成功运行。"
        )

    df = pd.read_csv(INPUT_FILE)

    # 检查关键字段，避免对不完整的数据作判断
    required = [
        "controller",
        "seed",
        "initial_offset_m",
        "initial_speed_mps",
        "noise_name",
        "lateral_rmse_m",
        "final_abs_lateral_error_m",
        "max_abs_lateral_error_m",
        "final_10_within_025m",
        "lane_check_violations",
        "ended_early",
        "finite_data",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要字段：{missing}")

    candidate = df[df["controller"] == "candidate_B"].copy()

    if candidate.empty:
        raise ValueError("未找到 candidate_B 的运行记录。")

    print("=" * 76)
    print("候选 B：最差运行条件分析")
    print("=" * 76)

    print("\n一、按最大横向偏差排序的前 10 次运行")
    worst = candidate.sort_values(
        "max_abs_lateral_error_m",
        ascending=False,
    )

    columns = [
        "noise_name",
        "seed",
        "initial_speed_mps",
        "initial_offset_m",
        "lateral_rmse_m",
        "final_abs_lateral_error_m",
        "max_abs_lateral_error_m",
        "final_10_within_025m",
    ]

    print(
        worst[columns]
        .head(10)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n二、按车速和初始偏差统计")
    grouped = (
        candidate
        .groupby(
            ["noise_name", "initial_speed_mps", "initial_offset_m"],
            as_index=False,
        )
        .agg(
            runs=("controller", "size"),
            mean_rmse_m=("lateral_rmse_m", "mean"),
            mean_final_abs_m=(
                "final_abs_lateral_error_m",
                "mean",
            ),
            worst_max_abs_m=(
                "max_abs_lateral_error_m",
                "max",
            ),
            stable_tail_runs=(
                "final_10_within_025m",
                "sum",
            ),
        )
        .sort_values(
            ["noise_name", "mean_rmse_m"],
            ascending=[True, False],
        )
    )

    print(
        grouped.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n三、执行有效性检查")
    print(
        "提前结束次数：",
        int(candidate["ended_early"].astype(bool).sum()),
    )
    print(
        "无效数据次数：",
        int((~candidate["finite_data"].astype(bool)).sum()),
    )
    print(
        "车道检查异常总数：",
        int(candidate["lane_check_violations"].sum()),
    )
    print(
        "最后 10 条记录全部位于 ±0.25 m 的次数：",
        int(candidate["final_10_within_025m"].astype(bool).sum()),
        "/",
        len(candidate),
    )

    output = Path("outputs/lcc_candidate_b_worst_cases.csv")
    worst.head(10).to_csv(
        output,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"\n最差 10 次运行已保存：{output.resolve()}")


if __name__ == "__main__":
    main()
