
"""LCC-FUN-003: Curved-lane tracking diagnostic.

Learning-only HighwayEnv test.
This is not a production ADAS acceptance test.
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
CSV_FILE = OUTPUT_DIR / "lcc_curve_hold.csv"
PLOT_FILE = OUTPUT_DIR / "lcc_curve_hold_analysis.png"

SEED = 42
MAX_STEPS = 60

# Starting position on the sine lane, in its longitudinal coordinates
INITIAL_LONGITUDINAL_M = 5.0
INITIAL_LATERAL_OFFSET_M = 0.0

# Diagnostic reference only, not a frozen acceptance threshold
REFERENCE_TOLERANCE_M = 0.25


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

        # In HighwayEnv 1.12.1, the reset routine first activates
        # a straight connector lane and retains the sine lane in
        # base.lanes. Activate the sine lane explicitly for this test.
        if not base.lanes:
            raise RuntimeError(
                "No subsequent lane was found after environment reset."
            )

        curved_lane = base.lanes[-1]

        if not hasattr(curved_lane, "amplitude"):
            raise RuntimeError(
                "The expected sine lane was not found."
            )

        base.lane = curved_lane

        # This is a standalone curved-lane test, not a test of the
        # connector-to-sine-lane transition.
        base.lanes = []

        lane = base.lane

        # Place the vehicle at the lane center with matching heading.
        base.vehicle.position = lane.position(
            INITIAL_LONGITUDINAL_M,
            INITIAL_LATERAL_OFFSET_M,
        )

        base.vehicle.heading = lane.heading_at(
            INITIAL_LONGITUDINAL_M
        )

        # Update the vehicle lane index to match its new position.
        base.vehicle.lane_index = (
            base.road.network.get_closest_lane_index(
                base.vehicle.position
            )
        )

        # Refresh observation after changing the environment state.
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
            "reward": None,
            "action_valid": True,
            "terminated": False,
            "truncated": False,
        })

        print("=" * 62)
        print("LCC-FUN-003：正弦曲线车道跟踪诊断")
        print("=" * 62)
        print(f"Lane type: {type(lane).__name__}")
        print(f"Lane amplitude: {lane.amplitude:.3f} m")
        print(f"Lane pulsation: {lane.pulsation:.6f} rad/m")
        print(f"Random seed: {SEED}")
        print(
            f"Initial longitudinal position: "
            f"{INITIAL_LONGITUDINAL_M:.3f} m"
        )
        print(
            f"Initial lateral error: "
            f"{initial['lateral_error_m']:+.6f} m"
        )
        print(
            f"Initial heading error: "
            f"{initial['heading_error_rad']:+.6f} rad"
        )
        print(f"Planned steps: {MAX_STEPS}")
        print("-" * 62)

        steps_completed = 0
        ended_early = False

        for step in range(1, MAX_STEPS + 1):

            # Controller input comes from environment observations.
            action = controller.compute_action(observation)

            action_valid = bool(
                env.action_space.contains(action)
            )

            if not action_valid:
                raise ValueError(
                    f"Invalid action at step {step}: {action}"
                )

            _, reward, terminated, truncated, _ = env.step(
                action
            )

            # Refresh observation after the simulator advances.
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
                    f"Non-finite data at step {step}"
                )

            records.append({
                "step": step,
                **current,
                "steering_action": float(action[0]),
                "reward": float(reward),
                "action_valid": action_valid,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            })

            steps_completed += 1

            if terminated or truncated:
                ended_early = step < MAX_STEPS
                break

    finally:
        env.close()

    # Save records. The initial state has no steering command,
    # so it is deliberately excluded from action statistics.
    fieldnames = list(records[0].keys())

    with CSV_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(records)

    steps = np.array(
        [row["step"] for row in records],
        dtype=int,
    )

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

    action_changes = (
        np.abs(np.diff(actions))
        if len(actions) >= 2
        else np.array([], dtype=float)
    )

    outside = np.flatnonzero(
        np.abs(lateral) > REFERENCE_TOLERANCE_M
    )

    if len(outside) == 0:
        settling_step = 0
    elif outside[-1] == len(lateral) - 1:
        settling_step = None
    else:
        settling_step = int(outside[-1] + 1)

    lane_violations = sum(
        not row["within_lane"]
        for row in records
    )

    print("\nDiagnostic results")
    print("-" * 62)
    print(f"Steps completed: {steps_completed}")
    print(f"Ended early: {ended_early}")
    print(f"Records: {len(records)}")
    print(f"Initial lateral error: {lateral[0]:+.6f} m")
    print(f"Final lateral error: {lateral[-1]:+.6f} m")
    print(
        f"Lateral RMSE: "
        f"{np.sqrt(np.mean(lateral ** 2)):.6f} m"
    )
    print(
        f"Mean absolute lateral error: "
        f"{np.mean(np.abs(lateral)):.6f} m"
    )
    print(
        f"Maximum absolute lateral error: "
        f"{np.max(np.abs(lateral)):.6f} m"
    )
    print(
        f"Maximum absolute heading error: "
        f"{np.max(np.abs(heading)):.6f} rad"
    )
    print(
        f"Steering action range: "
        f"[{actions.min():+.6f}, {actions.max():+.6f}]"
    )

    if len(action_changes):
        print(
            f"Maximum adjacent action change: "
            f"{np.max(action_changes):.6f}"
        )

    print(f"Lane-check violations: {lane_violations}")
    print(
        f"Reference band: "
        f"+/- {REFERENCE_TOLERANCE_M:.2f} m"
    )
    print(f"Last settling step: {settling_step}")

    # Draw diagnostic curves.
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    axes[0].plot(steps, lateral, label="Lateral error")
    axes[0].axhline(
        0.0,
        linestyle="--",
        label="Lane center",
    )
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
    axes[0].set_title("Curved-lane LCC: lateral error")
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

    print(f"\nCSV saved: {CSV_FILE.resolve()}")
    print(f"Plot saved: {PLOT_FILE.resolve()}")
    print(
        "This is a diagnostic run; no formal PASS/FAIL "
        "decision has been made."
    )


if __name__ == "__main__":
    main()
