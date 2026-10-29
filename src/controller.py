
"""Learning-only LCC feedback controller for HighwayEnv."""

from dataclasses import dataclass

import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class LCCGains:
    """Controller parameters; initial values require validation."""

    k_lateral: float = 0.025
    k_heading: float = 0.45
    max_action: float = 1.0


class LCCController:
    """Simple feedback controller using lateral and heading errors."""

    def __init__(self, gains: LCCGains | None = None):
        self.gains = gains or LCCGains()

    def compute_action(self, observation: dict) -> np.ndarray:
        """
        Calculate a normalized steering action.

        HighwayEnv lane-keeping observation:
        state[0]: vehicle lateral-coordinate state
        state[1]: vehicle heading
        reference_state[0]: target lateral-coordinate reference
        reference_state[1]: road reference heading
        """
        if "state" not in observation:
            raise ValueError("Observation is missing 'state'.")

        if "reference_state" not in observation:
            raise ValueError("Observation is missing 'reference_state'.")

        state = np.asarray(
            observation["state"], dtype=float
        ).reshape(-1)

        reference = np.asarray(
            observation["reference_state"], dtype=float
        ).reshape(-1)

        if state.size != 4 or reference.size != 4:
            raise ValueError(
                "Expected state and reference_state to contain 4 values."
            )

        if not (
            np.all(np.isfinite(state))
            and np.all(np.isfinite(reference))
        ):
            raise ValueError("Observation contains NaN or infinity.")

        # Error convention confirmed by the steering-direction probe.
        lateral_error = float(state[0] - reference[0])
        heading_error = wrap_angle(
            float(state[1] - reference[1])
        )

        # Negative lateral error should initially produce
        # a positive steering action in this environment.
        raw_action = (
            -self.gains.k_lateral * lateral_error
            - self.gains.k_heading * heading_error
        )

        action = float(
            np.clip(
                raw_action,
                -self.gains.max_action,
                self.gains.max_action,
            )
        )

        return np.array([action], dtype=np.float32)
