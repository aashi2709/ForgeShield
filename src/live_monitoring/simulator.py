from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd


class MachineSimulator:
    """
    Multi-machine telemetry simulator for the ForgeShield
    research proof of concept.

    Each simulated machine generates sensor readings that
    follow the feature schema required by the trained AI4I
    supervised models.
    """

    def __init__(self, machine_id: str, seed: int = 42):
        self.machine_id = machine_id
        self.rng = np.random.default_rng(seed)

        # Individual machine operating profile.
        self.base_air_temperature = self.rng.normal(300.0, 1.5)
        self.base_temp_diff = self.rng.normal(9.0, 1.0)
        self.base_speed = self.rng.normal(1450.0, 80.0)
        self.base_torque = self.rng.normal(45.0, 5.0)

        self.tool_wear = self.rng.uniform(20.0, 80.0)

        # Each machine begins at a different degradation level.
        self.degradation = self.rng.uniform(0.0, 0.25)

    def generate_reading(self) -> dict:
        """
        Generate one telemetry observation for this machine.
        """

        # Slowly increase machine degradation.
        self.degradation += self.rng.uniform(0.0, 0.015)
        self.degradation = min(self.degradation, 1.0)

        # Tool wear accumulates over time.
        self.tool_wear += self.rng.uniform(0.2, 1.5)
        self.tool_wear = min(self.tool_wear, 200.0)

        # Ambient temperature.
        air_temperature = (
            self.base_air_temperature
            + self.rng.normal(0, 0.8)
            + self.degradation * 3.0
        )

        # Temperature differential increases as degradation grows.
        temperature_differential = (
            self.base_temp_diff
            + self.degradation * 8.0
            + self.rng.normal(0, 0.5)
        )

        process_temperature = (
            air_temperature
            + temperature_differential
        )

        # Rotational speed changes with machine condition.
        rotational_speed = (
            self.base_speed
            + self.rng.normal(0, 35)
            - self.degradation * 100
        )

        # Torque increases under degraded operating conditions.
        torque = (
            self.base_torque
            + self.rng.normal(0, 2.5)
            + self.degradation * 15
        )

        # Mechanical power derived from RPM and torque.
        mechanical_power = (
            rotational_speed
            * torque
            * (2.0 * np.pi / 60.0)
        )

        # AI4I product category.
        # The trained model expects these categories to be
        # one-hot encoded as Type_H, Type_L and Type_M.
        product_type = self.rng.choice(
            ["H", "L", "M"],
            p=[0.15, 0.55, 0.30],
        )

        return {
            "machine_id": self.machine_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),

            # Categorical feature.
            "Type": product_type,

            # AI4I sensor features.
            "Air temperature": round(
                air_temperature, 2
            ),
            "Process temperature": round(
                process_temperature, 2
            ),
            "Rotational speed": round(
                rotational_speed, 2
            ),
            "Torque": round(
                torque, 2
            ),
            "Tool wear": round(
                self.tool_wear, 2
            ),

            # ForgeShield engineered features.
            "Temperature Differential": round(
                temperature_differential, 2
            ),
            "Mechanical Power": round(
                mechanical_power, 2
            ),

            # Used only by the simulator for visualization/debugging.
            "simulation_degradation": round(
                self.degradation, 3
            ),
        }


def create_machine_fleet(
    machine_count: int = 6,
    seed: int = 42,
) -> list[MachineSimulator]:
    """
    Create a deterministic fleet of simulated machines.
    """

    return [
        MachineSimulator(
            machine_id=f"M-{index:03d}",
            seed=seed + index,
        )
        for index in range(1, machine_count + 1)
    ]


def generate_fleet_snapshot(
    fleet: list[MachineSimulator],
) -> pd.DataFrame:
    """
    Generate one telemetry reading from every machine.
    """

    return pd.DataFrame(
        machine.generate_reading()
        for machine in fleet
    )
