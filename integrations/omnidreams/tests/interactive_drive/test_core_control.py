# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import numpy as np
import pytest
from omnidreams.interactive_drive.config import ChunkConfig, VehicleConfig
from omnidreams.interactive_drive.input.keyboard import command_from_snapshot
from omnidreams.interactive_drive.simulation.collision import CollisionWorld
from omnidreams.interactive_drive.simulation.ego_vehicle_kinematics import (
    integrate_vehicle,
    sample_chunk_trajectory,
)
from omnidreams.interactive_drive.types import (
    ControlSnapshot,
    DriverCommand,
    VehicleState,
    WorldVehicleBBoxTrack,
)



def _bbox_track(
    track_id: str, object_type: str, x_m: float = 3.0
) -> WorldVehicleBBoxTrack:
    return WorldVehicleBBoxTrack(
        track_id=track_id,
        object_type=object_type,
        timestamps_us=np.array([0, 100_000, 1_000_000], dtype=np.int64),
        centers_world=np.array(
            [[x_m, 0.0, 0.0], [x_m, 0.0, 0.0], [x_m, 0.0, 0.0]],
            dtype=np.float32,
        ),
        dimensions_lwh=np.array(
            [[4.0, 2.0, 1.6], [4.0, 2.0, 1.6], [4.0, 2.0, 1.6]],
            dtype=np.float32,
        ),
        orientations_xyzw=np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        max_extrapolation_us=0.0,
    )


def test_command_from_snapshot_maps_keyboard_state() -> None:
    snapshot = ControlSnapshot(pressed={"w", "a"})
    command = command_from_snapshot(snapshot)
    assert command.throttle == 1.0
    assert command.brake == 0.0
    assert command.steer == 1.0


def test_sample_chunk_trajectory_advances_pose_and_time() -> None:
    state = VehicleState(
        x_m=0.0, y_m=0.0, z_m=0.0, yaw_rad=0.0, speed_mps=0.0, steer_rad=0.0
    )
    snapshot = ControlSnapshot(pressed={"w"})
    command = command_from_snapshot(snapshot)

    chunk = sample_chunk_trajectory(
        start_state=state,
        start_timestamp_us=1000,
        command=command,
        chunk_size=4,
        chunk_config=ChunkConfig(fps=10, initial_chunk_frames=2, chunk_frames=2),
        vehicle_config=VehicleConfig(),
        ground_snapper=None,
    )

    assert list(chunk.timestamps_us) == [1000, 101000, 201000, 301000]
    assert chunk.rig_poses_world.shape == (4, 4, 4)
    assert chunk.boundary_state_after_chunk.x_m > 0.0
    assert chunk.boundary_state_after_chunk.speed_mps > 0.0


def test_manual_brake_overrides_throttle_to_a_stop() -> None:
    """Gas + brake pressed together must bleed speed toward a stop.

    Regression for the HUD/ego mismatch: the manual-control branch used to
    give throttle priority, so holding both pedals built speed. Brake now
    wins, matching the HUD's speed readout and real-car behaviour.
    """
    vehicle = VehicleConfig()
    state = VehicleState(
        x_m=0.0, y_m=0.0, z_m=0.0, yaw_rad=0.0, speed_mps=10.0, steer_rad=0.0
    )
    both = DriverCommand(throttle=1.0, brake=1.0, manual_control=True)

    decelerating = integrate_vehicle(state, both, dt_s=0.1, vehicle=vehicle)
    assert decelerating.speed_mps < state.speed_mps

    # Held long enough, the vehicle comes to rest rather than creeping.
    for _ in range(200):
        state = integrate_vehicle(state, both, dt_s=0.1, vehicle=vehicle)
    assert state.speed_mps == pytest.approx(0.0, abs=1e-6)


def test_manual_throttle_only_still_accelerates() -> None:
    """Throttle without brake keeps its acceleration behaviour."""
    vehicle = VehicleConfig()
    state = VehicleState(
        x_m=0.0, y_m=0.0, z_m=0.0, yaw_rad=0.0, speed_mps=0.0, steer_rad=0.0
    )
    throttle = DriverCommand(throttle=1.0, brake=0.0, manual_control=True)

    advanced = integrate_vehicle(state, throttle, dt_s=0.1, vehicle=vehicle)
    assert advanced.speed_mps > state.speed_mps


def test_vehicle_mass_changes_acceleration_response() -> None:
    sedan = VehicleConfig(mass_kg=1500.0)
    truck = VehicleConfig(mass_kg=6000.0)
    state = VehicleState(
        x_m=0.0, y_m=0.0, z_m=0.0, yaw_rad=0.0, speed_mps=0.0, steer_rad=0.0
    )
    command = DriverCommand(throttle=1.0)

    sedan_next = integrate_vehicle(state, command, dt_s=1.0, vehicle=sedan)
    truck_next = integrate_vehicle(state, command, dt_s=1.0, vehicle=truck)

    assert sedan_next.speed_mps > truck_next.speed_mps


def test_passive_drag_slows_vehicle_without_input() -> None:
    vehicle = VehicleConfig(drag_mps2=0.7, rolling_resistance_mps2=0.2)
    state = VehicleState(
        x_m=0.0, y_m=0.0, z_m=0.0, yaw_rad=0.0, speed_mps=8.0, steer_rad=0.0
    )

    slowed = integrate_vehicle(state, DriverCommand(), dt_s=1.0, vehicle=vehicle)

    assert 0.0 < slowed.speed_mps < state.speed_mps


def test_collision_with_obstacle_separates_and_reduces_speed() -> None:
    track = _bbox_track("parked-car", "Car")
    start = VehicleState(
        x_m=0.0, y_m=0.0, z_m=0.0, yaw_rad=0.0, speed_mps=5.0, steer_rad=0.0
    )

    chunk = sample_chunk_trajectory(
        start_state=start,
        start_timestamp_us=0,
        command=DriverCommand(),
        chunk_size=1,
        chunk_config=ChunkConfig(fps=10),
        vehicle_config=VehicleConfig(),
        ground_snapper=None,
        collision_world=CollisionWorld.from_tracks((track,)),
    )

    final = chunk.boundary_state_after_chunk
    assert final.x_m < 0.0
    assert final.speed_mps < start.speed_mps


def test_collision_moves_movable_obstacle_trajectory() -> None:
    track = _bbox_track("parked-car", "Car")
    collision_world = CollisionWorld.from_tracks((track,))
    assert collision_world is not None
    start = VehicleState(
        x_m=0.5, y_m=0.0, z_m=0.0, yaw_rad=0.0, speed_mps=8.0, steer_rad=0.0
    )

    _, hit = collision_world.resolve(start, VehicleConfig(), timestamp_us=0)
    shifted_now = collision_world.sample_track_center("parked-car", 0)
    shifted_later = collision_world.sample_track_center("parked-car", 100_000)

    assert hit is not None
    assert hit.movable
    assert shifted_now is not None
    assert shifted_later is not None
    assert shifted_now[0] > 3.0
    assert shifted_later[0] > shifted_now[0]


def test_collision_moves_truck_less_than_car() -> None:
    start = VehicleState(
        x_m=0.5, y_m=0.0, z_m=0.0, yaw_rad=0.0, speed_mps=8.0, steer_rad=0.0
    )
    car_world = CollisionWorld.from_tracks((_bbox_track("car", "Car"),))
    truck_world = CollisionWorld.from_tracks((_bbox_track("truck", "Truck"),))
    assert car_world is not None
    assert truck_world is not None

    car_world.resolve(start, VehicleConfig(), timestamp_us=0)
    truck_world.resolve(start, VehicleConfig(), timestamp_us=0)
    car_center = car_world.sample_track_center("car", 100_000)
    truck_center = truck_world.sample_track_center("truck", 100_000)

    assert car_center is not None
    assert truck_center is not None
    assert car_center[0] - 3.0 > truck_center[0] - 3.0


def test_integrate_vehicle_accumulates_steering_gradually() -> None:
    vehicle = VehicleConfig(
        max_steer_rad=0.5, steer_rate_rad_per_s=1.0, steer_return_rate_rad_per_s=0.5
    )
    state = VehicleState(
        x_m=0.0, y_m=0.0, z_m=0.0, yaw_rad=0.0, speed_mps=0.0, steer_rad=0.0
    )

    state = integrate_vehicle(
        state, DriverCommand(steer=1.0), dt_s=0.1, vehicle=vehicle
    )
    assert state.steer_rad == pytest.approx(0.1)

    state = integrate_vehicle(
        state, DriverCommand(steer=1.0), dt_s=0.1, vehicle=vehicle
    )
    assert state.steer_rad == pytest.approx(0.2)


def test_integrate_vehicle_recenters_steering_after_release() -> None:
    vehicle = VehicleConfig(
        max_steer_rad=0.5, steer_rate_rad_per_s=1.0, steer_return_rate_rad_per_s=0.5
    )
    state = VehicleState(
        x_m=0.0, y_m=0.0, z_m=0.0, yaw_rad=0.0, speed_mps=0.0, steer_rad=0.2
    )

    released = integrate_vehicle(
        state, DriverCommand(steer=0.0), dt_s=0.1, vehicle=vehicle
    )
    assert released.steer_rad == pytest.approx(0.15)

    released = integrate_vehicle(
        released, DriverCommand(steer=0.0), dt_s=0.3, vehicle=vehicle
    )
    assert released.steer_rad == pytest.approx(0.0)
