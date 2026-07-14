# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from omnidreams.interactive_drive.config import VehicleConfig
from omnidreams.interactive_drive.types import VehicleState, WorldVehicleBBoxTrack


@dataclass(frozen=True)
class CollisionHit:
    track_id: str
    normal_xy: tuple[float, float]
    penetration_m: float
    obstacle_mass_kg: float
    movable: bool


@dataclass
class _DynamicObstacleState:
    offset_xy: np.ndarray
    velocity_xy: np.ndarray
    last_update_us: int | None = None
    last_impact_us: int | None = None


@dataclass(frozen=True)
class OrientedBox2D:
    center_xy: tuple[float, float]
    half_extents_xy: tuple[float, float]
    yaw_rad: float

    @property
    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        c = math.cos(self.yaw_rad)
        s = math.sin(self.yaw_rad)
        return (
            np.array([c, s], dtype=np.float64),
            np.array([-s, c], dtype=np.float64),
        )

    @property
    def center(self) -> np.ndarray:
        return np.array(self.center_xy, dtype=np.float64)

    def projected_radius(self, axis: np.ndarray) -> float:
        axis_x, axis_y = self.axes
        half_x, half_y = self.half_extents_xy
        return float(
            half_x * abs(np.dot(axis, axis_x)) + half_y * abs(np.dot(axis, axis_y))
        )


def ego_box_from_state(state: VehicleState, vehicle: VehicleConfig) -> OrientedBox2D:
    return OrientedBox2D(
        center_xy=(state.x_m, state.y_m),
        half_extents_xy=(vehicle.aabb_length_m * 0.5, vehicle.aabb_width_m * 0.5),
        yaw_rad=state.yaw_rad,
    )


def _yaw_from_quaternion_xyzw(quaternion_xyzw: np.ndarray) -> float:
    x, y, z, w = [float(v) for v in quaternion_xyzw]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _sat_collision(
    a: OrientedBox2D, b: OrientedBox2D
) -> tuple[np.ndarray, float] | None:
    delta = a.center - b.center
    min_overlap = math.inf
    min_axis: np.ndarray | None = None
    for axis in (*a.axes, *b.axes):
        axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
        distance = abs(float(np.dot(delta, axis)))
        overlap = a.projected_radius(axis) + b.projected_radius(axis) - distance
        if overlap <= 0.0:
            return None
        if overlap < min_overlap:
            min_overlap = overlap
            min_axis = axis

    if min_axis is None:
        return None
    if float(np.dot(delta, min_axis)) < 0.0:
        min_axis = -min_axis
    return min_axis, float(min_overlap)


def _obstacle_mass_kg(object_type: str, fallback_kg: float) -> float:
    normalized = object_type.lower()
    if "truck" in normalized or "bus" in normalized:
        return 9000.0
    if "motor" in normalized or "bike" in normalized or "cycl" in normalized:
        return 280.0
    if "ped" in normalized:
        return 90.0
    return fallback_kg


def _is_movable_object(object_type: str) -> bool:
    normalized = object_type.lower()
    return any(kind in normalized for kind in ("car", "truck", "bus", "vehicle"))


class CollisionWorld:
    def __init__(self, tracks: tuple[WorldVehicleBBoxTrack, ...]) -> None:
        self._tracks = tracks
        self._dynamic: dict[str, _DynamicObstacleState] = {
            track.track_id: _DynamicObstacleState(
                offset_xy=np.zeros(2, dtype=np.float64),
                velocity_xy=np.zeros(2, dtype=np.float64),
            )
            for track in tracks
            if _is_movable_object(track.object_type)
        }

    @classmethod
    def from_tracks(
        cls, tracks: tuple[WorldVehicleBBoxTrack, ...]
    ) -> CollisionWorld | None:
        return cls(tracks) if tracks else None

    def _advance_dynamic_state(
        self, track_id: str, timestamp_us: int
    ) -> _DynamicObstacleState | None:
        dynamic = self._dynamic.get(track_id)
        if dynamic is None:
            return None
        if dynamic.last_update_us is None:
            dynamic.last_update_us = timestamp_us
            return dynamic

        dt_s = max(0.0, float(timestamp_us - dynamic.last_update_us) / 1_000_000.0)
        if dt_s > 0.0:
            dynamic.offset_xy += dynamic.velocity_xy * dt_s
            speed = float(np.linalg.norm(dynamic.velocity_xy))
            if speed > 0.0:
                decel = min(speed, 1.2 * dt_s)
                dynamic.velocity_xy *= max(0.0, (speed - decel) / speed)
            dynamic.last_update_us = timestamp_us
        return dynamic

    def _sample_track(
        self, track: WorldVehicleBBoxTrack, timestamp_us: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        sample = track.interpolate_at_timestamp(timestamp_us)
        if sample is None:
            return None
        center, dims_lwh, orientation = sample
        dynamic = self._advance_dynamic_state(track.track_id, timestamp_us)
        if dynamic is not None:
            center = center.copy()
            center[:2] += dynamic.offset_xy.astype(np.float32)
        return center, dims_lwh, orientation

    def sample_track_center(
        self, track_id: str, timestamp_us: int
    ) -> tuple[float, float] | None:
        for track in self._tracks:
            if track.track_id != track_id:
                continue
            sample = self._sample_track(track, timestamp_us)
            if sample is None:
                return None
            center, _, _ = sample
            return float(center[0]), float(center[1])
        return None

    def resolve(
        self, state: VehicleState, vehicle: VehicleConfig, timestamp_us: int
    ) -> tuple[VehicleState, CollisionHit | None]:
        ego = ego_box_from_state(state, vehicle)
        best_hit: CollisionHit | None = None
        best_penetration = 0.0

        for track in self._tracks:
            sample = self._sample_track(track, timestamp_us)
            if sample is None:
                continue
            center, dims_lwh, orientation = sample
            obstacle = OrientedBox2D(
                center_xy=(float(center[0]), float(center[1])),
                half_extents_xy=(float(dims_lwh[0]) * 0.5, float(dims_lwh[1]) * 0.5),
                yaw_rad=_yaw_from_quaternion_xyzw(orientation),
            )
            collision = _sat_collision(ego, obstacle)
            if collision is None:
                continue
            normal, penetration = collision
            if penetration > best_penetration:
                best_penetration = penetration
                best_hit = CollisionHit(
                    track_id=track.track_id,
                    normal_xy=(float(normal[0]), float(normal[1])),
                    penetration_m=penetration,
                    obstacle_mass_kg=_obstacle_mass_kg(
                        track.object_type, vehicle.obstacle_collision_mass_kg
                    ),
                    movable=track.track_id in self._dynamic,
                )

        if best_hit is None:
            return state, None

        normal = np.array(best_hit.normal_xy, dtype=np.float64)
        heading = np.array(
            [math.cos(state.yaw_rad), math.sin(state.yaw_rad)], dtype=np.float64
        )
        signed_speed = float(state.speed_mps)
        velocity = heading * signed_speed
        closing_speed = -float(np.dot(velocity, normal))
        total_mass = max(vehicle.mass_kg + best_hit.obstacle_mass_kg, 1.0)
        obstacle_share = best_hit.obstacle_mass_kg / total_mass
        ego_share = vehicle.mass_kg / total_mass
        speed_after = signed_speed
        dynamic = self._dynamic.get(best_hit.track_id) if best_hit.movable else None
        if closing_speed > 0.0:
            alignment = min(1.0, closing_speed / max(abs(signed_speed), 1e-6))
            loss = (1.0 + vehicle.collision_restitution) * obstacle_share * alignment
            speed_after = signed_speed * max(0.0, 1.0 - loss)
            if dynamic is not None and dynamic.last_impact_us != timestamp_us:
                hit_direction = -normal
                impulse_speed = (
                    (1.0 + vehicle.collision_restitution) * closing_speed * ego_share
                )
                dynamic.velocity_xy += hit_direction * impulse_speed
                dynamic.last_impact_us = timestamp_us

        slop_m = 0.03
        push = max(0.0, best_hit.penetration_m + slop_m)
        ego_push = push * (obstacle_share if dynamic is not None else 1.0)
        if dynamic is not None:
            dynamic.offset_xy += -normal * push * ego_share
        return (
            VehicleState(
                x_m=state.x_m + float(normal[0]) * ego_push,
                y_m=state.y_m + float(normal[1]) * ego_push,
                z_m=state.z_m,
                yaw_rad=state.yaw_rad,
                speed_mps=speed_after,
                steer_rad=state.steer_rad,
                pitch_rad=state.pitch_rad,
                roll_rad=state.roll_rad,
            ),
            best_hit,
        )
