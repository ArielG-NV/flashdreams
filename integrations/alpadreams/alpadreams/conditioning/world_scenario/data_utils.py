"""World scenario utilities.

This module contains utilities for the world scenario.
"""

from __future__ import annotations

from typing import Any, Final, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from alpadreams.conditioning.world_scenario.data_types import (
    DynamicObject,
    TrafficLightState,
)

# =============================================================================
# Coordinate Frame Conversion (FLU <-> RDF)
# =============================================================================
# FLU: Forward-Left-Up (common robotics convention, used by clients)
# RDF: Right-Down-Forward (OpenCV convention, used internally by renderer)

# Transformation matrix from FLU to OpenCV RDF
FLU_TO_RDF_MATRIX: NDArray[np.float32] = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)

# Transformation matrix from OpenCV RDF to FLU (inverse = transpose for orthonormal)
RDF_TO_FLU_MATRIX: NDArray[np.float32] = FLU_TO_RDF_MATRIX.T.astype(np.float32)

STATIC_HEADING_THRESHOLD: Final = 0.7


def convert_pose_flu_to_rdf(pose_matrix: np.ndarray) -> np.ndarray:
    """
    Convert a 4x4 pose matrix from FLU to RDF coordinates.

    This applies a basis change transformation:
    - Position: p_rdf = R * p_flu
    - Rotation: R_rdf = R * R_flu * R^T

    Args:
        pose_matrix: 4x4 transformation matrix in FLU coordinates.

    Returns:
        4x4 transformation matrix in RDF coordinates.
    """
    result = np.eye(4, dtype=np.float32)

    # Convert position
    result[:3, 3] = FLU_TO_RDF_MATRIX @ pose_matrix[:3, 3]

    # Convert rotation: R_rdf = S * R_flu * S^T (double-sided transformation)
    result[:3, :3] = FLU_TO_RDF_MATRIX @ pose_matrix[:3, :3] @ RDF_TO_FLU_MATRIX

    return result


def convert_pose_rdf_to_flu(pose_matrix: np.ndarray) -> np.ndarray:
    """
    Convert a 4x4 pose matrix from RDF to FLU coordinates.

    This applies a basis change transformation:
    - Position: p_flu = R^T * p_rdf
    - Rotation: R_flu = R^T * R_rdf * R

    Args:
        pose_matrix: 4x4 transformation matrix in RDF coordinates.

    Returns:
        4x4 transformation matrix in FLU coordinates.
    """
    result = np.eye(4, dtype=np.float32)

    # Convert position
    result[:3, 3] = RDF_TO_FLU_MATRIX @ pose_matrix[:3, 3]

    # Convert rotation: R_flu = S^T * R_rdf * S (double-sided transformation)
    result[:3, :3] = RDF_TO_FLU_MATRIX @ pose_matrix[:3, :3] @ FLU_TO_RDF_MATRIX

    return result


def convert_poses_flu_to_rdf(poses: np.ndarray) -> np.ndarray:
    """
    Convert an array of 4x4 pose matrices from FLU to RDF coordinates.

    Args:
        poses: Array of shape [N, 4, 4] in FLU coordinates.

    Returns:
        Array of shape [N, 4, 4] in RDF coordinates.
    """
    return np.stack([convert_pose_flu_to_rdf(p) for p in poses])


def convert_poses_rdf_to_flu(poses: np.ndarray) -> np.ndarray:
    """
    Convert an array of 4x4 pose matrices from RDF to FLU coordinates.

    Args:
        poses: Array of shape [N, 4, 4] in RDF coordinates.

    Returns:
        Array of shape [N, 4, 4] in FLU coordinates.
    """
    return np.stack([convert_pose_rdf_to_flu(p) for p in poses])


def convert_position_flu_to_rdf(position: np.ndarray) -> np.ndarray:
    """
    Convert a 3D position from FLU to RDF coordinates.

    Args:
        position: Position vector [x, y, z] in FLU coordinates.

    Returns:
        Position vector [x, y, z] in RDF coordinates.
    """
    return (FLU_TO_RDF_MATRIX @ position.reshape(3)).astype(np.float32)


def convert_position_rdf_to_flu(position: np.ndarray) -> np.ndarray:
    """
    Convert a 3D position from RDF to FLU coordinates.

    Args:
        position: Position vector [x, y, z] in RDF coordinates.

    Returns:
        Position vector [x, y, z] in FLU coordinates.
    """
    return (RDF_TO_FLU_MATRIX @ position.reshape(3)).astype(np.float32)


def convert_quaternion_flu_to_rdf(quat_xyzw: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion from FLU to RDF coordinates.

    Args:
        quat_xyzw: Quaternion [x, y, z, w] in FLU coordinates (scipy convention).

    Returns:
        Quaternion [x, y, z, w] in RDF coordinates (scipy convention).
    """
    rot_flu = Rotation.from_quat(quat_xyzw)
    rot_matrix_flu = rot_flu.as_matrix()
    rot_matrix_rdf = FLU_TO_RDF_MATRIX @ rot_matrix_flu @ RDF_TO_FLU_MATRIX
    return Rotation.from_matrix(rot_matrix_rdf).as_quat().astype(np.float32)


def convert_quaternion_rdf_to_flu(quat_xyzw: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion from RDF to FLU coordinates.

    Args:
        quat_xyzw: Quaternion [x, y, z, w] in RDF coordinates (scipy convention).

    Returns:
        Quaternion [x, y, z, w] in FLU coordinates (scipy convention).
    """
    rot_rdf = Rotation.from_quat(quat_xyzw)
    rot_matrix_rdf = rot_rdf.as_matrix()
    rot_matrix_flu = RDF_TO_FLU_MATRIX @ rot_matrix_rdf @ FLU_TO_RDF_MATRIX
    return Rotation.from_matrix(rot_matrix_flu).as_quat().astype(np.float32)


def convert_points_flu_to_rdf(points: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert an array of 3D points from FLU to OpenCV RDF coordinates."""

    if points.size == 0:
        return points.astype(np.float32, copy=False)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Points must have shape (N, 3), got {points.shape}")
    return (FLU_TO_RDF_MATRIX @ points.T).T.astype(np.float32)


def convert_quaternions_flu_to_rdf(
    quaternions: NDArray[np.float32],
    *,
    double_sided: bool = False,
) -> NDArray[np.float32]:
    """Convert an array of quaternions from FLU to OpenCV RDF coordinates."""

    if quaternions.size == 0:
        return quaternions.astype(np.float32, copy=False)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError(f"Quaternions must have shape (N, 4), got {quaternions.shape}")

    rotations_flu = Rotation.from_quat(quaternions)
    r_flu = rotations_flu.as_matrix()
    r_rdf = np.einsum("ij,njk->nik", FLU_TO_RDF_MATRIX, r_flu)
    if double_sided:
        r_rdf = np.matmul(r_rdf, FLU_TO_RDF_MATRIX.T)
    quaternions_rdf = Rotation.from_matrix(r_rdf).as_quat().astype(np.float32)
    return quaternions_rdf


def normalize_quaternions(
    quaternions: NDArray[np.float32], eps: float = 1e-6
) -> Tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Normalize quaternions and return mask of valid entries."""

    if quaternions.size == 0:
        mask = np.zeros((0,), dtype=bool)
        return quaternions.astype(np.float32, copy=False), mask

    norms = np.linalg.norm(quaternions, axis=1)
    valid_mask = norms > eps

    normalized = quaternions[valid_mask].copy()
    if normalized.size > 0:
        normalized /= norms[valid_mask][:, None]

    return normalized.astype(np.float32), valid_mask


def fix_static_objects(objects: dict[str, DynamicObject]) -> None:
    """Stabilize static DynamicObjects by averaging pose and dimensions."""

    for _, obj in objects.items():
        if obj.is_moving or obj.centers.size == 0:
            continue

        centers = obj.centers.astype(np.float32, copy=False)
        rotations = Rotation.from_quat(obj.orientations)
        rot_mats = rotations.as_matrix().astype(np.float32)

        front_dirs = rot_mats[:, :, 0]
        front_norms = np.linalg.norm(front_dirs, axis=1, keepdims=True)
        front_dirs = np.divide(
            front_dirs,
            np.maximum(front_norms, 1e-8),
            where=front_norms > 0,
        )

        mean_heading = front_dirs.mean(axis=0)
        heading_norm = np.linalg.norm(mean_heading)
        if heading_norm <= 1e-8:
            continue
        mean_heading /= heading_norm

        valid_idx = np.where((front_dirs @ mean_heading) > STATIC_HEADING_THRESHOLD)[0]
        if valid_idx.size == 0:
            continue

        translation_mean = centers[valid_idx].mean(axis=0)

        front_mean = rot_mats[valid_idx][:, :, 0].mean(axis=0)
        front_mean_norm = np.linalg.norm(front_mean)
        if front_mean_norm <= 1e-8:
            continue
        front_mean /= front_mean_norm

        up_mean = rot_mats[valid_idx][:, :, 2].mean(axis=0)
        up_mean_norm = np.linalg.norm(up_mean)
        if up_mean_norm <= 1e-8:
            continue
        up_mean /= up_mean_norm

        left_mean = np.cross(up_mean, front_mean)
        left_norm = np.linalg.norm(left_mean)
        if left_norm <= 1e-8:
            continue
        left_mean /= left_norm

        # Re-orthogonalize up vector to ensure numerical stability
        up_mean = np.cross(front_mean, left_mean)
        up_mean /= max(np.linalg.norm(up_mean), 1e-8)

        rotation_matrix = np.stack([front_mean, left_mean, up_mean], axis=1)
        mean_quat = Rotation.from_matrix(rotation_matrix).as_quat().astype(np.float32)

        obj.centers = np.repeat(
            translation_mean[None, :], centers.shape[0], axis=0
        ).astype(np.float32)
        obj.orientations = np.repeat(
            mean_quat[None, :], obj.orientations.shape[0], axis=0
        )

        dims = obj.dimensions
        dims_array = np.asarray(dims, dtype=np.float32)
        if dims_array.ndim == 2:
            mean_dims = dims_array.mean(axis=0)
        else:
            mean_dims = dims_array
        obj.dimensions = np.repeat(
            mean_dims.reshape(1, -1), obj.centers.shape[0], axis=0
        ).astype(np.float32)


_TRAFFIC_LIGHT_STATE_BY_KEY: dict[str, TrafficLightState] = {
    state.value.lower(): state for state in TrafficLightState
}


def coerce_traffic_light_state(
    value: Any,
    *,
    feature_id: str | None = None,
    frame_idx: int | None = None,
) -> TrafficLightState:
    """Convert a raw traffic light state value to ``TrafficLightState``.

    Args:
        value: Raw state value (string, enum, or None).
        feature_id: Optional feature identifier for error context.
        frame_idx: Optional frame index for error context.

    Returns:
        TrafficLightState: Normalized traffic light state enum.

    Raises:
        ValueError: If the state is not recognized.
        TypeError: If the value has an unsupported type.
    """

    if isinstance(value, TrafficLightState):
        return value

    if value is None:
        return TrafficLightState.UNKNOWN

    candidate = getattr(value, "value", value)
    if not isinstance(candidate, str):
        raise TypeError(f"Unsupported traffic light state type: {type(value)!r}")

    normalized = candidate.strip().lower()
    if not normalized:
        return TrafficLightState.UNKNOWN

    state = _TRAFFIC_LIGHT_STATE_BY_KEY.get(normalized)
    if state is None:
        expected = ", ".join(sorted(_TRAFFIC_LIGHT_STATE_BY_KEY.keys()))
        context_bits = []
        if feature_id is not None:
            context_bits.append(f"feature_id={feature_id}")
        if frame_idx is not None:
            context_bits.append(f"frame={frame_idx}")
        context = f" ({'; '.join(context_bits)})" if context_bits else ""
        raise ValueError(
            f"Unsupported traffic light state '{candidate}'{context}; expected one of [{expected}]."
        )

    return state


def normalize_traffic_light_state_sequence(
    states: Sequence[Any] | None,
    num_frames: int,
    *,
    feature_id: str | None = None,
) -> list[TrafficLightState]:
    """Normalize a sequence of raw states to a fixed-length enum list.

    Args:
        states: Iterable of raw state values.
        num_frames: Target number of frames for the sequence.
        feature_id: Optional feature identifier for error context.

    Returns:
        List of ``TrafficLightState`` with length ``num_frames``.
    """

    normalized: list[TrafficLightState] = []
    source_states = list(states) if states is not None else []

    for frame_idx in range(num_frames):
        if frame_idx < len(source_states):
            normalized_state = coerce_traffic_light_state(
                source_states[frame_idx],
                feature_id=feature_id,
                frame_idx=frame_idx,
            )
        else:
            normalized_state = TrafficLightState.UNKNOWN
        normalized.append(normalized_state)

    return normalized
