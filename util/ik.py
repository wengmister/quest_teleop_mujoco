"""Inverse-kinematics helpers for MuJoCo models."""

import math

import mujoco
import numpy as np

from util.quaternion import quaternion_to_matrix


def solve_position_ik(
    model: mujoco.MjModel,
    workspace: mujoco.MjData,
    site_id: int,
    target_pos: np.ndarray,
    q_init: np.ndarray,
    *,
    max_iters: int = 30,
    tol: float = 1e-4,
    damping: float = 1e-3,
) -> np.ndarray:
    """Levenberg-Marquardt IK for a site position."""
    q = np.asarray(q_init, dtype=np.float64).copy()
    for _ in range(max_iters):
        workspace.qpos[: model.nq] = q
        workspace.qvel[:] = 0.0
        mujoco.mj_forward(model, workspace)
        err = target_pos - workspace.site_xpos[site_id]
        if np.linalg.norm(err) < tol:
            break
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, workspace, jacp, None, site_id)
        JJ = jacp @ jacp.T + damping * np.eye(3)
        dq = jacp.T @ np.linalg.solve(JJ, err)
        q += dq
    return q


def solve_pose_ik(
    model: mujoco.MjModel,
    workspace: mujoco.MjData,
    site_id: int,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    q_init: np.ndarray,
    *,
    max_iters: int = 30,
    tol: float = 1e-4,
    rot_weight: float = 1.0,
    home_qpos: np.ndarray | None = np.array(
        [0.0, 0.9, -0.9, 0.0, 0.4, 0.0, 0.0, 0.0], dtype=np.float64
    ),
    home_weight: float = 0.01,
    skip_tail_joints: int = 2,
    damping: float = 1e-3,
) -> np.ndarray:
    """Levenberg-Marquardt IK for a site pose (position + orientation)."""
    q = np.asarray(q_init, dtype=np.float64).copy()
    target_rot = np.asarray(quaternion_to_matrix(target_quat), dtype=np.float64)
    for _ in range(max_iters):
        workspace.qpos[: model.nq] = q
        workspace.qvel[:] = 0.0
        mujoco.mj_forward(model, workspace)
        current_pos = workspace.site_xpos[site_id]
        current_rot = workspace.site_xmat[site_id].reshape(3, 3)

        err_pos = target_pos - current_pos
        rot_err = _rotation_error(target_rot, current_rot)
        err = np.hstack([err_pos, rot_weight * rot_err])
        if np.linalg.norm(err) < tol:
            break

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, workspace, jacp, jacr, site_id)
        jac = np.vstack([jacp, rot_weight * jacr])
        if skip_tail_joints:
            jac[:, -skip_tail_joints:] = 0.0
        if home_weight > 0.0 and home_qpos is not None:
            home = np.asarray(home_qpos, dtype=np.float64)[: model.nq]
            scale = math.sqrt(home_weight)
            err_home = scale * (home - q[: model.nq])
            jac_home = scale * np.eye(model.nv)
            if skip_tail_joints:
                err_home[-skip_tail_joints:] = 0.0
                jac_home[:, -skip_tail_joints:] = 0.0
            err = np.hstack([err, err_home])
            jac = np.vstack([jac, jac_home])
        JJ = jac @ jac.T + damping * np.eye(jac.shape[0])
        dq = jac.T @ np.linalg.solve(JJ, err)
        if skip_tail_joints:
            dq[-skip_tail_joints:] = 0.0
        q += dq
    return q


def _rotation_error(target_rot: np.ndarray, current_rot: np.ndarray) -> np.ndarray:
    r_err = target_rot @ current_rot.T
    trace = np.trace(r_err)
    cos_angle = (trace - 1.0) * 0.5
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    if angle < 1e-8:
        return np.zeros(3)
    axis = np.array(
        [
            r_err[2, 1] - r_err[1, 2],
            r_err[0, 2] - r_err[2, 0],
            r_err[1, 0] - r_err[0, 1],
        ],
        dtype=np.float64,
    )
    axis /= 2.0 * np.sin(angle)
    return axis * angle
