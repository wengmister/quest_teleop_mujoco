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

    # Heuristic: assume home_qpos defines the active robot joints.
    # Everything else (e.g. free joints for objects) is ignored.
    n_robot = 0
    if home_qpos is not None:
        n_robot = len(home_qpos)

    # Sanity check: if n_robot > model.nq, something is wrong, clamp it.
    if n_robot > model.nq:
        n_robot = model.nq
    
    # If home_qpos is not provided or empty, default to full state (not recommended with free extra objects)
    if n_robot == 0:
        n_robot = model.nq

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

        # Zero out non-robot columns (e.g. cube free joint)
        # Note: We assume robot joints are contiguous at the start and correspond 1:1 with nv indices
        # This is true for hinge/slide joints which Piper uses.
        if n_robot < model.nv:
            jac[:, n_robot:] = 0.0

        if skip_tail_joints:
            # Skip the last few joints of the ROBOT (e.g. gripper fingers)
            # Ensure we don't go negative
            start_skip = max(0, n_robot - skip_tail_joints)
            jac[:, start_skip:n_robot] = 0.0

        if home_weight > 0.0 and home_qpos is not None:
             # Only penalize deviation for robot joints
            home = np.asarray(home_qpos, dtype=np.float64)
            scale = math.sqrt(home_weight)
            # Deviation only for the n_robot joints
            err_home = scale * (home - q[:n_robot])
            
            # Jacobian for home term
            # We extend jac to include n_robot rows
            jac_home = np.zeros((n_robot, model.nv))
            # Set diagonal for the robot part
            np.fill_diagonal(jac_home[:n_robot, :n_robot], scale)
            
            if skip_tail_joints:
                start_skip = max(0, n_robot - skip_tail_joints)
                err_home[start_skip:] = 0.0
                jac_home[start_skip:, :] = 0.0 # Zero out rows for skipped joints
                # Note: columns are also effectively handled by identity structure but let's be safe
                jac_home[:, start_skip:n_robot] = 0.0

            err = np.hstack([err, err_home])
            jac = np.vstack([jac, jac_home])

        JJ = jac @ jac.T + damping * np.eye(jac.shape[0])
        dq = jac.T @ np.linalg.solve(JJ, err)
        
        # Zero out non-robot dq again to be safe
        if n_robot < model.nv:
            dq[n_robot:] = 0.0
        
        # Update q using integratePos to handle free joints correctly (even if we don't move them)
        mujoco.mj_integratePos(model, q, dq, 1.0)
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
