from __future__ import annotations

import argparse
import socket
import time
from pathlib import Path

import mujoco
import numpy as np
from mujoco import viewer

from util.ik import solve_position_ik
from util.quaternion import (
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_euler_xyz,
    transform_vr_to_robot_pose,
)
from util.udp_socket import parse_right_wrist_pose


def _default_scene_path() -> Path:
    return Path(__file__).resolve().parent / "scene" / "scene_piper.xml"


def _apply_initial_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    qpos = np.array([0.0, 0.9, -0.9, 0.0, 0.4, 0.0, 0.0], dtype=np.float64)
    n = min(model.nq, qpos.shape[0])
    data.qpos[:n] = qpos[:n]
    n_ctrl = min(model.nu, n)
    if n_ctrl:
        data.ctrl[:n_ctrl] = qpos[:n_ctrl]
    mujoco.mj_forward(model, data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Teleop the piper with wrist residuals.")
    parser.add_argument(
        "--scene",
        default=str(_default_scene_path()),
        help="Path to a MuJoCo XML scene file.",
    )
    parser.add_argument("--port", type=int, default=9000, help="UDP port to listen on.")
    parser.add_argument(
        "--site",
        default="piper_ee_site",
        help="End-effector site name.",
    )
    parser.add_argument(
        "--position-scale",
        type=float,
        default=1.0,
        help="Scale for wrist position residuals.",
    )
    args = parser.parse_args()

    xml_path = Path(args.scene).expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    ik_data = mujoco.MjData(model)

    _apply_initial_pose(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, args.site)
    if site_id == -1 and args.site.startswith("piper_"):
        site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, args.site[len("piper_") :]
        )
    if site_id == -1:
        raise ValueError(f"Site '{args.site}' not found in model.")

    mujoco.mj_forward(model, data)
    initial_site_pos = data.site_xpos[site_id].copy()
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "piper_base_link")
    if base_body_id == -1:
        base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    base_xmat = None
    if base_body_id != -1:
        base_xmat = data.xmat[base_body_id].reshape(3, 3).copy()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", args.port))
    sock.setblocking(False)

    initial_wrist_position = None
    initial_wrist_quaternion = None
    target_position = initial_site_pos.copy()
    last_log_time = time.time()
    latest_residual = None
    latest_euler_residual = None

    with viewer.launch_passive(model, data) as vis:
        while vis.is_running():
            try:
                packet, _ = sock.recvfrom(1024)
            except BlockingIOError:
                packet = None

            if packet is not None:
                message = packet.decode("utf-8", errors="ignore")
                wrist_pose = parse_right_wrist_pose(message)
                if wrist_pose is not None:
                    wrist_position = (wrist_pose[0], wrist_pose[1], wrist_pose[2])
                    wrist_quaternion = (
                        wrist_pose[3],
                        wrist_pose[4],
                        wrist_pose[5],
                        wrist_pose[6],
                    )
                    robot_position, robot_quaternion = transform_vr_to_robot_pose(
                        wrist_position, wrist_quaternion
                    )
                    if initial_wrist_position is None:
                        initial_wrist_position = robot_position
                        initial_wrist_quaternion = robot_quaternion
                    else:
                        residual = np.array(
                            [
                                robot_position[0] - initial_wrist_position[0],
                                robot_position[1] - initial_wrist_position[1],
                                robot_position[2] - initial_wrist_position[2],
                            ],
                            dtype=np.float64,
                        )
                        if base_xmat is not None:
                            residual = base_xmat @ residual
                        target_position = initial_site_pos + args.position_scale * residual

                        relative_quaternion = quaternion_multiply(
                            robot_quaternion,
                            quaternion_inverse(initial_wrist_quaternion),
                        )
                        euler_residual = quaternion_to_euler_xyz(
                            relative_quaternion[0],
                            relative_quaternion[1],
                            relative_quaternion[2],
                            relative_quaternion[3],
                        )
                        latest_residual = residual
                        latest_euler_residual = euler_residual

            now = time.time()
            if (
                latest_residual is not None
                and latest_euler_residual is not None
                and now - last_log_time > 1.0
            ):
                print(
                    f"Wrist residual (xyz): {latest_residual.tolist()} "
                    f"euler: {list(latest_euler_residual)}"
                )
                last_log_time = now

            q_sol = solve_position_ik(
                model, ik_data, site_id, target_position, data.qpos[: model.nq]
            )
            data.qpos[: model.nq] = q_sol
            n_ctrl = min(model.nu, q_sol.shape[0])
            if n_ctrl:
                data.ctrl[:n_ctrl] = q_sol[:n_ctrl]
            mujoco.mj_step(model, data)
            vis.sync()

            sleep_time = model.opt.timestep
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
