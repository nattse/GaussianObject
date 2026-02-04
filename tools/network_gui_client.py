import os
import argparse
import json
import math
import socket
import struct
import time
from dataclasses import dataclass
from typing import Tuple
import cv2
import numpy as np


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed while receiving data")
        buf.extend(chunk)
    return bytes(buf)


def _send_json(sock: socket.socket, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    sock.sendall(struct.pack("<I", len(data)))
    sock.sendall(data)


def _save_ppm(path: str, width: int, height: int, rgb_bytes: bytes) -> None:
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    with open(path, "wb") as f:
        f.write(header)
        f.write(rgb_bytes)


def _projection_matrix(znear: float, zfar: float, fovx: float, fovy: float) -> np.ndarray:
    tan_half_fov_y = math.tan(fovy / 2.0)
    tan_half_fov_x = math.tan(fovx / 2.0)
    top = tan_half_fov_y * znear
    bottom = -top
    right = tan_half_fov_x * znear
    left = -right

    p = np.zeros((4, 4), dtype=np.float32)
    z_sign = 1.0  # matches utils.graphics_utils.getProjectionMatrix
    p[0, 0] = 2.0 * znear / (right - left)
    p[1, 1] = 2.0 * znear / (top - bottom)
    p[0, 2] = (right + left) / (right - left)
    p[1, 2] = (top + bottom) / (top - bottom)
    p[3, 2] = z_sign
    p[2, 2] = z_sign * zfar / (zfar - znear)
    p[2, 3] = -(zfar * znear) / (zfar - znear)
    return p


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up2 = np.cross(right, forward)

    w2c = np.eye(4, dtype=np.float32)
    w2c[0, :3] = right
    w2c[1, :3] = up2
    w2c[2, :3] = forward
    w2c[0, 3] = -np.dot(right, eye)
    w2c[1, 3] = -np.dot(up2, eye)
    w2c[2, 3] = -np.dot(forward, eye)
    return w2c


@dataclass
class CameraParams:
    width: int
    height: int
    fovx: float
    fovy: float
    znear: float
    zfar: float
    cam_dist: float
    yaw_deg: float
    pitch_deg: float
    target: Tuple[float, float, float]

    def build_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        yaw = math.radians(self.yaw_deg)
        pitch = math.radians(self.pitch_deg)
        eye = np.array(
            [
                self.cam_dist * math.cos(pitch) * math.cos(yaw),
                self.cam_dist * math.sin(pitch),
                self.cam_dist * math.cos(pitch) * math.sin(yaw),
            ],
            dtype=np.float32,
        )
        target = np.array(self.target, dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        w2c = _look_at(eye, target, up)
        view_t = w2c.T
        proj = _projection_matrix(self.znear, self.zfar, self.fovx, self.fovy)
        proj_t = proj.T
        view_proj_t = view_t @ proj_t
        return view_t, view_proj_t


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal client for gaussian_renderer.network_gui")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7007)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--fovx", type=float, default=1.6)
    parser.add_argument("--fovy", type=float, default=1.06)
    parser.add_argument("--znear", type=float, default=0.01)
    parser.add_argument("--zfar", type=float, default=100.0)
    parser.add_argument("--cam-dist", type=float, default=2.0)
    parser.add_argument("--yaw-deg", type=float, default=45.0)
    parser.add_argument("--pitch-deg", type=float, default=-15.0)
    parser.add_argument("--target-x", type=float, default=0.0)
    parser.add_argument("--target-y", type=float, default=0.0)
    parser.add_argument("--target-z", type=float, default=0.0)
    parser.add_argument("--eye-x", type=float, default=float("nan"))
    parser.add_argument("--eye-y", type=float, default=float("nan"))
    parser.add_argument("--eye-z", type=float, default=float("nan"))
    parser.add_argument("--scaling-modifier", type=float, default=1.0)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--keep-alive", action="store_true")
    parser.add_argument("--shs-python", action="store_true")
    parser.add_argument("--rot-scale-python", action="store_true")
    parser.add_argument("--out", default="splatting_progress.mp4")
    parser.add_argument("--live", action="store_true", help="Continuously request frames.")
    parser.add_argument("--fps", type=float, default=10.0, help="Request rate when --live is set.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames when --live is set (0 = no limit).")
    parser.add_argument("--show", action="store_true", help="Display frames with OpenCV if available.")
    parser.add_argument("--no-image", action="store_true", help="Send a zero-res request (no image data returned).")
    args = parser.parse_args()

    if args.no_image:
        width = 0
        height = 0
        view_t = np.eye(4, dtype=np.float32)
        view_proj_t = np.eye(4, dtype=np.float32)
    else:
        width = args.width
        height = args.height
        target = (args.target_x, args.target_y, args.target_z)
        cam = CameraParams(
            width=width,
            height=height,
            fovx=args.fovx,
            fovy=args.fovy,
            znear=args.znear,
            zfar=args.zfar,
            cam_dist=args.cam_dist,
            yaw_deg=args.yaw_deg,
            pitch_deg=args.pitch_deg,
            target=target,
        )
        view_t, view_proj_t = cam.build_matrices()
        # network_gui.receive flips Y/Z columns; pre-flip to preserve intended pose.
        view_t[:, 1] *= -1.0
        view_t[:, 2] *= -1.0
        view_proj_t[:, 1] *= -1.0
        view_proj_t[:, 2] *= -1.0
        if not (math.isnan(args.eye_x) or math.isnan(args.eye_y) or math.isnan(args.eye_z)):
            eye = np.array([args.eye_x, args.eye_y, args.eye_z], dtype=np.float32)
            target = np.array(target, dtype=np.float32)
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            view_t = _look_at(eye, target, up).T
            proj = _projection_matrix(args.znear, args.zfar, args.fovx, args.fovy)
            view_proj_t = view_t @ proj.T
            view_t[:, 1] *= -1.0
            view_t[:, 2] *= -1.0
            view_proj_t[:, 1] *= -1.0
            view_proj_t[:, 2] *= -1.0

    payload = {
        "resolution_x": width,
        "resolution_y": height,
        "fov_y": args.fovy,
        "fov_x": args.fovx,
        "z_near": args.znear,
        "z_far": args.zfar,
        "train": int(args.train),
        "shs_python": int(args.shs_python),
        "rot_scale_python": int(args.rot_scale_python),
        "keep_alive": int(args.keep_alive),
        "scaling_modifier": float(args.scaling_modifier),
        "view_matrix": view_t.flatten().tolist(),
        "view_projection_matrix": view_proj_t.flatten().tolist(),
    }

    use_cv2 = True
    frame_idx = 0
    interval = 1.0 / max(args.fps, 0.1) if args.live else 0.0
    interval_ms = int(interval * 1000)

    vw = cv2.VideoWriter(
        args.out + '.mp4',
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )
    
    try:
        with socket.create_connection((args.host, args.port)) as sock:
            print('sending first payload...')
            while True:
                _send_json(sock, payload)

                if width > 0 and height > 0:
                    expected = width * height * 3
                    rgb_bytes = _recv_exact(sock, expected)
                    if use_cv2:
                        img = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape((height, width, 3))
                        img = img[:, :, ::-1]  # RGB -> BGR
                        vw.write(img)
                        if frame_idx == 0:
                            print(f'Wrote first video frame...')
                        #cv2.imshow("network_gui", img)
                        #if cv2.waitKey(1) & 0xFF == ord("q"):
                        #keypress = cv2.waitKey(interval_ms) & 0xFF
                        #if keypress == ord("q"):
                        #    break
                    else:
                        _save_ppm(args.out, width, height, rgb_bytes)
                        if not args.live:
                            print(f"Wrote image: {args.out} ({width}x{height})")

                verify_len = struct.unpack("<I", _recv_exact(sock, 4))[0]
                verify = _recv_exact(sock, verify_len).decode("ascii", errors="replace")
                if not args.live:
                    print(f"verify: {verify}")
                frame_idx += 1
                if args.max_frames and frame_idx >= args.max_frames:
                    break
                if not args.live:
                    break
    except KeyboardInterrupt:
        print(f'Stopped by keyboard exit')
    finally:
        print("Releasing resources...")
        vw.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    pid = os.getpid()
    print(f'This process opened PID {pid}')
    main()
