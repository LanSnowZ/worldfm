"""受限版交互式世界探索后端核心逻辑."""

from __future__ import annotations

import io
import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import quote, unquote

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image, UnidentifiedImageError

from modules.depth_selector import ConditionDB
from modules.pano_postprocess import PostProcessResult
from modules.point_renderer import TorchPointCloudRenderer
from modules.transforms_io import scale_K_for_resize
from modules.worldfm_infer import WorldFMTriConditionInprocess
from run_pipeline import (
    DEFAULT_CFG,
    WORLDFM_ROOT,
    setup_external_repos,
    step1_panogen,
    step2_moge_pipeline,
    step3_init,
    step3_render_one,
    step4_init,
    step4_infer_one,
)

LOGGER = logging.getLogger("worldfm.web_demo")
RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
PRESET_PICTURES_DIR = RESOURCE_DIR / "pictures"
PRESET_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

ActionName = Literal[
    "move_forward",
    "move_backward",
    "move_left",
    "move_right",
    "turn_left",
    "turn_right",
    "look_up",
    "look_down",
]


class ExplorerError(RuntimeError):
    """Web Demo 领域内的可预期异常."""

    def __init__(self, code: str, message: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class ExplorerLimits:
    """探索行为限制参数."""

    move_step: float
    turn_step_deg: float
    max_radius: float
    min_pitch_deg: float
    max_pitch_deg: float
    min_request_interval_sec: float
    initial_yaw_deg: float
    initial_pitch_deg: float


@dataclass(frozen=True)
class DemoRuntimeConfig:
    """Web Demo 运行配置."""

    host: str
    port: int
    output_root: Path
    max_sessions: int
    session_ttl_sec: float
    max_upload_bytes: int
    limits: ExplorerLimits


@dataclass
class PresetImage:
    """Web Demo 预设图片元数据."""

    preset_id: str
    title: str
    file_path: Path
    image_url: str


@dataclass
class PoseState:
    """相机位姿状态."""

    x: float
    y: float
    z: float
    yaw_deg: float
    pitch_deg: float

    def position(self) -> np.ndarray:
        """返回世界坐标平移向量."""
        return np.asarray([self.x, self.y, self.z], dtype=np.float64)

    def radius(self) -> float:
        """返回相对初始原点的平移半径."""
        return float(np.linalg.norm(self.position()))

    def as_dict(self) -> dict[str, float]:
        """返回可序列化姿态数据."""
        return {
            "x": float(self.x),
            "y": float(self.y),
            "z": float(self.z),
            "yaw_deg": float(self.yaw_deg),
            "pitch_deg": float(self.pitch_deg),
            "radius": float(self.radius()),
        }

    def to_c2w(self) -> np.ndarray:
        """返回 OpenCV 相机到世界坐标变换矩阵."""
        yaw = math.radians(self.yaw_deg)
        pitch = math.radians(self.pitch_deg)
        right, up, forward = _basis_from_yaw_pitch(yaw, pitch)
        camera_z = (-forward).astype(np.float64)

        transform = np.eye(4, dtype=np.float64)
        transform[:3, 0] = right
        transform[:3, 1] = up
        transform[:3, 2] = camera_z

        axis_flip = np.diag(np.asarray([1.0, -1.0, -1.0, 1.0], dtype=np.float64))
        c2w = axis_flip @ transform @ axis_flip
        c2w[:3, 3] = self.position()
        return c2w


@dataclass
class SceneSession:
    """单个探索 session 的完整运行态."""

    session_id: str
    scene_dir: Path
    input_path: Path
    panorama_image: Image.Image
    postprocess_result: PostProcessResult
    renderer: TorchPointCloudRenderer
    condition_db: ConditionDB
    worldfm_service: WorldFMTriConditionInprocess
    render_config: Any
    worldfm_config: Any
    render_size: int
    camera_K: np.ndarray
    limits: ExplorerLimits
    pose: PoseState
    status: str
    message: str
    latest_frame_rgb: np.ndarray
    latest_frame_png: bytes
    frame_version: int
    last_request_at: float
    updated_at: float
    lock: Any = field(default_factory=threading.Lock, repr=False)
    busy: bool = False

    def touch(self) -> None:
        """刷新最近访问时间."""
        self.updated_at = time.monotonic()

    def set_frame(self, frame_rgb: np.ndarray) -> None:
        """更新最新帧缓存."""
        self.latest_frame_rgb = frame_rgb
        self.latest_frame_png = encode_png(frame_rgb)
        self.frame_version += 1
        self.touch()

    def close(self) -> None:
        """释放 session 占用的运行资源."""
        try:
            self.panorama_image.close()
        except Exception:
            pass
        for attr in ("renderer", "condition_db", "worldfm_service", "postprocess_result"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ExplorerEnvironment:
    """管理运行环境初始化和 session 生命周期."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.runtime = build_runtime_config(cfg)
        self._presets = load_preset_images(PRESET_PICTURES_DIR)
        self._sessions: dict[str, SceneSession] = {}
        self._lock = threading.Lock()
        self._prepare_lock = threading.Lock()
        self._prepared = False

    @property
    def sessions(self) -> dict[str, SceneSession]:
        """返回当前 session 映射."""
        return self._sessions

    def list_presets(self) -> list[PresetImage]:
        """返回可用预设图片列表."""
        return list(self._presets.values())

    def prepare(self) -> None:
        """初始化外部仓库路径和输出目录."""
        with self._prepare_lock:
            if self._prepared:
                return
            self.runtime.output_root.mkdir(parents=True, exist_ok=True)
            gpu_index = int(self.cfg.pipeline.gpu_index)
            if gpu_index >= 0 and torch.cuda.is_available():
                torch.cuda.set_device(gpu_index)
            setup_external_repos(
                hw_path=str(self.cfg.submodules.hw_path),
                moge_path=str(self.cfg.submodules.moge_path),
            )
            self._prepared = True
            LOGGER.info("Web Demo 环境初始化完成")

    def get_preset(self, preset_id: str) -> PresetImage:
        """根据 ID 获取预设图片."""
        preset = self._presets.get(str(preset_id))
        if preset is None:
            raise ExplorerError("preset_not_found", "预设图片不存在, 请重新选择", 404)
        return preset

    def cleanup_expired_sessions(self) -> None:
        """清理超时 session."""
        now = time.monotonic()
        expired: list[str] = []
        with self._lock:
            for session_id, session in self._sessions.items():
                if now - session.updated_at > self.runtime.session_ttl_sec:
                    expired.append(session_id)
            for session_id in expired:
                session = self._sessions.pop(session_id)
                session.close()
        if expired:
            LOGGER.info("已清理 %d 个过期 session", len(expired))

    def clear_all_sessions(self) -> None:
        """关闭所有 session."""
        with self._lock:
            session_ids = list(self._sessions.keys())
            for session_id in session_ids:
                session = self._sessions.pop(session_id)
                session.close()

    def get_session(self, session_id: str) -> SceneSession:
        """获取有效 session."""
        self.cleanup_expired_sessions()
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ExplorerError("session_not_found", "当前 session 已失效, 请重新生成场景", 404)
            session.touch()
            return session

    def create_session(self, image_bytes: bytes, filename: str) -> SceneSession:
        """创建新 session 并完成场景初始化."""
        if len(image_bytes) > self.runtime.max_upload_bytes:
            raise ExplorerError(
                "image_too_large",
                f"上传图片过大, 请控制在 {self.runtime.max_upload_bytes // (1024 * 1024)} MB 以内",
                413,
            )

        session_id, scene_dir, input_path = self._prepare_new_session_dir()
        save_uploaded_image(image_bytes, filename, input_path)
        return self._initialize_session(
            session_id=session_id,
            scene_dir=scene_dir,
            input_path=input_path,
        )

    def create_session_from_preset(self, preset_id: str) -> SceneSession:
        """使用预设图片创建新 session."""
        preset = self.get_preset(preset_id)
        session_id, scene_dir, input_path = self._prepare_new_session_dir()
        save_preset_image(preset.file_path, input_path)
        LOGGER.info("使用预设图片 %s 初始化 session %s", preset.preset_id, session_id)
        return self._initialize_session(
            session_id=session_id,
            scene_dir=scene_dir,
            input_path=input_path,
        )

    def _prepare_new_session_dir(self) -> tuple[str, Path, Path]:
        """为新 session 准备输出目录并执行必要回收."""
        self.prepare()
        self.cleanup_expired_sessions()

        with self._lock:
            while len(self._sessions) >= self.runtime.max_sessions and self._sessions:
                old_session_id, old_session = self._sessions.popitem()
                old_session.close()
                LOGGER.info("为新 session 回收旧 session: %s", old_session_id)

        session_id = uuid.uuid4().hex
        scene_dir = self.runtime.output_root / session_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        input_path = scene_dir / "input.png"
        return session_id, scene_dir, input_path

    def _initialize_session(
        self,
        *,
        session_id: str,
        scene_dir: Path,
        input_path: Path,
    ) -> SceneSession:
        """复用现有 Step 1-4 链路初始化场景."""
        LOGGER.info("开始初始化 session %s", session_id)
        try:
            panorama_image = step1_panogen(
                image_path=str(input_path),
                output_dir=scene_dir,
                cfg=self.cfg,
            )
            panorama_image.save(scene_dir / "panorama.png")

            postprocess_result = step2_moge_pipeline(
                panorama_img=panorama_image,
                output_dir=scene_dir,
                cfg=self.cfg,
            )

            renderer, condition_db, render_config, render_size = step3_init(
                postprocess_result,
                cfg=self.cfg,
            )
            worldfm_service, worldfm_config = build_worldfm_service(self.cfg)

            camera_K = build_default_camera_K(postprocess_result, render_size)
            pose = PoseState(
                x=0.0,
                y=0.0,
                z=0.0,
                yaw_deg=self.runtime.limits.initial_yaw_deg,
                pitch_deg=self.runtime.limits.initial_pitch_deg,
            )
            first_frame = render_frame(
                renderer=renderer,
                condition_db=condition_db,
                postprocess_result=postprocess_result,
                worldfm_service=worldfm_service,
                camera_K=camera_K,
                pose=pose,
                render_config=render_config,
                worldfm_config=worldfm_config,
                render_size=render_size,
            )
        except ExplorerError:
            raise
        except FileNotFoundError as exc:
            raise ExplorerError("model_not_ready", f"模型文件不存在: {exc}", 500) from exc
        except ImportError as exc:
            raise ExplorerError("environment_not_ready", f"运行环境未准备好: {exc}", 500) from exc
        except Exception as exc:
            raise ExplorerError("scene_init_failed", f"场景初始化失败: {exc}", 500) from exc

        now = time.monotonic()
        session = SceneSession(
            session_id=session_id,
            scene_dir=scene_dir,
            input_path=input_path,
            panorama_image=panorama_image,
            postprocess_result=postprocess_result,
            renderer=renderer,
            condition_db=condition_db,
            worldfm_service=worldfm_service,
            render_config=render_config,
            worldfm_config=worldfm_config,
            render_size=render_size,
            camera_K=camera_K,
            limits=self.runtime.limits,
            pose=pose,
            status="ready",
            message="可探索",
            latest_frame_rgb=first_frame,
            latest_frame_png=encode_png(first_frame),
            frame_version=1,
            last_request_at=now,
            updated_at=now,
        )

        with self._lock:
            self._sessions[session_id] = session
        LOGGER.info("session %s 初始化完成", session_id)
        return session

    def apply_action(self, session_id: str, action: ActionName) -> SceneSession:
        """执行一次探索动作并生成新视角."""
        session = self.get_session(session_id)
        with session.lock:
            if session.busy:
                raise ExplorerError("busy", "上一帧仍在生成中, 请稍后再试", 409)

            now = time.monotonic()
            if now - session.last_request_at < session.limits.min_request_interval_sec:
                raise ExplorerError(
                    "rate_limited",
                    f"请求过于频繁, 请至少间隔 {session.limits.min_request_interval_sec:.2f} 秒",
                    429,
                )

            session.busy = True
            session.status = "rendering"
            session.message = "生成下一帧中"
            session.touch()

            try:
                next_pose = apply_action_to_pose(session.pose, action, session.limits)
                frame = render_frame(
                    renderer=session.renderer,
                    condition_db=session.condition_db,
                    postprocess_result=session.postprocess_result,
                    worldfm_service=session.worldfm_service,
                    camera_K=session.camera_K,
                    pose=next_pose,
                    render_config=session.render_config,
                    worldfm_config=session.worldfm_config,
                    render_size=session.render_size,
                )
                session.pose = next_pose
                session.set_frame(frame)
                session.status = "ready"
                session.message = "可探索"
            except ExplorerError as exc:
                if exc.code == "out_of_bounds":
                    session.status = "out_of_bounds"
                    session.message = exc.message
                else:
                    session.status = "error"
                    session.message = exc.message
                raise
            except Exception as exc:
                session.status = "error"
                session.message = f"推理失败: {exc}"
                raise ExplorerError("frame_infer_failed", session.message, 500) from exc
            finally:
                session.busy = False
                session.last_request_at = time.monotonic()
                session.touch()

        return session


def load_demo_config(
    *,
    config_path: str = "",
    host: str = "",
    port: int = 0,
    output_root: str = "",
    gpu_index: Optional[int] = None,
) -> DictConfig:
    """加载 Web Demo 配置."""
    cfg = OmegaConf.create(DEFAULT_CFG)
    if config_path:
        user_cfg = OmegaConf.load(config_path)
        cfg = OmegaConf.merge(cfg, user_cfg)

    overrides: dict[str, Any] = {}
    if host:
        overrides.setdefault("web_demo", {})["host"] = host
    if port > 0:
        overrides.setdefault("web_demo", {})["port"] = int(port)
    if output_root:
        overrides.setdefault("web_demo", {})["output_root"] = output_root
    if gpu_index is not None:
        overrides.setdefault("pipeline", {})["gpu_index"] = int(gpu_index)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


def build_runtime_config(cfg: DictConfig) -> DemoRuntimeConfig:
    """将 OmegaConf 转换为运行时强类型配置."""
    web_cfg = cfg.web_demo
    output_root = Path(str(web_cfg.output_root))
    if not output_root.is_absolute():
        output_root = (WORLDFM_ROOT / output_root).resolve()

    limits = ExplorerLimits(
        move_step=float(web_cfg.move_step),
        turn_step_deg=float(web_cfg.turn_step_deg),
        max_radius=float(web_cfg.max_radius),
        min_pitch_deg=float(web_cfg.min_pitch_deg),
        max_pitch_deg=float(web_cfg.max_pitch_deg),
        min_request_interval_sec=float(web_cfg.min_request_interval_sec),
        initial_yaw_deg=float(web_cfg.initial_yaw_deg),
        initial_pitch_deg=float(web_cfg.initial_pitch_deg),
    )

    return DemoRuntimeConfig(
        host=str(web_cfg.host),
        port=int(web_cfg.port),
        output_root=output_root,
        max_sessions=max(1, int(web_cfg.max_sessions)),
        session_ttl_sec=max(60.0, float(web_cfg.session_ttl_minutes) * 60.0),
        max_upload_bytes=max(1, int(web_cfg.max_upload_mb) * 1024 * 1024),
        limits=limits,
    )


def build_worldfm_service(cfg: DictConfig) -> tuple[WorldFMTriConditionInprocess, Any]:
    """构建可复用的 WorldFM 推理服务."""
    return step4_init(cfg=cfg)


def save_uploaded_image(image_bytes: bytes, filename: str, output_path: Path) -> None:
    """校验并保存上传图片."""
    if not image_bytes:
        raise ExplorerError("invalid_image", "上传图片为空", 400)

    safe_name = unquote(filename or "upload.png").strip()
    try:
        with Image.open(io.BytesIO(image_bytes)) as uploaded:
            rgb = uploaded.convert("RGB")
            rgb.save(output_path, format="PNG")
    except (UnidentifiedImageError, OSError) as exc:
        raise ExplorerError("invalid_image", f"无法解析上传图片 {safe_name}", 400) from exc


def save_preset_image(source_path: Path, output_path: Path) -> None:
    """校验并保存预设图片."""
    safe_name = source_path.name
    try:
        with Image.open(source_path) as preset_image:
            rgb = preset_image.convert("RGB")
            rgb.save(output_path, format="PNG")
    except FileNotFoundError as exc:
        raise ExplorerError("preset_invalid", f"预设图片不存在: {safe_name}", 500) from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise ExplorerError("preset_invalid", f"无法解析预设图片 {safe_name}", 500) from exc


def load_preset_images(preset_dir: Path) -> dict[str, PresetImage]:
    """扫描预设图片目录并构建元数据."""
    presets: dict[str, PresetImage] = {}
    if not preset_dir.exists():
        return presets

    for file_path in sorted(preset_dir.iterdir()):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in PRESET_IMAGE_EXTENSIONS:
            continue

        preset_id = file_path.stem.lower()
        if preset_id in presets:
            raise ValueError(f"预设图片 ID 重复: {preset_id}")

        presets[preset_id] = PresetImage(
            preset_id=preset_id,
            title=file_path.stem,
            file_path=file_path,
            image_url=f"/resources/pictures/{quote(file_path.name)}",
        )

    return presets


def build_default_camera_K(postprocess_result: PostProcessResult, render_size: int) -> np.ndarray:
    """根据条件图相机参数构建默认渲染内参."""
    frames = postprocess_result.transforms.get("frames", [])
    if not frames:
        raise ExplorerError("invalid_transforms", "条件视角数据缺失, 无法构建默认相机", 500)
    first_frame = frames[0]
    base_K = np.asarray(first_frame["K"], dtype=np.float64)
    src_wh = (int(first_frame["width"]), int(first_frame["height"]))
    dst_wh = (int(render_size), int(render_size))
    return scale_K_for_resize(base_K, src_wh=src_wh, dst_wh=dst_wh)


def render_frame(
    *,
    renderer: TorchPointCloudRenderer,
    condition_db: ConditionDB,
    postprocess_result: PostProcessResult,
    worldfm_service: WorldFMTriConditionInprocess,
    camera_K: np.ndarray,
    pose: PoseState,
    render_config: Any,
    worldfm_config: Any,
    render_size: int,
) -> np.ndarray:
    """根据当前位姿渲染一帧图像."""
    render_u8, cond_nearest_rgb = step3_render_one(
        renderer,
        condition_db,
        postprocess_result,
        camera_K,
        pose.to_c2w(),
        rcfg=render_config,
        render_size=render_size,
    )
    return step4_infer_one(
        worldfm_service,
        render_u8,
        cond_nearest_rgb,
        wcfg=worldfm_config,
    )


def encode_png(frame_rgb: np.ndarray) -> bytes:
    """编码当前帧为 PNG."""
    buffer = io.BytesIO()
    Image.fromarray(frame_rgb, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def apply_action_to_pose(pose: PoseState, action: ActionName, limits: ExplorerLimits) -> PoseState:
    """对位姿施加单次用户动作."""
    next_pose = PoseState(
        x=float(pose.x),
        y=float(pose.y),
        z=float(pose.z),
        yaw_deg=float(pose.yaw_deg),
        pitch_deg=float(pose.pitch_deg),
    )

    if action == "turn_left":
        next_pose.yaw_deg = wrap_angle(next_pose.yaw_deg - limits.turn_step_deg)
        return next_pose
    if action == "turn_right":
        next_pose.yaw_deg = wrap_angle(next_pose.yaw_deg + limits.turn_step_deg)
        return next_pose
    if action == "look_up":
        next_pose.pitch_deg = clamp(next_pose.pitch_deg + limits.turn_step_deg, limits.min_pitch_deg, limits.max_pitch_deg)
        return next_pose
    if action == "look_down":
        next_pose.pitch_deg = clamp(next_pose.pitch_deg - limits.turn_step_deg, limits.min_pitch_deg, limits.max_pitch_deg)
        return next_pose

    c2w = pose.to_c2w()
    forward = normalize_horizontal(c2w[:3, 2], fallback=np.asarray([1.0, 0.0, 0.0], dtype=np.float64))
    right = normalize_horizontal(c2w[:3, 0], fallback=np.asarray([0.0, 0.0, -1.0], dtype=np.float64))

    if action == "move_forward":
        delta = forward * limits.move_step
    elif action == "move_backward":
        delta = -forward * limits.move_step
    elif action == "move_left":
        delta = -right * limits.move_step
    elif action == "move_right":
        delta = right * limits.move_step
    else:
        raise ExplorerError("invalid_action", f"不支持的动作: {action}", 400)

    next_pose.x += float(delta[0])
    next_pose.y += float(delta[1])
    next_pose.z += float(delta[2])

    radius = next_pose.radius()
    if radius > limits.max_radius + 1e-6:
        raise ExplorerError(
            "out_of_bounds",
            f"已超出探索边界, 最大半径为 {limits.max_radius:.2f} m, 当前尝试为 {radius:.2f} m",
            400,
        )

    return next_pose


def normalize_horizontal(vector: np.ndarray, *, fallback: np.ndarray) -> np.ndarray:
    """将向量投影到水平面后归一化."""
    projected = np.asarray(vector, dtype=np.float64).copy()
    projected[1] = 0.0
    norm = float(np.linalg.norm(projected))
    if norm <= 1e-8:
        return fallback.astype(np.float64)
    return projected / norm


def clamp(value: float, lower: float, upper: float) -> float:
    """夹紧数值范围."""
    return max(lower, min(upper, value))


def wrap_angle(angle_deg: float) -> float:
    """将角度包裹到 [-180, 180) 区间."""
    wrapped = (angle_deg + 180.0) % 360.0 - 180.0
    if wrapped == -180.0:
        return 180.0
    return wrapped


def _basis_from_yaw_pitch(yaw_rad: float, pitch_rad: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """复用条件图生成中的 yaw/pitch 基向量定义."""
    cyaw = math.cos(yaw_rad)
    syaw = math.sin(yaw_rad)
    cp = math.cos(pitch_rad)
    sp = math.sin(pitch_rad)

    forward = np.asarray([cyaw * cp, sp, syaw * cp], dtype=np.float64)
    forward = forward / max(float(np.linalg.norm(forward)), 1e-8)

    world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(forward, world_up).astype(np.float64)
    right = right / max(float(np.linalg.norm(right)), 1e-8)

    up = np.cross(right, forward).astype(np.float64)
    up = up / max(float(np.linalg.norm(up)), 1e-8)

    return right, up, forward
