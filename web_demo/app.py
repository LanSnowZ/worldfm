"""FastAPI Web Demo 应用."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import Body, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from web_demo.explorer import (
    ActionName,
    ExplorerEnvironment,
    ExplorerError,
    SceneSession,
    load_demo_config,
)

LOGGER = logging.getLogger("worldfm.web_demo")
STATIC_DIR = Path(__file__).resolve().parent / "static"


class MoveRequest(BaseModel):
    """单次探索动作请求."""

    action: ActionName = Field(..., description="探索动作名称")


class PoseResponse(BaseModel):
    """相机位姿响应."""

    x: float
    y: float
    z: float
    yaw_deg: float
    pitch_deg: float
    radius: float


class LimitsResponse(BaseModel):
    """交互限制参数."""

    move_step: float
    turn_step_deg: float
    max_radius: float
    min_pitch_deg: float
    max_pitch_deg: float
    min_request_interval_ms: int


class SessionResponse(BaseModel):
    """Session 状态响应."""

    session_id: str
    status: str
    message: str
    frame_url: Optional[str] = None
    frame_version: int = 0
    busy: bool = False
    pose: Optional[PoseResponse] = None
    limits: Optional[LimitsResponse] = None


def create_app(
    *,
    config_path: str = "",
    host: str = "",
    port: int = 0,
    output_root: str = "",
    gpu_index: Optional[int] = None,
) -> FastAPI:
    """创建 FastAPI 应用."""
    cfg = load_demo_config(
        config_path=config_path,
        host=host,
        port=port,
        output_root=output_root,
        gpu_index=gpu_index,
    )
    environment = ExplorerEnvironment(cfg)

    app = FastAPI(title="WorldFM Restricted Explorer", version="0.1.0")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.state.environment = environment
    app.state.runtime_config = environment.runtime

    @app.exception_handler(ExplorerError)
    def handle_explorer_error(_: Request, exc: ExplorerError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "status": "error",
                "message": exc.message,
                "code": exc.code,
            },
        )

    @app.exception_handler(Exception)
    def handle_unknown_error(_: Request, exc: Exception) -> JSONResponse:
        LOGGER.exception("未处理异常: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"服务端异常: {exc}",
                "code": "internal_error",
            },
        )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/runtime")
    def runtime_info() -> dict[str, Any]:
        runtime = app.state.runtime_config
        return {
            "host": runtime.host,
            "port": runtime.port,
            "max_sessions": runtime.max_sessions,
            "session_ttl_sec": runtime.session_ttl_sec,
        }

    @app.post("/api/sessions", response_model=SessionResponse)
    def create_session(request: Request, payload: bytes = Body(..., media_type="application/octet-stream")) -> SessionResponse:
        filename = request.headers.get("x-filename", "upload.png")
        session = app.state.environment.create_session(payload, filename)
        return session_to_response(session)

    @app.get("/api/sessions/{session_id}", response_model=SessionResponse)
    def get_session(session_id: str) -> SessionResponse:
        session = app.state.environment.get_session(session_id)
        return session_to_response(session)

    @app.get("/api/sessions/{session_id}/frame")
    def get_frame(session_id: str) -> Response:
        session = app.state.environment.get_session(session_id)
        if not session.latest_frame_png:
            raise ExplorerError("frame_not_ready", "当前帧尚未生成完成", 404)
        return Response(
            content=session.latest_frame_png,
            media_type="image/png",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/api/sessions/{session_id}/actions", response_model=SessionResponse)
    def apply_action(session_id: str, move: MoveRequest) -> SessionResponse:
        session = app.state.environment.apply_action(session_id, move.action)
        return session_to_response(session)

    @app.on_event("shutdown")
    def shutdown_event() -> None:
        app.state.environment.clear_all_sessions()

    return app


def session_to_response(session: SceneSession) -> SessionResponse:
    """将 session 运行态转换为响应结构."""
    limits = session.limits
    return SessionResponse(
        session_id=session.session_id,
        status=session.status,
        message=session.message,
        frame_url=f"/api/sessions/{session.session_id}/frame?v={session.frame_version}",
        frame_version=session.frame_version,
        busy=bool(session.busy),
        pose=PoseResponse(**session.pose.as_dict()),
        limits=LimitsResponse(
            move_step=limits.move_step,
            turn_step_deg=limits.turn_step_deg,
            max_radius=limits.max_radius,
            min_pitch_deg=limits.min_pitch_deg,
            max_pitch_deg=limits.max_pitch_deg,
            min_request_interval_ms=int(round(limits.min_request_interval_sec * 1000)),
        ),
    )
