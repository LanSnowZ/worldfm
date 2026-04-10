#!/usr/bin/env python3
"""受限版交互式世界探索 Web Demo 启动入口."""

from __future__ import annotations

import argparse
import logging

import uvicorn

from web_demo.app import create_app


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器."""
    parser = argparse.ArgumentParser(
        description="Launch the restricted interactive WorldFM web demo.",
    )
    parser.add_argument("--config", type=str, default="", help="额外配置文件路径")
    parser.add_argument("--host", type=str, default="", help="监听地址")
    parser.add_argument("--port", type=int, default=0, help="监听端口")
    parser.add_argument("--output_root", type=str, default="", help="Web Demo 输出目录")
    parser.add_argument("--gpu_index", type=int, default=None, help="CUDA 设备编号")
    parser.add_argument("--log_level", type=str, default="info", help="uvicorn 日志级别")
    return parser


def main() -> int:
    """启动 uvicorn 服务."""
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(name)s: %(message)s",
    )

    app = create_app(
        config_path=args.config,
        host=args.host,
        port=args.port,
        output_root=args.output_root,
        gpu_index=args.gpu_index,
    )
    runtime = app.state.runtime_config

    uvicorn.run(
        app,
        host=runtime.host,
        port=runtime.port,
        log_level=str(args.log_level),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
