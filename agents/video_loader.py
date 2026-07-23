"""视频接口预留层。

当前只做视频文件发现和路径匹配，不实际解码视频。
后续如果需要逐帧读取，可以在这里接入 OpenCV、PyAV 或其他视频库。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class VideoMatch:
    prefix: str
    video_path: Path | None
    available: bool
    note: str = ""


def discover_video_files(video_dir: str | Path | None) -> dict[str, Path]:
    if video_dir is None:
        return {}
    root = Path(video_dir)
    if not root.exists():
        return {}

    discovered: dict[str, Path] = {}
    for file_path in sorted(root.glob("*")):
        if file_path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
            prefix = file_path.stem.split(".")[0]
            discovered[prefix] = file_path
    return discovered


def match_video(prefix: str, video_dir: str | Path | None) -> VideoMatch:
    videos = discover_video_files(video_dir)
    video_path = videos.get(prefix)
    if video_path is None:
        return VideoMatch(prefix=prefix, video_path=None, available=False, note="未找到对应视频，当前仅使用 CSV。")
    return VideoMatch(prefix=prefix, video_path=video_path, available=True, note="视频接口已预留，暂未解码。")


def build_video_context(video_dir: str | Path | None, prefixes: list[str]) -> dict[str, Any]:
    matches = [match_video(prefix, video_dir) for prefix in prefixes]
    available = {item.prefix: str(item.video_path) for item in matches if item.available and item.video_path}
    missing = [item.prefix for item in matches if not item.available]
    return {
        "video_dir": str(video_dir) if video_dir is not None else None,
        "available_videos": available,
        "missing_video_prefixes": missing,
        "video_interface_ready": True,
        "video_decode_enabled": False,
    }
