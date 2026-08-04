"""足球战术分析智能体 — 本地 Web 后端。

基于 FastAPI，为前端提供：
  - 文件上传（person + ball CSV）
  - 对话流式输出（SSE，真流式 LLM）
  - 原始轨迹数据（JSON，前端 Canvas 逐帧渲染，复刻 showAll.py 逻辑）
  - 反事实轨迹生成（表单提交 → SSE 流式 + 轨迹文件）
  - 反事实轨迹对比数据

启动：
  python -m web.backend
  或
  uvicorn web.backend:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import sys
import traceback
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# 确保项目根目录在 sys.path 上（独立运行 web.backend 时需要）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from harness import Harness, HarnessConfig
from agents.llm_client import set_stream_callback, reset_stream_callback
from agents.progress import set_progress_callback, reset_progress_callback
from agents.player.tracker import _read_csv_rows, _safe_int

FIELD_WIDTH = 1200
FIELD_HEIGHT = 700
FPS = 30

# SSE 事件间隔：无 delta 时定期发送进度/心跳，保持连接活跃并给用户即时反馈
# （修复反事实 MCTS 长时间无输出导致前端"界面无反应"的体验问题）
PROGRESS_INTERVAL = 2.0  # 秒

WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
UPLOAD_DIR = WEB_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = _PROJECT_ROOT / "Output"

load_dotenv(_PROJECT_ROOT / ".env")

app = FastAPI(title="足球战术分析智能体", version="1.0")

# 全局单例 Harness（本地单用户）；Harness 透明包装 SessionManager，
# 默认 passthrough 模式（仅 trace 观测），行为与直接使用 SessionManager 等价。
# 可通过环境变量 HARNESS_MODE/HARNESS_GOLDEN_DIR 等切换 record/replay。
harness: Harness = Harness(HarnessConfig.from_env())


# ════════════════════════════════════════════════════════════════
# 静态资源（禁用缓存，确保前端改动即时生效，避免浏览器持有旧 JS）
# ════════════════════════════════════════════════════════════════

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)


@app.get("/")
async def index():
    return FileResponse(
        str(STATIC_DIR / "index.html"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.middleware("http")
async def no_cache_static(request, call_next):
    """对所有响应注入 no-cache 头，防止浏览器缓存旧版 JS/CSS。"""
    response = await call_next(request)
    if request.url.path.startswith("/static/") or request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# ════════════════════════════════════════════════════════════════
# 状态 & 球员
# ════════════════════════════════════════════════════════════════

@app.get("/api/status")
async def get_status():
    loaded = harness.corpus is not None
    info: dict[str, Any] = {"loaded": loaded}
    if loaded and harness.corpus is not None:
        c = harness.corpus
        all_frames = set()
        for p in c.players.values():
            all_frames.update(p.frames)
        all_frames.update(c.ball_frames.keys())
        frame_max = max(all_frames) if all_frames else 0
        frame_min = min(all_frames) if all_frames else 0
        info.update({
            "prefix": c.prefix,
            "player_count": c.player_count,
            "goalkeeper_count": len(c.goalkeepers()),
            "frame_min": frame_min,
            "frame_max": frame_max,
            "duration_seconds": round(frame_max / FPS, 2),
            "person_file": Path(harness.data_files[0]).name if harness.data_files[0] else "",
            "ball_file": Path(harness.data_files[1]).name if harness.data_files[1] else "",
        })
    return info


@app.get("/api/players")
async def get_players():
    if harness.corpus is None:
        return []
    players = []
    team_map = {"A": "A队", "B": "B队", "C": "门将"}
    for p in sorted(harness.corpus.players.values(), key=lambda x: _safe_sort(x.track_id)):
        players.append({
            "jersey": p.jersey_label,
            "track_id": p.track_id,
            "color": p.color,
            "team": team_map.get(p.color, f"{p.color}队"),
            "is_goalkeeper": p.is_goalkeeper,
            "avg_x": round(p.avg_x(), 1),
            "avg_y": round(p.avg_y(), 1),
            "dominant_zone": p.dominant_zone,
        })
    return players


# ════════════════════════════════════════════════════════════════
# 文件上传
# ════════════════════════════════════════════════════════════════

@app.post("/api/upload")
async def upload_files(
    person_file: UploadFile = File(...),
    ball_file: UploadFile = File(...),
):
    # 保存到 uploads/
    person_path = UPLOAD_DIR / f"person_{person_file.filename}"
    ball_path = UPLOAD_DIR / f"ball_{ball_file.filename}"
    for path, fobj in [(person_path, person_file), (ball_path, ball_file)]:
        with path.open("wb") as f:
            f.write(await fobj.read())

    # 重建一个全新会话，避免旧数据残留
    harness.reset()
    ok = harness.load_data(person_path, ball_path)
    if not ok:
        return JSONResponse(
            {"ok": False, "error": "数据加载失败，请检查 CSV 格式（person: frame_num,track_id,x,y,color; ball: frame_num,x,y,z）"},
            status_code=400,
        )
    return await get_status()


# ════════════════════════════════════════════════════════════════
# 对话（SSE 流式）
# ════════════════════════════════════════════════════════════════

@app.post("/api/chat")
async def chat(message: str = Form(...)):
    if harness.corpus is None:
        return JSONResponse({"error": "请先上传数据文件"}, status_code=400)

    async def event_gen():
        q: queue.SimpleQueue = queue.SimpleQueue()

        def cb(chunk: str):
            q.put(("delta", chunk))

        def progress_cb(stage: str, content: str, extra: dict):
            q.put(("stage", {"stage": stage, "content": content, **extra}))

        token = set_stream_callback(cb)
        ptoken = set_progress_callback(progress_cb)
        task = asyncio.create_task(asyncio.to_thread(harness.handle_input, message))
        last_progress = asyncio.get_event_loop().time()
        try:
            # 立即发送一个进度事件，让前端马上进入"生成中"状态
            yield _sse({"type": "progress", "content": "正在思考…"})
            while True:
                # 先把队列里所有 chunk 排空
                drained_any = False
                while not q.empty():
                    kind, payload = q.get_nowait()
                    if kind == "delta" and payload:
                        drained_any = True
                        yield _sse({"type": "delta", "content": payload})
                    elif kind == "stage":
                        drained_any = True
                        yield _sse({"type": "stage", **payload})
                if task.done():
                    # 任务完成后做最后一次排空（防止竞态）
                    while not q.empty():
                        kind, payload = q.get_nowait()
                        if kind == "delta" and payload:
                            yield _sse({"type": "delta", "content": payload})
                        elif kind == "stage":
                            yield _sse({"type": "stage", **payload})
                    break
                if not drained_any:
                    # 无 delta/阶段 时周期性发送进度心跳，保持连接活跃
                    now = asyncio.get_event_loop().time()
                    if now - last_progress >= PROGRESS_INTERVAL:
                        last_progress = now
                        yield _sse({"type": "progress", "content": "正在生成回复…"})
                    await asyncio.sleep(0.02)
            response = await task
            yield _sse({"type": "done", "content": response})
        except Exception as e:
            traceback.print_exc()
            yield _sse({"type": "error", "content": str(e)})
        finally:
            reset_stream_callback(token)
            reset_progress_callback(ptoken)

    return StreamingResponse(event_gen(), media_type="text/event-stream; charset=utf-8", headers=_sse_headers())


@app.post("/api/clear")
async def clear_session():
    harness.reset()
    return {"ok": True}


# ════════════════════════════════════════════════════════════════
# 反事实（表单 → SSE 流式 + 轨迹文件）
# ════════════════════════════════════════════════════════════════

@app.post("/api/counterfactual/generate")
async def counterfactual_generate(
    subject_player: str = Form(...),
    time_second: float = Form(...),
    altered_action: str = Form(...),
    altered_target: str = Form(""),
    duration_seconds: float = Form(0.0),
):
    if harness.corpus is None:
        return JSONResponse({"error": "请先上传数据文件"}, status_code=400)

    async def event_gen():
        q: queue.SimpleQueue = queue.SimpleQueue()

        def cb(chunk: str):
            q.put(("delta", chunk))

        def progress_cb(stage: str, content: str, extra: dict):
            q.put(("stage", {"stage": stage, "content": content, **extra}))

        token = set_stream_callback(cb)
        ptoken = set_progress_callback(progress_cb)

        def run():
            return harness.handle_counterfactual_form(
                subject_player, time_second, altered_action,
                altered_target, duration_seconds,
            )

        task = asyncio.create_task(asyncio.to_thread(run))
        last_progress = asyncio.get_event_loop().time()
        try:
            # 立即反馈：MCTS 推演耗时长，先告知用户已开始，避免"界面无反应"
            yield _sse({"type": "progress", "content": "正在启动反事实推演（MCTS 2000 次模拟，预计 1-3 分钟）…"})
            while True:
                drained_any = False
                while not q.empty():
                    kind, payload = q.get_nowait()
                    if kind == "delta" and payload:
                        drained_any = True
                        yield _sse({"type": "delta", "content": payload})
                    elif kind == "stage":
                        drained_any = True
                        yield _sse({"type": "stage", **payload})
                if task.done():
                    while not q.empty():
                        kind, payload = q.get_nowait()
                        if kind == "delta" and payload:
                            yield _sse({"type": "delta", "content": payload})
                        elif kind == "stage":
                            yield _sse({"type": "stage", **payload})
                    break
                if not drained_any:
                    now = asyncio.get_event_loop().time()
                    if now - last_progress >= PROGRESS_INTERVAL:
                        last_progress = now
                        yield _sse({"type": "progress", "content": "MCTS 推演中，正在模拟战术决策树…"})
                    await asyncio.sleep(0.02)
            response = await task
            files = harness.last_counterfactual_files
            yield _sse({"type": "done", "content": response, "files": files})
        except Exception as e:
            traceback.print_exc()
            yield _sse({"type": "error", "content": str(e)})
        finally:
            reset_stream_callback(token)
            reset_progress_callback(ptoken)

    return StreamingResponse(event_gen(), media_type="text/event-stream; charset=utf-8", headers=_sse_headers())


@app.post("/api/counterfactual/chat")
async def counterfactual_chat(message: str = Form(...)):
    """反事实二级菜单内的自然语言对话。

    业务边界：与主对话 /api/chat 解耦。
    - 主对话框：常规问答（数据分析、咨询），回复显示在主对话区。
    - 本端点：反事实工作台内的自然语言交互。用户用自然语言描述反事实
      设想（如"如果7号在第3秒直接射门并生成后面5秒轨迹"），由 LLM 路由
      理解后触发反事实分析/轨迹生成；回复以独立流式气泡显示在二级菜单
      内的消息区，不污染主对话区。
    """
    if harness.corpus is None:
        return JSONResponse({"error": "请先上传数据文件"}, status_code=400)

    async def event_gen():
        q: queue.SimpleQueue = queue.SimpleQueue()

        def cb(chunk: str):
            q.put(("delta", chunk))

        def progress_cb(stage: str, content: str, extra: dict):
            q.put(("stage", {"stage": stage, "content": content, **extra}))

        token = set_stream_callback(cb)
        ptoken = set_progress_callback(progress_cb)
        task = asyncio.create_task(asyncio.to_thread(harness.handle_input, message))
        last_progress = asyncio.get_event_loop().time()
        try:
            yield _sse({"type": "progress", "content": "正在理解反事实指令…"})
            while True:
                drained_any = False
                while not q.empty():
                    kind, payload = q.get_nowait()
                    if kind == "delta" and payload:
                        drained_any = True
                        yield _sse({"type": "delta", "content": payload})
                    elif kind == "stage":
                        drained_any = True
                        yield _sse({"type": "stage", **payload})
                if task.done():
                    while not q.empty():
                        kind, payload = q.get_nowait()
                        if kind == "delta" and payload:
                            yield _sse({"type": "delta", "content": payload})
                        elif kind == "stage":
                            yield _sse({"type": "stage", **payload})
                    break
                if not drained_any:
                    now = asyncio.get_event_loop().time()
                    if now - last_progress >= PROGRESS_INTERVAL:
                        last_progress = now
                        yield _sse({"type": "progress", "content": "反事实推演中，正在模拟战术决策树…"})
                    await asyncio.sleep(0.02)
            response = await task
            files = harness.last_counterfactual_files
            yield _sse({"type": "done", "content": response, "files": files})
        except Exception as e:
            traceback.print_exc()
            yield _sse({"type": "error", "content": str(e)})
        finally:
            reset_stream_callback(token)
            reset_progress_callback(ptoken)

    return StreamingResponse(event_gen(), media_type="text/event-stream; charset=utf-8", headers=_sse_headers())


@app.get("/api/counterfactuals")
async def list_counterfactuals():
    """列出 Output 目录中所有反事实轨迹文件对。"""
    if harness.corpus is None:
        return []
    prefix = harness.corpus.prefix
    pat = re.compile(rf"^{re.escape(prefix)}_cf_f(\d+)_(person|ball)\.csv$")
    pairs: dict[int, dict[str, str]] = {}
    for p in OUTPUT_DIR.iterdir() if OUTPUT_DIR.is_dir() else []:
        m = pat.match(p.name)
        if not m:
            continue
        frame = int(m.group(1))
        kind = m.group(2)
        pairs.setdefault(frame, {})[kind] = str(p)
    result = []
    for frame in sorted(pairs.keys()):
        entry = pairs[frame]
        if "person" in entry and "ball" in entry:
            result.append({
                "frame": frame,
                "time_second": round(frame / FPS, 2),
                "person_file": entry["person"],
                "ball_file": entry["ball"],
                "label": f"第{frame / FPS:.1f}秒干预 (帧{frame})",
            })
    return result


# ════════════════════════════════════════════════════════════════
# 轨迹数据（JSON，前端 Canvas 渲染）
# ════════════════════════════════════════════════════════════════

@app.get("/api/trajectory/original")
async def trajectory_original():
    if harness.corpus is None or not harness.data_files[0]:
        return JSONResponse({"error": "未加载数据"}, status_code=400)
    person_csv, ball_csv = harness.data_files
    return _build_trajectory_json(person_csv, ball_csv, label="原始轨迹")


@app.get("/api/trajectory/file")
async def trajectory_file(person: str, ball: str):
    """根据给定 CSV 路径返回轨迹 JSON（用于反事实对比）。"""
    p, b = Path(person), Path(ball)
    if not p.is_file() or not b.is_file():
        return JSONResponse({"error": "文件不存在"}, status_code=404)
    return _build_trajectory_json(str(p), str(b), label="反事实轨迹")


def _build_trajectory_json(person_csv: str, ball_csv: str, label: str = "") -> JSONResponse:
    """复刻 showAll.py 的插值逻辑，返回前端可直接逐帧渲染的 JSON。"""
    try:
        import numpy as np
    except ImportError:
        return JSONResponse({"error": "服务端缺少 numpy"}, status_code=500)

    try:
        _, person_rows = _read_csv_rows(person_csv)
        _, ball_rows = _read_csv_rows(ball_csv)
    except Exception as e:
        return JSONResponse({"error": f"读取 CSV 失败: {e}"}, status_code=500)

    # 解析球员行
    person_data: dict[str, list] = {}
    person_color: dict[str, str] = {}
    person_frames: dict[str, list[int]] = {}
    for row in person_rows:
        frame = _safe_int(row.get("frame_num"))
        if frame is None:
            continue
        tid = str(row.get("track_id", "")).strip()
        if not tid:
            continue
        try:
            x = float(row.get("x", ""))
            y = float(row.get("y", ""))
        except (TypeError, ValueError):
            continue
        person_data.setdefault(tid, []).append((frame, x, y))
        person_frames.setdefault(tid, []).append(frame)
        if tid not in person_color:
            person_color[tid] = str(row.get("color", "")).strip()

    # 解析球行
    ball_pts: list[tuple[int, float, float, float]] = []
    for row in ball_rows:
        frame = _safe_int(row.get("frame_num"))
        if frame is None:
            continue
        try:
            x = float(row.get("x", ""))
            y = float(row.get("y", ""))
            z = float(row.get("z", "0") or 0)
        except (TypeError, ValueError):
            continue
        ball_pts.append((frame, x, y, z))

    if not person_data or not ball_pts:
        return JSONResponse({"error": "CSV 无有效数据"}, status_code=400)

    # 帧范围交集
    person_frame_min = min(min(v) for v in person_frames.values())
    person_frame_max = max(max(v) for v in person_frames.values())
    ball_frame_min = min(f for f, *_ in ball_pts)
    ball_frame_max = max(f for f, *_ in ball_pts)
    frame_min = max(person_frame_min, ball_frame_min)
    frame_max = min(person_frame_max, ball_frame_max)
    if frame_max < frame_min:
        return JSONResponse({"error": "球员与球帧范围无交集"}, status_code=400)
    all_frames = np.arange(frame_min, frame_max + 1)

    # 插值球员
    players_out = []
    for tid in sorted(person_data.keys(), key=lambda t: _safe_sort(t)):
        pts = sorted(person_data[tid], key=lambda r: r[0])
        frames_t = [r[0] for r in pts]
        xs_t = [r[1] for r in pts]
        ys_t = [r[2] for r in pts]
        interp_x = np.interp(all_frames, frames_t, xs_t).tolist()
        interp_y = np.interp(all_frames, frames_t, ys_t).tolist()
        players_out.append({
            "track_id": tid,
            "color": person_color.get(tid, ""),
            "frames": all_frames.tolist(),
            "xs": [round(v, 1) for v in interp_x],
            "ys": [round(v, 1) for v in interp_y],
        })

    # 插值球
    ball_frames_t = [r[0] for r in sorted(ball_pts, key=lambda r: r[0])]
    ball_xs = [r[1] for r in sorted(ball_pts, key=lambda r: r[0])]
    ball_ys = [r[2] for r in sorted(ball_pts, key=lambda r: r[0])]
    ball_zs = [r[3] for r in sorted(ball_pts, key=lambda r: r[0])]
    interp_bx = np.interp(all_frames, ball_frames_t, ball_xs).tolist()
    interp_by = np.interp(all_frames, ball_frames_t, ball_ys).tolist()
    interp_bz = np.interp(all_frames, ball_frames_t, ball_zs).tolist()

    return JSONResponse({
        "label": label,
        "frame_min": int(frame_min),
        "frame_max": int(frame_max),
        "fps": FPS,
        "field": {"width": FIELD_WIDTH, "height": FIELD_HEIGHT},
        "players": players_out,
        "ball": {
            "frames": all_frames.tolist(),
            "xs": [round(v, 1) for v in interp_bx],
            "ys": [round(v, 1) for v in interp_by],
            "zs": [round(v, 1) for v in interp_bz],
        },
    })


# ════════════════════════════════════════════════════════════════
# 辅助
# ════════════════════════════════════════════════════════════════

def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _sse_headers() -> dict[str, str]:
    return {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }


def _safe_sort(track_id: str) -> tuple:
    try:
        return (0, int(track_id), track_id)
    except (ValueError, TypeError):
        return (1, track_id, track_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "web.backend:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
