"""球员追踪结果输出。

把解说稿和结构化球员数据落盘到 Output 目录。
"""

from __future__ import annotations

from pathlib import Path
import json

from .player.tracker import PrefixPlayerCorpus
from .types import FinalDecision


def write_player_trajectories(
    output_dir: str | Path,
    results: dict[str, tuple[PrefixPlayerCorpus, FinalDecision]],
) -> list[Path]:
    """将球员追踪分析结果写入 Output 目录。

    每个前缀输出两个文件：
    - {prefix}_summary.txt  → 足球解说稿
    - {prefix}_summary.json → 结构化球员数据 + 决策 + 球路事件
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written_files: list[Path] = []

    for prefix, (corpus, decision) in sorted(results.items()):
        # txt 文件：使用 composed_summary
        txt_content = decision.composed_summary
        if not txt_content.strip():
            txt_content = f"【{prefix}】无球员轨迹数据。"

        # 补充逐球员的详细轨迹坐标
        player_data = {}
        for player in corpus.sorted_players():
            player_data[player.track_id] = {
                "jersey": player.jersey_label,
                "color": player.color,
                "frame_count": player.frame_count,
                "total_distance": player.total_distance,
                "avg_speed": player.avg_speed,
                "max_speed": player.max_speed,
                "x_range": list(player.x_range),
                "y_range": list(player.y_range),
                "field_coverage_pct": player.field_coverage_pct,
                "dominant_zone": player.dominant_zone,
                "zone_distribution": player.zone_distribution,
                "avg_ball_distance": player.avg_ball_distance,
                "ball_proximity_pct": player.ball_proximity_pct,
                "frames": player.frames,
                "xs": [round(x, 2) for x in player.xs],
                "ys": [round(y, 2) for y in player.ys],
                "speed_curve": player.speed_curve,
                "ball_distances": player.ball_distances,
                "is_goalkeeper": player.is_goalkeeper,
            }

        # 球路事件数据
        ball_events = []
        if corpus.ball_analysis:
            for e in corpus.ball_analysis.events:
                ball_events.append({
                    "type": e.event_type,
                    "start_frame": e.start_frame,
                    "end_frame": e.end_frame,
                    "start_pos": list(e.start_pos),
                    "end_pos": list(e.end_pos),
                    "speed": e.speed,
                    "distance": e.distance,
                    "direction": e.direction,
                    "nearest_player": e.nearest_player,
                    "nearest_player_end": e.nearest_player_end,
                })

        json_content = {
            "prefix": prefix,
            "player_count": corpus.player_count,
            "commentary": decision.composed_summary,
            "ball_events": ball_events,
            "decision": decision.to_dict(),
            "players": player_data,
        }

        txt_file = output_path / f"{prefix}_summary.txt"
        json_file = output_path / f"{prefix}_summary.json"
        txt_file.write_text(txt_content, encoding="utf-8")
        json_file.write_text(json.dumps(json_content, ensure_ascii=False, indent=2), encoding="utf-8")
        written_files.extend([txt_file, json_file])

    return written_files
