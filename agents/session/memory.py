"""会话记忆系统。

跨轮次保留分析结果和对话上下文，
支持 LLM 摘要生成、分析缓存和上下文注入。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents.llm_client import call_llm


# ── LLM 摘要提示词 ──

SUMMARY_PROMPT = """## 角色
你是足球分析系统的**会话摘要生成器 (Session Summarizer)**。职责是将对话历史压缩为一段事实性摘要，用于跨轮次记忆注入。

## 输入
最近 N 轮对话的原始文本（用户 + 系统回答）

## 输出
纯文本，每条一行，格式为：
- <关键事实>

## 核心规则
1. **聚焦比赛事实**：只保留已确认的球员信息、事件、战术描述，而非"用户问了什么"
2. **保留关键数据**：球员号、队伍、角色、位置、风格标签等
3. **忽略无关内容**：跳过系统提示、错误信息、寒暄对话
4. **纯文本输出**：不要 markdown 格式

## 输出示例
- 7号：A队右边锋，压迫型前锋
- 比赛第3秒：7号在右路接球后传给11号
- 阵型：A队整体压上至前场侧重左路"""


class MemoryStore:
    """跨轮次记忆存储器。

    保留：
    - 已分析的球员画像和模型缓存（避免重复计算）
    - LLM 生成的对话上下文摘要（注入 Agent prompt）
    """

    def __init__(self) -> None:
        self.analyzed_profiles: dict[str, Any] = {}   # jersey → PlayerProfile
        self.analyzed_models: dict[str, Any] = {}     # jersey → PlayerBehaviorModel
        self.context_summary: str = ""                # LLM 生成的对话摘要
        self._full_history: list[dict[str, str]] = [] # 原始对话（用于摘要生成）
        self.turn_count: int = 0
        self._summary_interval: int = 3  # 每 N 轮对话更新一次摘要
        self._summary_dirty: bool = False  # 摘要是否过期（懒加载）

    # ── 摘要管理 ──

    def update_after_turn(self, user_input: str, assistant_response: str) -> None:
        """每轮对话后更新（摘要改为惰性生成，避免阻塞响应）。"""
        # 仅对超长响应（>2000字符）截断，避免内存问题
        truncated = assistant_response[:2000] if len(assistant_response) > 2000 else assistant_response
        self._full_history.append({"role": "user", "content": user_input})
        self._full_history.append({"role": "assistant", "content": truncated})
        self.turn_count += 1

        # 每 N 轮标记摘要为过期（下次 get_summary 时惰性重生成）
        if self.turn_count >= self._summary_interval and self.turn_count % self._summary_interval == 0:
            self._summary_dirty = True

    def _regenerate_summary(self) -> None:
        """用 LLM 重新生成上下文摘要。"""
        if not self._full_history:
            return

        # 取最近 10 轮
        recent = self._full_history[-20:]
        history_text = "\n".join(
            f"[{h['role']}]: {h['content'][:200]}"
            for h in recent
        )

        result = call_llm(SUMMARY_PROMPT, history_text)
        if result:
            self.context_summary = result.strip()

    def get_summary(self) -> str:
        """获取当前对话上下文摘要（惰性生成：过期时才重算）。"""
        if self._summary_dirty:
            self._regenerate_summary()
            self._summary_dirty = False
        if not self.context_summary:
            return "（尚未建立对话上下文）"
        return self.context_summary

    def get_full_context(self) -> str:
        """获取完整上下文（摘要 + 最近几轮对话）。"""
        parts: list[str] = []

        # 摘要（空时不加入无意义空段）
        if self.context_summary:
            parts.append(f"[对话上下文]\n{self.context_summary}")

        # 附加最近 2 轮原始对话
        recent = self._full_history[-4:]
        if recent:
            parts.append("\n[最近对话]")
            for h in recent:
                role = "用户" if h["role"] == "user" else "系统"
                parts.append(f"{role}: {h['content'][:500]}")

        return "\n\n".join(parts) if parts else ""

    # ── 分析缓存 ──

    def cache_profile(self, jersey: str, profile: Any) -> None:
        """缓存球员画像。"""
        self.analyzed_profiles[jersey] = profile

    def cache_model(self, jersey: str, model: Any) -> None:
        """缓存球员行为模型。"""
        self.analyzed_models[jersey] = model

    def get_cached_profile(self, jersey: str) -> Any | None:
        """获取缓存的球员画像。"""
        return self.analyzed_profiles.get(jersey)

    def get_cached_model(self, jersey: str) -> Any | None:
        """获取缓存的球员行为模型。"""
        return self.analyzed_models.get(jersey)

    def is_cached(self, jersey: str) -> bool:
        """检查球员是否已分析。"""
        return jersey in self.analyzed_profiles and jersey in self.analyzed_models

    # ── 持久化 ──

    def save_to_disk(self, path: str | Path) -> None:
        """保存记忆到磁盘 JSON 文件。"""
        data = {
            "context_summary": self.context_summary,
            "_full_history": self._full_history[-50:],  # 保留最近 50 轮
            "turn_count": self.turn_count,
            "cached_jerseys": list(self.analyzed_models.keys()),
        }
        Path(path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── 清除 ──

    def clear(self) -> None:
        """清除所有记忆。"""
        self.analyzed_profiles.clear()
        self.analyzed_models.clear()
        self.context_summary = ""
        self._summary_dirty = False
        self._full_history.clear()
        self.turn_count = 0
