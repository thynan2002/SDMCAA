"""工具调用（function calling）层单元测试。

不依赖真实 API：以脚本化传输 mock 替换 _call_llm_impl（与
test_harness_equivalence 同款模式），通过工具交换通道观测 payload 与
回填 assistant 消息。

覆盖：
1. tools 参数 + 终止型工具单轮 → 返回 json.dumps(工具参数)
2. 数据型工具多轮循环：工具结果回填 → 模型最终文本
3. 同一响应内多个独立工具请求（并行 tool_calls）一次性执行 + 结果合并
4. 工具执行异常 → 错误结果回填模型（可继续）
5. 工具参数非法 → invalid_arguments 回填模型
6. 循环超过 max_rounds 未收敛 → None
7. bind_prompt_tools 提示词绑定（调用点零改动）
8. 模型未发起工具调用 → 原文返回（mock / 旧 golden 文本回退路径）
9. 回放兼容：ReplayLLM 含 assistant_message 的条目 / 旧条目
10. 流式 tool_calls 增量片段累积（_merge_tool_call_delta）

等价性：不传 tools 且无绑定时走原路径，由 test_harness_equivalence 的
dispatcher 探针与 A/B 用例直接证明，此处不重复。
"""

from __future__ import annotations

import json

from agents import llm_client
from agents.llm_client import ToolExchange, tool_exchange_var
from agents.tools.base import ToolRegistry, ToolSpec, get_default_registry


# ─────────────────────────────────────────────────────────────────
# 脚本化传输 mock（读取交换通道，可脚本化回填 tool_calls）
# ─────────────────────────────────────────────────────────────────

class ScriptedTransport:
    """按轮次脚本驱动的 _call_llm_impl 替身。

    rounds: 每轮一个脚本；None 表示该轮返回纯文本（不触发工具调用）。
    记录每轮传入的 messages / tools，便于断言回填正确性。
    """

    def __init__(self, rounds: list[dict | None], texts: list[str | None] | None = None):
        self.rounds = rounds
        self.texts = texts or [None] * len(rounds)
        self.calls: list[dict] = []
        self.attempts = 0

    def __call__(self, system_prompt, user_message, config=None, max_tokens=None,
                 retries=1, stream=False):
        self.attempts += 1
        exchange = llm_client.get_tool_exchange()
        assert exchange is not None, "工具调用必须处于交换通道内"
        idx = len(self.calls)
        self.calls.append({
            "system_prompt": system_prompt,
            "user_message": user_message,
            "messages": exchange.messages,
            "tools": exchange.tools,
            "tool_choice": exchange.tool_choice,
            "round": exchange.round,
        })
        script = self.rounds[idx] if idx < len(self.rounds) else None
        if script is None:
            return self.texts[idx] if idx < len(self.texts) else None
        exchange.assistant_message = {
            "role": "assistant",
            "content": "",
            "tool_calls": script["tool_calls"],
        }
        return ""


def _tool_call(name: str, args: dict, call_id: str = "call_1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


def _submit_spec(name: str = "submit_demo") -> ToolSpec:
    def handler(args: dict) -> dict:
        return dict(args)

    return ToolSpec(
        name=name,
        description="提交演示结果",
        parameters={
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
        },
        handler=handler,
        terminal=True,
    )


def _data_spec(name: str = "echo_data", terminal: bool = False) -> ToolSpec:
    def handler(args: dict) -> dict:
        return {"echoed": args.get("text", "")}

    return ToolSpec(
        name=name,
        description="回显数据",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
        },
        handler=handler,
        terminal=terminal,
    )


# ─────────────────────────────────────────────────────────────────
# 1. tools 参数 + 终止型工具单轮
# ─────────────────────────────────────────────────────────────────

class TestTerminalToolRound:
    def test_single_round_returns_arguments(self, monkeypatch):
        transport = ScriptedTransport([
            {"tool_calls": [_tool_call("submit_demo", {"answer": 42})]},
        ])
        monkeypatch.setattr(llm_client, "_call_llm_impl", transport)
        out = llm_client.call_llm("sys", "usr", tools=[_submit_spec()])
        assert json.loads(out) == {"answer": 42}
        assert transport.attempts == 1, "终止型工具应立即返回，不产生第二轮"

    def test_payload_contains_tools_and_messages(self, monkeypatch):
        transport = ScriptedTransport([{"tool_calls": [_tool_call("submit_demo", {})]}])
        monkeypatch.setattr(llm_client, "_call_llm_impl", transport)
        llm_client.call_llm("sys", "usr", tools=[_submit_spec()], tool_choice="required")
        call = transport.calls[0]
        assert call["tool_choice"] == "required"
        assert call["tools"][0]["function"]["name"] == "submit_demo"
        # 首轮 messages 必须包含 system + user（完整上下文）
        assert [m["role"] for m in call["messages"]] == ["system", "user"]
        assert call["messages"][0]["content"] == "sys"
        assert call["messages"][1]["content"] == "usr"

    def test_no_tools_legacy_path_unaffected(self, monkeypatch):
        """不传 tools 且无绑定 → 走原路径（交换通道不激活）。"""
        received = {}

        def impl(system_prompt, user_message, config=None, max_tokens=None,
                 retries=1, stream=False):
            received["exchange"] = llm_client.get_tool_exchange()
            return "text"

        monkeypatch.setattr(llm_client, "_call_llm_impl", impl)
        assert llm_client.call_llm("sys", "usr") == "text"
        assert received["exchange"] is None


# ─────────────────────────────────────────────────────────────────
# 2. 数据型工具多轮循环
# ─────────────────────────────────────────────────────────────────

class TestDataToolLoop:
    def test_result_fed_back_then_final_text(self, monkeypatch):
        transport = ScriptedTransport(
            [{"tool_calls": [_tool_call("echo_data", {"text": "球在中路"})]}],
            texts=["", "最终回答"],
        )
        monkeypatch.setattr(llm_client, "_call_llm_impl", transport)
        out = llm_client.call_llm(
            "sys", "usr", tools=[_data_spec()], max_rounds=4,
        )
        assert out == "最终回答"
        assert transport.attempts == 2
        # 第二轮 messages 必须保留完整上下文：system → user → assistant(工具调用) → tool(结果)
        messages = transport.calls[1]["messages"]
        assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool"]
        assert messages[2]["tool_calls"][0]["function"]["name"] == "echo_data"
        tool_msg = messages[3]
        assert tool_msg["role"] == "tool"
        assert "球在中路" in tool_msg["content"]
        assert tool_msg["tool_call_id"] == "call_1"


# ─────────────────────────────────────────────────────────────────
# 3. 并行 tool_calls（同一响应多个独立工具请求，一次性执行）
# ─────────────────────────────────────────────────────────────────

class TestParallelToolCalls:
    def test_all_executed_in_one_go_and_merged(self, monkeypatch):
        executed: list[str] = []

        def handler_a(args: dict) -> dict:
            executed.append("a")
            return {"a": args.get("v")}

        def handler_b(args: dict) -> dict:
            executed.append("b")
            return {"b": args.get("v")}

        tools = [
            ToolSpec(name="submit_a", description="", parameters={},
                     handler=handler_a, terminal=True),
            ToolSpec(name="submit_b", description="", parameters={},
                     handler=handler_b, terminal=True),
        ]
        transport = ScriptedTransport([{
            "tool_calls": [
                _tool_call("submit_a", {"v": 1}, call_id="c1"),
                _tool_call("submit_b", {"v": 2}, call_id="c2"),
            ],
        }])
        monkeypatch.setattr(llm_client, "_call_llm_impl", transport)
        out = json.loads(llm_client.call_llm("sys", "usr", tools=tools))
        assert out == {"a": 1, "b": 2}
        assert executed == ["a", "b"]
        assert transport.attempts == 1

    def test_parallel_then_text_round(self, monkeypatch):
        """并行数据工具（非终止）→ 回填 → 下一轮文本。"""
        transport = ScriptedTransport(
            [
                {"tool_calls": [
                    _tool_call("echo_data", {"text": "x"}, call_id="c1"),
                    _tool_call("echo_data", {"text": "y"}, call_id="c2"),
                ]},
            ],
            texts=["", "合并后回答"],
        )
        monkeypatch.setattr(llm_client, "_call_llm_impl", transport)
        out = llm_client.call_llm("sys", "usr", tools=[_data_spec()])
        assert out == "合并后回答"
        messages = transport.calls[1]["messages"]
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2"]


# ─────────────────────────────────────────────────────────────────
# 4/5. 失败分类与回填
# ─────────────────────────────────────────────────────────────────

class TestToolFailures:
    def test_execution_error_fed_back(self, monkeypatch):
        def boom(args: dict) -> dict:
            raise RuntimeError("数据源崩溃")

        tool = ToolSpec(name="submit_broken", description="", parameters={},
                        handler=boom, terminal=True)
        transport = ScriptedTransport(
            [{"tool_calls": [_tool_call("submit_broken", {})]}],
            texts=["", "已降级处理"],
        )
        monkeypatch.setattr(llm_client, "_call_llm_impl", transport)
        out = llm_client.call_llm("sys", "usr", tools=[tool])
        assert out == "已降级处理"
        tool_msg = [m for m in transport.calls[1]["messages"] if m["role"] == "tool"][0]
        payload = json.loads(tool_msg["content"])
        assert payload["kind"] == "execution_error"
        assert "工具执行异常" in payload["error"]

    def test_invalid_arguments_fed_back_then_success(self, monkeypatch):
        transport = ScriptedTransport([
            {"tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "submit_demo", "arguments": "{bad json"},
            }]},
            {"tool_calls": [_tool_call("submit_demo", {"answer": 7})]},
        ])
        monkeypatch.setattr(llm_client, "_call_llm_impl", transport)
        out = llm_client.call_llm("sys", "usr", tools=[_submit_spec()])
        assert json.loads(out) == {"answer": 7}
        assert transport.attempts == 2
        tool_msg = [m for m in transport.calls[1]["messages"] if m["role"] == "tool"][0]
        payload = json.loads(tool_msg["content"])
        assert payload["kind"] == "invalid_arguments"

    def test_unknown_tool_reported(self, monkeypatch):
        transport = ScriptedTransport(
            [{"tool_calls": [_tool_call("not_registered", {})]}],
            texts=["", "说明"],
        )
        monkeypatch.setattr(llm_client, "_call_llm_impl", transport)
        out = llm_client.call_llm("sys", "usr", tools=[_data_spec()])
        assert out == "说明"
        tool_msg = [m for m in transport.calls[1]["messages"] if m["role"] == "tool"][0]
        assert json.loads(tool_msg["content"])["kind"] == "unknown_tool"

    def test_loop_not_converging_returns_none(self, monkeypatch):
        transport = ScriptedTransport(
            [
                {"tool_calls": [_tool_call("echo_data", {"text": "1"})]},
                {"tool_calls": [_tool_call("echo_data", {"text": "2"})]},
                {"tool_calls": [_tool_call("echo_data", {"text": "3"})]},
                {"tool_calls": [_tool_call("echo_data", {"text": "4"})]},
            ],
        )
        monkeypatch.setattr(llm_client, "_call_llm_impl", transport)
        out = llm_client.call_llm("sys", "usr", tools=[_data_spec()], max_rounds=3)
        assert out is None
        assert transport.attempts == 3

    def test_text_response_without_tool_calls_returned_as_is(self, monkeypatch):
        """模型直接返回文本（未发起工具调用）→ 原文返回（mock/旧 golden 兼容）。"""
        def impl(system_prompt, user_message, config=None, max_tokens=None,
                 retries=1, stream=False):
            return "{\"intent\": \"general_qa\"}"

        monkeypatch.setattr(llm_client, "_call_llm_impl", impl)
        out = llm_client.call_llm("sys", "usr", tools=[_submit_spec()])
        assert out == "{\"intent\": \"general_qa\"}"


# ─────────────────────────────────────────────────────────────────
# 6. 提示词绑定（调用点零改动迁移）
# ─────────────────────────────────────────────────────────────────

class TestPromptBinding:
    def test_binding_activates_tools_path(self, monkeypatch):
        llm_client.bind_prompt_tools("绑定提示词", [_submit_spec()], tool_choice="required")
        try:
            transport = ScriptedTransport([{"tool_calls": [_tool_call("submit_demo", {})]}])
            monkeypatch.setattr(llm_client, "_call_llm_impl", transport)
            llm_client.call_llm("绑定提示词", "用户输入")
            assert transport.calls[0]["tool_choice"] == "required"
            assert transport.calls[0]["tools"][0]["function"]["name"] == "submit_demo"
        finally:
            llm_client.unbind_prompt_tools("绑定提示词")

    def test_explicit_tools_overrides_binding(self, monkeypatch):
        llm_client.bind_prompt_tools("p", [_submit_spec("bound_tool")])
        try:
            transport = ScriptedTransport([{"tool_calls": [_tool_call("submit_demo", {})]}])
            monkeypatch.setattr(llm_client, "_call_llm_impl", transport)
            llm_client.call_llm("p", "u", tools=[_submit_spec("submit_demo")])
            names = [t["function"]["name"] for t in transport.calls[0]["tools"]]
            assert names == ["submit_demo"]
        finally:
            llm_client.unbind_prompt_tools("p")

    def test_no_binding_legacy_path(self, monkeypatch):
        seen = {}

        def impl(system_prompt, user_message, config=None, max_tokens=None,
                 retries=1, stream=False):
            seen["exchange"] = llm_client.get_tool_exchange()
            return "x"

        monkeypatch.setattr(llm_client, "_call_llm_impl", impl)
        llm_client.call_llm("未绑定", "u")
        assert seen["exchange"] is None


# ─────────────────────────────────────────────────────────────────
# 7. 回放兼容（ReplayLLM）
# ─────────────────────────────────────────────────────────────────

class TestReplayCompatibility:
    def _entry(self, response, assistant_message=None, **extra):
        e = {"seq": 0, "stream": False, "system_prompt": "s",
             "user_message": "u", "response": response}
        if assistant_message is not None:
            e["assistant_message"] = assistant_message
        e.update(extra)
        return e

    def test_old_entry_replays_text(self):
        from harness.mock import ReplayLLM
        rl = ReplayLLM([self._entry("旧文本响应")])
        exchange = ToolExchange(tools=[], round=0)
        token = tool_exchange_var.set(exchange)
        try:
            out = rl("s", "u")
        finally:
            tool_exchange_var.reset(token)
        assert out == "旧文本响应"
        assert exchange.assistant_message is None, "旧条目不触发工具路径"

    def test_entry_with_assistant_message_replays_tool_calls(self):
        from harness.mock import ReplayLLM
        am = {
            "role": "assistant", "content": "",
            "tool_calls": [_tool_call("submit_demo", {"answer": 3})],
        }
        rl = ReplayLLM([self._entry(None, assistant_message=am)])
        exchange = ToolExchange(tools=[], round=0)
        token = tool_exchange_var.set(exchange)
        try:
            rl("s", "u")
        finally:
            tool_exchange_var.reset(token)
        assert exchange.assistant_message["tool_calls"][0]["function"]["arguments"] == \
            "{\"answer\": 3}"

    def test_replay_drives_terminal_tool_loop(self, monkeypatch):
        """ReplayLLM 回放的 tool_calls 能驱动真实工具循环。"""
        from harness.mock import ReplayLLM
        am = {
            "role": "assistant", "content": "",
            "tool_calls": [_tool_call("submit_demo", {"answer": 99})],
        }
        rl = ReplayLLM([self._entry(None, assistant_message=am)])
        monkeypatch.setattr(llm_client, "_llm_dispatcher", rl)
        try:
            out = llm_client.call_llm("s", "u", tools=[_submit_spec()])
        finally:
            monkeypatch.setattr(llm_client, "_llm_dispatcher", None)
        assert json.loads(out) == {"answer": 99}


# ─────────────────────────────────────────────────────────────────
# 8. 流式 tool_calls 增量累积
# ─────────────────────────────────────────────────────────────────

class TestStreamingToolDeltas:
    def test_merge_tool_call_delta_accumulates_fragments(self):
        acc: dict[int, dict] = {}
        llm_client._merge_tool_call_delta(acc, [{
            "index": 0, "id": "c1", "type": "function",
            "function": {"name": "submit_", "arguments": "{\"a\":"},
        }])
        llm_client._merge_tool_call_delta(acc, [{
            "index": 0, "function": {"name": "demo", "arguments": "1}"},
        }])
        llm_client._merge_tool_call_delta(acc, [{
            "index": 1, "id": "c2", "function": {"name": "other", "arguments": "{}"},
        }])
        assert acc[0]["id"] == "c1"
        assert acc[0]["function"]["name"] == "submit_demo"
        assert acc[0]["function"]["arguments"] == "{\"a\":1}"
        assert acc[1]["function"]["name"] == "other"


# ─────────────────────────────────────────────────────────────────
# 9. 注册表基础行为
# ─────────────────────────────────────────────────────────────────

class TestRegistryBasics:
    def test_register_and_execute(self):
        registry = ToolRegistry()
        spec = _data_spec("echo")
        registry.register(spec)
        assert registry.names() == ["echo"]
        assert registry.get("echo") is spec
        assert registry.execute("echo", {"text": "hi"}) == {"echoed": "hi"}
        assert registry.execute("missing", {})["kind"] == "unknown_tool"

    def test_default_registry_execution(self):
        get_default_registry().register(_data_spec("echo_default"))
        try:
            out = get_default_registry().execute("echo_default", {"text": "t"})
            assert out == {"echoed": "t"}
        finally:
            get_default_registry()._tools.pop("echo_default", None)


# ─────────────────────────────────────────────────────────────────
# 10. record → replay 闭环（含工具调用条目）
# ─────────────────────────────────────────────────────────────────

class TestRecordReplayToolCalls:
    """录制含 assistant_message 的 golden → 回放驱动真实工具循环。

    证明：新格式（tools / assistant_message 可选字段）向后兼容、
    ReplayLLM 能按录制轨迹重现工具调用、回放回归四维 PASS。
    """

    def test_record_replay_roundtrip_with_tool_calls(self, tmp_path):
        import agents.llm_client as llm_client_module
        import harness.interceptors as interceptors_module
        from harness import Harness, HarnessConfig
        from harness.config import MODE_RECORD, PROJECT_ROOT
        from harness.verify.regression import verify_golden
        from tests._common import BALL_CSV, PERSON_CSV

        # 清空 Output 中可能残留的 12s_cf_* 产物，保证文件清单比对封闭
        output_dir = PROJECT_ROOT / "Output"
        if output_dir.is_dir():
            for stale in output_dir.glob("12s_cf_*.csv"):
                stale.unlink()

        def impl(system_prompt, user_message, config=None, max_tokens=None,
                 retries=1, stream=False):
            # 路由提示词：模拟模型发起 submit_intent 工具调用
            ex = llm_client_module.get_tool_exchange()
            if ex is not None and "意图路由器" in system_prompt:
                ex.assistant_message = {
                    "role": "assistant", "content": "",
                    "tool_calls": [_tool_call(
                        "submit_intent",
                        {"intent": "focus_player", "target_players": ["7号"],
                         "confidence": 0.9, "scenario": None},
                        call_id="c_rt",
                    )],
                }
                return ""
            return "模拟回复[tool_roundtrip]"

        orig_llm = llm_client_module._call_llm_impl
        orig_int = interceptors_module._call_llm_impl
        llm_client_module._call_llm_impl = impl
        interceptors_module._call_llm_impl = impl
        golden_dir = tmp_path / "golden_tools"
        try:
            cfg = HarnessConfig(
                mode=MODE_RECORD, golden_dir=golden_dir, seed=42,
                trace_dir=tmp_path / "t1", trace_enabled=False,
            )
            with Harness(cfg) as h:
                h.init_recording({
                    "name": "tools_roundtrip", "seed": 42,
                    "dataset": {
                        "person": str(PERSON_CSV.resolve()),
                        "ball": str(BALL_CSV.resolve()),
                    },
                })
                assert h.load_data(PERSON_CSV, BALL_CSV)
                # 规则未覆盖的输入（无聚焦/对比/风格等关键词）→ 走 LLM 路由
                resp = h.handle_input("介绍一下球员情况")
                assert resp, "路由工具契约应产出非空响应"
        finally:
            llm_client_module._call_llm_impl = orig_llm
            interceptors_module._call_llm_impl = orig_int

        # 录制条目应包含工具可选字段
        entries = [
            json.loads(l)
            for l in (golden_dir / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        router_entries = [e for e in entries if "意图路由器" in e["system_prompt"]]
        assert router_entries, "应录制到路由调用"
        entry = router_entries[0]
        assert entry.get("assistant_message", {}).get("tool_calls"), "应录制 assistant_message"
        tool_names = [t["function"]["name"] for t in entry.get("tools", [])]
        assert "submit_intent" in tool_names

        # 回放：严格 mock 下四维比对 PASS，工具循环按录制轨迹重现
        result = verify_golden(golden_dir, trace_dir=tmp_path / "t2")
        assert not result.errors, f"回放异常: {result.errors}"
        assert result.llm_sequence_equal, result.llm_first_mismatch
        assert not result.turn_diffs, "逐轮输出或快照不一致"
        assert result.passed
