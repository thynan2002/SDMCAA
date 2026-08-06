/* ═══════════════════════════════════════════════════════════
   chat.js — 对话模块（SSE 流式输出）
   修复点：
   - SSE 文本用 TextDecoder({stream:true}) 增量解码，结束时 flush 残留字节，
     杜绝 UTF-8 多字节字符被字节截断导致的乱码。
   - 完成后用安全的 Markdown -> HTML 渲染（先转义再转换），避免错误转义。
   - 新增 progress 事件处理：长时间无 delta 时显示"生成中"状态。
   ═══════════════════════════════════════════════════════════ */

const Chat = (() => {
  const messagesEl = document.getElementById("chat-messages");
  const inputEl = document.getElementById("chat-input");
  const sendBtn = document.getElementById("chat-send");
  const clearBtn = document.getElementById("clear-chat-btn");

  let busy = false;

  function _clearEmpty() {
    const empty = messagesEl.querySelector(".chat-empty");
    if (empty) empty.remove();
  }

  function _scroll() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function addMessage(role, text) {
    _clearEmpty();
    const msg = document.createElement("div");
    msg.className = `msg ${role}`;
    const bubble = document.createElement("div");
    bubble.className = "msg-bubble";
    if (role === "assistant") {
      bubble.innerHTML = renderMarkdown(text);
    } else {
      bubble.textContent = text;
    }
    const meta = document.createElement("div");
    meta.className = "msg-meta";
    meta.textContent = role === "user" ? "你" : "智能体";
    msg.appendChild(bubble);
    msg.appendChild(meta);
    messagesEl.appendChild(msg);
    _scroll();
    return bubble;
  }

  /* 流式助手气泡：返回 bubble / status 节点 / cursor 节点，便于过程更新 */
  function addStreamingAssistant(initialStatus) {
    _clearEmpty();
    const msg = document.createElement("div");
    msg.className = "msg assistant";
    const bubble = document.createElement("div");
    bubble.className = "msg-bubble streaming";
    // 阶段进度列表（气泡顶部）
    const stageList = document.createElement("div");
    stageList.className = "stage-list";
    bubble.appendChild(stageList);
    // 模型操作轨迹（LLM 调用 / 思考过程 / tool_calls / 工具结果）
    const opLog = document.createElement("div");
    opLog.className = "op-log";
    bubble.appendChild(opLog);
    // 状态行（进度提示，仅在有 stage 前显示）
    const status = document.createElement("div");
    status.className = "stream-status";
    if (initialStatus) {
      status.innerHTML = '<span class="dots"></span>' + escapeHtml(initialStatus);
    }
    bubble.appendChild(status);
    // 正文区（流式增量）
    const body = document.createElement("span");
    body.className = "stream-body";
    bubble.appendChild(body);
    const cursor = document.createElement("span");
    cursor.className = "cursor";
    const meta = document.createElement("div");
    meta.className = "msg-meta";
    meta.textContent = "智能体";
    msg.appendChild(bubble);
    msg.appendChild(meta);
    messagesEl.appendChild(msg);
    _scroll();
    return { bubble, body, status, cursor, stageList, opLog };
  }

  function setBusy(b) {
    busy = b;
    sendBtn.disabled = b;
    inputEl.disabled = b;
    if (b) {
      sendBtn.classList.add("loading");
      sendBtn.textContent = "生成中";
    } else {
      sendBtn.classList.remove("loading");
      sendBtn.textContent = "发送";
      inputEl.focus();
    }
  }

  /* ── 核心：流式接收与渲染 ── */
  async function sendStream(url, formData, opts = {}) {
    const initialStatus = opts.initialStatus || "正在思考…";
    if (busy) {
      // 防止并发流式：若已在生成中，直接返回
      return null;
    }
    setBusy(true);
    const { bubble, body, status, cursor, stageList, opLog } = addStreamingAssistant(initialStatus);
    const stages = new StageTracker(stageList);
    const ops = new OpTracker(opLog);
    let acc = "";

    try {
      const res = await fetch(url, { method: "POST", body: formData });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        status.innerHTML = "";
        body.innerHTML = '<span class="err">❌ ' + escapeHtml(err.error || `请求失败 (${res.status})`) + "</span>";
        setBusy(false);
        return null;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8"); // 显式 UTF-8
      let buf = "";
      let doneContent = null;
      let hasBody = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        // stream:true 保留跨 chunk 的不完整多字节字符在解码器内部缓冲
        buf += decoder.decode(value, { stream: true });
        // 兼容 \n 与 \r\n 分隔
        const lines = buf.split(/\r?\n/);
        buf = lines.pop(); // 保留最后不完整的行
        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          const payload = line.slice(5).trim();
          if (!payload) continue;
          let evt;
          try { evt = JSON.parse(payload); } catch { continue; }
          if (evt.type === "delta") {
            if (!hasBody) {
              // 第一个 delta 到达，清掉进度状态行（阶段列表保留在顶部）
              status.innerHTML = "";
              body.appendChild(cursor);
              hasBody = true;
            }
            acc += evt.content;
            // 流式过程用 textContent 显示纯文本（安全、无闪烁）
            // 把 cursor 移到末尾
            cursor.remove();
            body.textContent = acc;
            body.appendChild(cursor);
            _scroll();
          } else if (evt.type === "stage") {
            // 收到阶段事件：隐藏静态"正在思考"状态行，渲染阶段列表
            status.innerHTML = "";
            stages.handle(evt);
            _scroll();
          } else if (evt.type === "op") {
            // 模型操作事件（LLM 调用/思考/tool_calls/工具结果）
            ops.handle(evt);
            _scroll();
          } else if (evt.type === "progress") {
            // 仅在没有正文且没有阶段项时显示静态进度心跳；阶段开始后忽略
            if (!hasBody && stages.items.length === 0) {
              status.innerHTML = '<span class="dots"></span>' + escapeHtml(evt.content || "处理中…");
              _scroll();
            }
          } else if (evt.type === "done") {
            doneContent = evt.content;
          } else if (evt.type === "error") {
            doneContent = "❌ " + evt.content;
          }
        }
      }
      // flush 解码器内部残留字节（防止末尾多字节字符丢失）
      buf += decoder.decode();
      if (buf.startsWith("data:")) {
        const payload = buf.slice(5).trim();
        if (payload) {
          try {
            const evt = JSON.parse(payload);
            if (evt.type === "done") doneContent = evt.content;
            else if (evt.type === "error") doneContent = "❌ " + evt.content;
          } catch { /* 忽略 */ }
        }
      }

      // 渲染最终结果：用 Markdown 渲染（安全转义 + 格式转换）
      // 阶段进度列表与流式正文整体替换为最终 markdown
      status.innerHTML = "";
      stageList.innerHTML = "";
      cursor.remove();
      ops.finish(); // 操作轨迹折叠为可展开面板，保留在正文上方
      const finalText = doneContent || acc || "（无响应）";
      bubble.classList.remove("streaming");
      body.innerHTML = renderMarkdown(finalText);
      _scroll();
      setBusy(false);
      return doneContent;
    } catch (err) {
      status.innerHTML = "";
      cursor.remove();
      body.innerHTML = '<span class="err">❌ 网络错误：' + escapeHtml(err.message) + "</span>";
      setBusy(false);
      return null;
    }
  }

  async function send(text) {
    if (busy || !text.trim()) return;
    addMessage("user", text);
    inputEl.value = "";
    const fd = new FormData();
    fd.append("message", text);
    const result = await sendStream("/api/chat", fd, { initialStatus: "正在思考…" });
    if (result && window.Counterfactual) window.Counterfactual.refreshList();
    return result;
  }

  sendBtn.addEventListener("click", () => send(inputEl.value));
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send(inputEl.value);
    }
  });

  clearBtn.addEventListener("click", async () => {
    await fetch("/api/clear", { method: "POST" });
    messagesEl.innerHTML = "";
    const empty = document.createElement("div");
    empty.className = "chat-empty";
    empty.textContent = "上下文已清除。请重新上传数据后再开始对话。";
    messagesEl.appendChild(empty);
    if (window.App) window.App.onDataCleared();
  });

  return { send, sendStream, addMessage, isBusy: () => busy };
})();

/* ── 安全 Markdown 渲染：先 HTML 转义，再做格式转换 ── */
function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function renderMarkdown(text) {
  if (!text) return "";
  // 1) 先转义 HTML，杜绝 XSS / 错误转义
  let s = escapeHtml(text);
  // 2) 代码块 ```...```
  s = s.replace(/```([\s\S]*?)```/g, (_, code) => '<pre class="md-code">' + code.replace(/^\n/, "") + "</pre>");
  // 3) 行内代码 `...`
  s = s.replace(/`([^`\n]+)`/g, '<code class="md-inline">$1</code>');
  // 4) 标题 # / ## / ###
  s = s.replace(/^###\s+(.+)$/gm, '<div class="md-h3">$1</div>');
  s = s.replace(/^##\s+(.+)$/gm, '<div class="md-h2">$1</div>');
  s = s.replace(/^#\s+(.+)$/gm, '<div class="md-h1">$1</div>');
  // 5) 粗体 **...** / 斜体 *...*
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, "<em>$1</em>");
  // 6) 无序列表 - / •
  s = s.replace(/(^|\n)(?:-|•)\s+(.+)/g, (_, br, c) => br + '<div class="md-li">• ' + c + "</div>");
  // 7) 有序列表 1.
  s = s.replace(/(^|\n)(\d+)\.\s+(.+)/g, (_, br, n, c) => br + '<div class="md-li">' + n + ". " + c + "</div>");
  // 8) 换行
  s = s.replace(/\n/g, "<br/>");
  // 9) 清理 pre 内被插入的 <br/>
  s = s.replace(/<pre class="md-code">([\s\S]*?)<\/pre>/g, (m, c) => m.replace(/<br\/>/g, "\n"));
  return s;
}

window.Chat = Chat;
window.renderMarkdown = renderMarkdown;
window.escapeHtml = escapeHtml;

/* ═══════════════════════════════════════════════════════════
   StageTracker — 阶段进度渲染器
   接收后端 "stage" 事件，在 AI 气泡顶部渲染动态进度列表：
   - 新阶段到达 → 之前 active 项标记 done，追加新 active 项
   - 同 key 更新（进度）→ 更新该项 detail/percent/status
   - done 事件到达 → 整体替换为最终 markdown（由调用方处理）
   ═══════════════════════════════════════════════════════════ */
class StageTracker {
  constructor(container) {
    this.container = container;   // 进度列表挂载点（bubble 内的 .stage-list）
    this.items = [];              // [{ key, content, status, detail }]
  }

  reset() {
    this.items = [];
    this.container.innerHTML = "";
  }

  /* 处理一个 stage 事件，更新 DOM */
  handle(evt) {
    const key = evt.stage;
    if (!key) return;
    const content = evt.content || "";
    const status = evt.status || "active";
    const detail = evt.detail || (evt.percent != null ? `${evt.percent}%` : "");

    const existing = this.items.find(it => it.key === key);
    if (existing) {
      // 更新已有项（进度更新 / 完成标记）
      existing.content = content || existing.content;
      existing.status = status;
      existing.detail = detail;
    } else {
      // 新阶段：之前的 active 项全部标记为 done
      for (const it of this.items) {
        if (it.status === "active") it.status = "done";
      }
      this.items.push({ key, content, status, detail });
    }
    this._render();
  }

  _render() {
    const html = this.items.map(it => {
      const done = it.status === "done";
      const icon = done ? "✓" : "●";
      const dots = done ? "" : '<span class="stage-dots"><i></i><i></i><i></i></span>';
      const detailHtml = it.detail
        ? `<span class="stage-detail">${escapeHtml(it.detail)}</span>` : "";
      return `<div class="stage-item ${done ? "done" : "active"}">
        <span class="stage-icon">${icon}</span>
        <span class="stage-text">${escapeHtml(it.content)}</span>
        ${detailHtml}${dots}
      </div>`;
    }).join("");
    this.container.innerHTML = html;
  }
}
window.StageTracker = StageTracker;

/* ═══════════════════════════════════════════════════════════
   OpTracker — 模型操作轨迹渲染器
   接收后端 "op" 事件（event 字段区分类型），在气泡内渲染
   「模型操作轨迹」列表：
   - llm_start / llm_end     每次 LLM 调用起止
   - round_start             工具循环轮次
   - reasoning_start / reasoning_delta / reasoning_end / reasoning
                             模型思考过程（流式增量 / 非流式完整）
   - tool_calls              模型发起的工具调用（名称 + 参数）
   - tool_result             工具执行结果（成功/失败 + JSON）
   ═══════════════════════════════════════════════════════════ */
class OpTracker {
  constructor(container) {
    this.container = container;
    this.items = []; // [{ seq, kind, title, status, detail, body, round }]
    this.seq = 0;
  }

  _add({ kind, title, status }) {
    const it = { seq: ++this.seq, kind, title, status, detail: "", body: "", round: null };
    this.items.push(it);
    return it;
  }

  _last(pred) {
    for (let i = this.items.length - 1; i >= 0; i--) {
      if (typeof pred === "string" ? this.items[i].kind === pred : pred(this.items[i])) return this.items[i];
    }
    return null;
  }

  _pretty(s) {
    try {
      const j = JSON.parse(s);
      return JSON.stringify(j, null, 2);
    } catch { return s; }
  }

  /* 处理一个 op 事件，更新列表 */
  handle(evt) {
    switch (evt.event) {
      case "llm_start":
        this._add({
          kind: "llm",
          title: "🧠 调用 LLM" + (evt.model ? `（${evt.model}）` : "") + (evt.has_tools ? " · 工具模式" : "") + (evt.call_seq ? ` · 第 ${evt.call_seq} 次调用` : ""),
          status: "active",
        });
        break;
      case "llm_end": {
        const it = this._last("llm");
        if (it) {
          it.status = evt.ok ? "done" : "fail";
          it.detail = evt.ok ? `完成${evt.length != null ? `（${evt.length} 字符）` : ""}` : "调用失败";
        }
        break;
      }
      case "round_start":
        this._add({
          kind: "round",
          title: `🔄 工具循环第 ${evt.round} 轮${evt.tool_count != null ? `（${evt.tool_count} 个工具可用）` : ""}`,
          status: "done",
        });
        break;
      case "reasoning_start": {
        this._add({ kind: "reasoning", title: "💭 模型思考中…", status: "active" });
        break;
      }
      case "reasoning_delta": {
        let it = this._last("reasoning");
        if (!it) it = this._add({ kind: "reasoning", title: "💭 模型思考中…", status: "active" });
        it.body = (it.body || "") + evt.content;
        break;
      }
      case "reasoning_end": {
        const it = this._last("reasoning");
        if (it) {
          it.status = "done";
          it.title = `💭 思考过程${evt.length != null ? `（${evt.length} 字符）` : ""}`;
        }
        break;
      }
      case "reasoning": {
        // 非流式调用的完整思考过程（中间 LLM 调用）
        const it = this._add({ kind: "reasoning", title: `💭 思考过程${evt.content ? `（${evt.content.length} 字符）` : ""}`, status: "done" });
        it.body = evt.content || "";
        break;
      }
      case "tool_calls": {
        const calls = (evt.calls || []).map(c => {
          const raw = typeof c.arguments === "string" ? c.arguments : JSON.stringify(c.arguments || {}, null, 2);
          return `<div class="op-tool-call"><span class="op-fn">${escapeHtml(c.name || "?")}</span><pre class="op-json">${escapeHtml(this._pretty(raw))}</pre></div>`;
        }).join("");
        const it = this._add({
          kind: "tool",
          title: `🔧 调用工具${evt.round != null ? `（第 ${evt.round} 轮）` : ""}`,
          status: "done",
        });
        it.round = evt.round != null ? evt.round : null;
        it.body = calls;
        break;
      }
      case "tool_result": {
        // 追加到同一轮最近一次工具调用项下；无匹配项时单独成项
        let it = this._last(i => i.kind === "tool" && (evt.round == null || i.round === evt.round));
        if (!it) {
          it = this._add({ kind: "tool", title: `🔧 工具结果${evt.name ? `（${evt.name}）` : ""}`, status: "done" });
          it.round = evt.round != null ? evt.round : null;
        }
        const ok = !!evt.ok;
        const badge = ok ? '<span class="op-badge ok">成功</span>' : '<span class="op-badge fail">失败</span>';
        const head = evt.name ? `<span class="op-fn">${escapeHtml(evt.name)}</span>` : "";
        const bodyHtml = evt.result != null ? `<pre class="op-json">${escapeHtml(this._pretty(JSON.stringify(evt.result, null, 2)))}</pre>` : "";
        it.body = (it.body || "") + `<div class="op-result ${ok ? "ok" : "fail"}">${head}${badge}${bodyHtml}</div>`;
        break;
      }
    }
    this._render();
  }

  /* 流式结束：整体折叠为可展开面板（保留在正文上方） */
  finish() {
    if (this.items.length === 0) {
      this.container.innerHTML = "";
      return;
    }
    const n = this.items.length;
    const inner = this.container.innerHTML;
    this.container.innerHTML = `<details class="op-log-wrap" open>
      <summary class="op-log-summary"><span class="op-log-icon">🔧</span>模型操作轨迹（${n}）
        <span class="op-detail">LLM 调用 · 思考过程 · tool_calls · 工具结果</span></summary>
      <div class="op-log-body">${inner}</div></details>`;
  }

  _render() {
    const html = this.items.map(it => {
      const cls = `op-item ${it.kind} ${it.status}`;
      const icon = it.status === "active" ? "◌" : it.status === "fail" ? "✕" : "✓";
      const dots = it.status === "active" ? '<span class="stage-dots"><i></i><i></i><i></i></span>' : "";
      const detail = it.detail ? `<span class="op-detail">${escapeHtml(it.detail)}</span>` : "";
      // 思考过程是纯文本，其余按 HTML 内容（已转义）渲染
      let body = "";
      if (it.body) {
        body = it.kind === "reasoning"
          ? `<pre class="op-text">${escapeHtml(it.body)}</pre>`
          : `<div class="op-item-body">${it.body}</div>`;
      }
      const open = it.status === "active" || it.kind === "tool" || it.kind === "reasoning";
      return `<details class="${cls}"${open ? " open" : ""}>
        <summary><span class="op-status">${icon}</span><span class="op-title">${escapeHtml(it.title)}</span>${detail}${dots}</summary>${body}</details>`;
    }).join("");
    this.container.innerHTML = html;
  }
}
window.OpTracker = OpTracker;
