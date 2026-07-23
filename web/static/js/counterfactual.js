/* ═══════════════════════════════════════════════════════════
   counterfactual.js — 反事实工作台（二级菜单）
   解耦设计：
   - 结构化表单 → /api/counterfactual/generate（结构化参数）
   - 自然语言对话 → /api/counterfactual/chat（独立消息区，不污染主对话）
   两者的回复均显示在二级菜单内的独立消息区（cf-chat-messages），
   不写入主对话区（chat-messages）。
   ═══════════════════════════════════════════════════════════ */

const Counterfactual = (() => {
  // ── 二级菜单元素 ──
  const triggerBtn = document.getElementById("cf-trigger-btn");
  const modal = document.getElementById("cf-modal");
  const modalClose = document.getElementById("cf-modal-close");
  const submitBtn = document.getElementById("cf-submit");
  const subjectSel = document.getElementById("cf-subject");
  const timeInput = document.getElementById("cf-time");
  const actionSel = document.getElementById("cf-action");
  const targetRow = document.getElementById("cf-target-row");
  const targetSel = document.getElementById("cf-target");
  const durationInput = document.getElementById("cf-duration");

  // 二级菜单内独立对话区
  const cfChatMessages = document.getElementById("cf-chat-messages");
  const cfChatInput = document.getElementById("cf-chat-input");
  const cfChatSend = document.getElementById("cf-chat-send");

  // 对比区
  const cfSelect = document.getElementById("cf-select");
  const cfLoadBtn = document.getElementById("cf-load-btn");
  const compareBody = document.getElementById("compare-body");
  const cmpPlayBtn = document.getElementById("cmp-play-btn");
  const cmpSlider = document.getElementById("cmp-slider");
  const cmpFrameLabel = document.getElementById("cmp-frame-label");
  const cmpFocusBtn = document.getElementById("cmp-focus-btn");
  const cmpFocusLabel = document.getElementById("cmp-focus-label");

  let origData = null;
  let cfData = null;
  let cmpPlaying = false;
  let cmpRaf = null;
  let cmpFrame = 0;
  let cmpFrameMin = 0, cmpFrameMax = 0;
  let cmpMode = "split";
  let cmpPrimary = "original";       // 叠加模式主要轨迹：original | counterfactual
  let cmpHoverHighlight = false;     // 悬停时高亮次要轨迹
  let cfBusy = false; // 二级菜单对话独立 busy 状态，与主对话解耦

  const canvasLeft = document.getElementById("canvas-left");
  const canvasRight = document.getElementById("canvas-right");
  const capLeft = document.getElementById("cap-left");
  const capRight = document.getElementById("cap-right");

  // ── 弹窗开关 ──
  triggerBtn.addEventListener("click", openModal);
  modalClose.addEventListener("click", closeModal);
  modal.addEventListener("click", (e) => { if (e.target === modal) closeModal(); });

  actionSel.addEventListener("change", () => {
    targetRow.hidden = actionSel.value !== "pass";
  });

  function openModal() {
    if (!window.App || !window.App.isLoaded()) {
      alert("请先上传数据文件后再打开反事实工作台。");
      return;
    }
    populatePlayers();
    modal.hidden = false;
    requestAnimationFrame(() => modal.classList.add("show"));
  }
  function closeModal() {
    modal.classList.remove("show");
    setTimeout(() => { modal.hidden = true; }, 220);
  }

  async function populatePlayers() {
    try {
      const res = await fetch("/api/players");
      const players = await res.json();
      const opts = players.map(p => `<option value="${p.jersey}">${p.jersey} · ${p.team}${p.is_goalkeeper ? "（门将）" : ""} · ${p.dominant_zone}</option>`).join("");
      subjectSel.innerHTML = opts || '<option value="">无球员</option>';
      targetSel.innerHTML = '<option value="">不指定</option>' + opts;
    } catch (e) {
      subjectSel.innerHTML = '<option value="">加载失败</option>';
    }
  }

  // ════════════════════════════════════════════════════════════
  // 二级菜单内独立消息区工具函数
  // ════════════════════════════════════════════════════════════
  function _cfScroll() { cfChatMessages.scrollTop = cfChatMessages.scrollHeight; }
  function _cfClearEmpty() {
    const empty = cfChatMessages.querySelector(".cf-chat-empty");
    if (empty) empty.remove();
  }

  function cfAddMessage(role, text) {
    _cfClearEmpty();
    const msg = document.createElement("div");
    msg.className = `cf-msg ${role}`;
    const bubble = document.createElement("div");
    bubble.className = "cf-msg-bubble";
    if (role === "assistant") bubble.innerHTML = window.renderMarkdown(text);
    else bubble.textContent = text;
    msg.appendChild(bubble);
    cfChatMessages.appendChild(msg);
    _cfScroll();
    return bubble;
  }

  function cfAddStreaming(initialStatus) {
    _cfClearEmpty();
    const msg = document.createElement("div");
    msg.className = "cf-msg assistant";
    const bubble = document.createElement("div");
    bubble.className = "cf-msg-bubble streaming";
    // 阶段进度列表（气泡顶部）
    const stageList = document.createElement("div");
    stageList.className = "stage-list";
    bubble.appendChild(stageList);
    const status = document.createElement("div");
    status.className = "stream-status";
    if (initialStatus) status.innerHTML = '<span class="dots"></span>' + window.escapeHtml(initialStatus);
    bubble.appendChild(status);
    const body = document.createElement("span");
    body.className = "stream-body";
    bubble.appendChild(body);
    const cursor = document.createElement("span");
    cursor.className = "cursor";
    msg.appendChild(bubble);
    cfChatMessages.appendChild(msg);
    _cfScroll();
    return { bubble, body, status, cursor, stageList };
  }

  function cfSetBusy(b) {
    cfBusy = b;
    cfChatSend.disabled = b;
    submitBtn.disabled = b;
    cfChatInput.disabled = b;
    cfChatSend.textContent = b ? "生成中" : "发送";
  }

  /* 独立流式接收：写入二级菜单消息区，不碰主对话区 */
  async function cfSendStream(url, formData, initialStatus) {
    if (cfBusy) return null;
    cfSetBusy(true);
    const { bubble, body, status, cursor, stageList } = cfAddStreaming(initialStatus);
    const stages = new window.StageTracker(stageList);
    let acc = "";
    try {
      const res = await fetch(url, { method: "POST", body: formData });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        status.innerHTML = "";
        body.innerHTML = '<span class="err">❌ ' + window.escapeHtml(err.error || `请求失败 (${res.status})`) + "</span>";
        cfSetBusy(false);
        return null;
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buf = "", doneContent = null, hasBody = false;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split(/\r?\n/);
        buf = lines.pop();
        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          const payload = line.slice(5).trim();
          if (!payload) continue;
          let evt;
          try { evt = JSON.parse(payload); } catch { continue; }
          if (evt.type === "delta") {
            if (!hasBody) { status.innerHTML = ""; body.appendChild(cursor); hasBody = true; }
            acc += evt.content;
            cursor.remove();
            body.textContent = acc;
            body.appendChild(cursor);
            _cfScroll();
          } else if (evt.type === "stage") {
            status.innerHTML = "";
            stages.handle(evt);
            _cfScroll();
          } else if (evt.type === "progress") {
            if (!hasBody && stages.items.length === 0) {
              status.innerHTML = '<span class="dots"></span>' + window.escapeHtml(evt.content || "处理中…"); _cfScroll();
            }
          } else if (evt.type === "done") {
            doneContent = evt.content;
          } else if (evt.type === "error") {
            doneContent = "❌ " + evt.content;
          }
        }
      }
      buf += decoder.decode();
      if (buf.startsWith("data:")) {
        try {
          const evt = JSON.parse(buf.slice(5).trim());
          if (evt.type === "done") doneContent = evt.content;
          else if (evt.type === "error") doneContent = "❌ " + evt.content;
        } catch { /* 忽略 */ }
      }
      status.innerHTML = "";
      stageList.innerHTML = "";
      cursor.remove();
      bubble.classList.remove("streaming");
      body.innerHTML = window.renderMarkdown(doneContent || acc || "（无响应）");
      _cfScroll();
      cfSetBusy(false);
      return doneContent;
    } catch (err) {
      status.innerHTML = "";
      cursor.remove();
      body.innerHTML = '<span class="err">❌ 网络错误：' + window.escapeHtml(err.message) + "</span>";
      cfSetBusy(false);
      return null;
    }
  }

  // ════════════════════════════════════════════════════════════
  // 结构化表单提交
  // ════════════════════════════════════════════════════════════
  submitBtn.addEventListener("click", async () => {
    const subject = subjectSel.value;
    const time = parseFloat(timeInput.value);
    const action = actionSel.value;
    const target = action === "pass" ? targetSel.value : "";
    const duration = parseFloat(durationInput.value) || 0;

    if (!subject) { alert("请选择主体球员"); return; }
    if (isNaN(time) || time < 0) { alert("请输入有效的干预时间"); return; }

    const label = actionLabel(action);
    // 显示在二级菜单消息区（不污染主对话区）
    cfAddMessage("user", `【结构化】如果 ${subject} 在第 ${time}s 改为${label}${target ? "给 " + target : ""}，生成${duration > 0 ? duration + "s" : "到结束"}的轨迹数据`);

    const fd = new FormData();
    fd.append("subject_player", subject);
    fd.append("time_second", time);
    fd.append("altered_action", action);
    fd.append("altered_target", target);
    fd.append("duration_seconds", duration);

    const result = await cfSendStream("/api/counterfactual/generate", fd, "正在启动反事实推演（MCTS 2000 次模拟，预计 1-3 分钟）…");
    await refreshList();
    if (cfSelect.options.length > 0 && result !== null) {
      cfSelect.selectedIndex = 0;
      await loadSelected();
      switchToCompareTab();
    }
  });

  // ════════════════════════════════════════════════════════════
  // 自然语言反事实对话（二级菜单内，独立于主对话）
  // ════════════════════════════════════════════════════════════
  async function cfSendNaturalLanguage() {
    const text = cfChatInput.value.trim();
    if (!text || cfBusy) return;
    cfChatInput.value = "";
    cfAddMessage("user", text);
    const fd = new FormData();
    fd.append("message", text);
    const result = await cfSendStream("/api/counterfactual/chat", fd, "正在理解反事实指令…");
    // 若触发了轨迹生成，刷新对比列表
    await refreshList();
    if (cfSelect.options.length > 0 && result !== null) {
      cfSelect.selectedIndex = 0;
      await loadSelected();
      switchToCompareTab();
    }
  }
  cfChatSend.addEventListener("click", cfSendNaturalLanguage);
  cfChatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      cfSendNaturalLanguage();
    }
  });

  function actionLabel(a) {
    return { shoot: "射门", dribble: "盘带突破", pass: "传球", clear: "解围", hold: "控球" }[a] || a;
  }

  // ── 反事实列表 ──
  async function refreshList() {
    try {
      const res = await fetch("/api/counterfactuals");
      const list = await res.json();
      if (!list.length) {
        cfSelect.innerHTML = '<option value="">暂无反事实轨迹</option>';
        return;
      }
      cfSelect.innerHTML = list.map(it => `<option value="${it.person_file}|${it.ball_file}">${it.label}</option>`).join("");
    } catch (e) {
      cfSelect.innerHTML = '<option value="">加载失败</option>';
    }
  }

  cfLoadBtn.addEventListener("click", loadSelected);

  async function loadSelected() {
    const val = cfSelect.value;
    if (!val || val === "") return;
    const [person, ball] = val.split("|");
    try {
      const res = await fetch(`/api/trajectory/file?person=${encodeURIComponent(person)}&ball=${encodeURIComponent(ball)}`);
      cfData = await res.json();
      capRight.textContent = "反事实轨迹：" + (cfData.label || "");
      if (!origData) await loadOriginal();
      setupCompare();
    } catch (e) {
      alert("加载反事实轨迹失败：" + e.message);
    }
  }

  async function loadOriginal() {
    try {
      const res = await fetch("/api/trajectory/original");
      origData = await res.json();
    } catch (e) {
      origData = null;
    }
  }

  // ── 对比渲染 ──
  function setupCompare() {
    if (!origData && !cfData) return;
    // 重载时先停止旧播放，避免 RAF 残留
    if (cmpPlaying) pauseCompare();
    const oMax = origData ? origData.frame_max : 0;
    const cMax = cfData ? cfData.frame_max : 0;
    const oMin = origData ? origData.frame_min : 0;
    const cMin = cfData ? cfData.frame_min : 0;
    cmpFrameMin = Math.min(oMin, cMin);
    cmpFrameMax = Math.max(oMax, cMax);
    cmpFrame = cmpFrameMin;
    cmpSlider.min = cmpFrameMin;
    cmpSlider.max = cmpFrameMax;
    cmpSlider.value = cmpFrame;
    cmpSlider.disabled = false;
    cmpPlayBtn.disabled = false;
    renderCompare();
  }

  function renderCompare() {
    if (cmpMode === "split") {
      compareBody.classList.remove("overlay");
      compareBody.querySelectorAll(".viz-canvas-wrap.half").forEach(w => w.style.display = "");
      if (origData) drawOn(canvasLeft, origData, cmpFrame);
      else clearCanvas(canvasLeft);
      if (cfData) drawOn(canvasRight, cfData, cmpFrame);
      else clearCanvas(canvasRight);
    } else {
      compareBody.classList.add("overlay");
      const halves = compareBody.querySelectorAll(".viz-canvas-wrap.half");
      if (halves.length >= 2) halves[1].style.display = "none";
      if (origData || cfData) {
        window.renderOverlay(canvasLeft, origData, cfData, cmpFrame, {
          primary: cmpPrimary,
          highlightSecondary: cmpHoverHighlight,
        });
      }
    }
    cmpFrameLabel.textContent = `帧 ${Math.round(cmpFrame)} / ${cmpFrameMax}`;
  }

  function updateFocusBtn() {
    if (cmpMode !== "overlay") {
      cmpFocusBtn.hidden = true;
      return;
    }
    cmpFocusBtn.hidden = false;
    cmpFocusLabel.textContent = cmpPrimary === "original" ? "聚焦：原始" : "聚焦：反事实";
  }

  function drawOn(canvas, data, frame) {
    const v = canvas.__viz || (canvas.__viz = new TrajectoryViz(canvas));
    if (v.data !== data) v.load(data);
    v.setFrame(frame);
  }
  function clearCanvas(canvas) {
    const v = canvas.__viz || (canvas.__viz = new TrajectoryViz(canvas));
    v.clear();
  }

  cmpSlider.addEventListener("input", () => {
    cmpFrame = parseInt(cmpSlider.value);
    renderCompare();
  });

  cmpPlayBtn.addEventListener("click", () => {
    cmpPlaying ? pauseCompare() : playCompare();
  });

  function playCompare() {
    if (!origData && !cfData) return;
    if (cmpFrameMax <= cmpFrameMin) return; // 帧范围无效
    // 起始帧已到末尾时，先回到起点再播放
    if (cmpFrame >= cmpFrameMax) {
      cmpFrame = cmpFrameMin;
      cmpSlider.value = Math.round(cmpFrame);
      renderCompare();
    }
    cmpPlaying = true;
    cmpPlayBtn.textContent = "暂停";
    let last = performance.now();
    const step = (ts) => {
      if (!cmpPlaying) return;
      const dt = (ts - last) / 1000; last = ts;
      cmpFrame += dt * 30;
      if (cmpFrame >= cmpFrameMax) {
        // 播放到末尾停止，按钮文字同步回"播放"
        cmpFrame = cmpFrameMax;
        cmpSlider.value = Math.round(cmpFrame);
        renderCompare();
        pauseCompare();
        return;
      }
      cmpSlider.value = Math.round(cmpFrame);
      renderCompare();
      cmpRaf = requestAnimationFrame(step);
    };
    cmpRaf = requestAnimationFrame(step);
  }
  function pauseCompare() {
    cmpPlaying = false;
    cmpPlayBtn.textContent = "播放";
    if (cmpRaf) cancelAnimationFrame(cmpRaf);
  }

  document.querySelectorAll('input[name="compare-mode"]').forEach(r => {
    r.addEventListener("change", () => {
      cmpMode = document.querySelector('input[name="compare-mode"]:checked').value;
      updateFocusBtn();
      renderCompare();
    });
  });

  // 叠加对比 — 聚焦按钮：悬停高亮次要轨迹，点击切换主要轨迹
  cmpFocusBtn.addEventListener("mouseenter", () => {
    cmpHoverHighlight = true;
    if (cmpMode === "overlay") renderCompare();
  });
  cmpFocusBtn.addEventListener("mouseleave", () => {
    cmpHoverHighlight = false;
    if (cmpMode === "overlay") renderCompare();
  });
  cmpFocusBtn.addEventListener("click", () => {
    cmpPrimary = cmpPrimary === "original" ? "counterfactual" : "original";
    cmpHoverHighlight = false; // 点击后次要不再高亮（已切换主要）
    updateFocusBtn();
    renderCompare();
  });

  function switchToCompareTab() {
    if (window.App) window.App.activateTab("compare");
  }

  async function setOriginalFromLoaded() {
    await loadOriginal();
    if (origData) setupCompare();
  }

  function clear() {
    // 停止播放，清空数据与画布，重置控件
    if (cmpPlaying) pauseCompare();
    origData = null;
    cfData = null;
    cmpFrame = 0;
    cmpFrameMin = 0;
    cmpFrameMax = 0;
    cmpMode = "split";
    cmpPrimary = "original";
    cmpHoverHighlight = false;
    cmpSlider.min = 0; cmpSlider.max = 0; cmpSlider.value = 0; cmpSlider.disabled = true;
    cmpPlayBtn.disabled = true;
    cmpFrameLabel.textContent = "帧 0 / 0";
    cmpFocusBtn.hidden = true;
    cfSelect.innerHTML = '<option value="">暂无反事实轨迹</option>';
    capLeft.textContent = "原始轨迹";
    capRight.textContent = "反事实轨迹";
    clearCanvas(canvasLeft);
    clearCanvas(canvasRight);
    // 恢复分屏布局（叠加模式可能隐藏了右半区）
    compareBody.classList.remove("overlay");
    compareBody.querySelectorAll(".viz-canvas-wrap.half").forEach(w => w.style.display = "");
    // 清空二级菜单消息区
    cfChatMessages.innerHTML = "";
    const empty = document.createElement("div");
    empty.className = "cf-chat-empty";
    empty.innerHTML = "用自然语言描述反事实设想，由大模型理解后生成数据。<br/>例如：如果7号在第3秒直接射门，生成后面5秒的轨迹数据";
    cfChatMessages.appendChild(empty);
  }

  return { refreshList, loadSelected, setOriginalFromLoaded, switchToCompareTab, getOrigData: () => origData, clear };
})();
window.Counterfactual = Counterfactual;
