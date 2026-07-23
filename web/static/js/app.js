/* ═══════════════════════════════════════════════════════════
   app.js — 主入口，串联各模块
   职责：主对话区仅常规问答；反事实工作台（二级菜单）由
   counterfactual.js 独立管理，二者解耦。
   ═══════════════════════════════════════════════════════════ */

const App = (() => {
  const statusBadge = document.getElementById("status-badge");
  const statusDetail = document.getElementById("status-detail");
  const frameSlider = document.getElementById("frame-slider");
  const frameLabel = document.getElementById("frame-label");
  const playBtn = document.getElementById("play-btn");
  const vizEmpty = document.getElementById("viz-empty");
  const canvasOriginal = document.getElementById("canvas-original");
  const tabIndicator = document.getElementById("tab-indicator");
  const tabs = document.querySelectorAll(".viz-tab");
  const panes = document.querySelectorAll(".viz-pane");

  let loaded = false;
  let viz = new TrajectoryViz(canvasOriginal);
  let playing = false;

  // ── Tab 滑动指示器 ──
  function moveIndicator(tab) {
    if (!tab || !tabIndicator) return;
    const rect = tab.getBoundingClientRect();
    const parentRect = tab.parentElement.getBoundingClientRect();
    tabIndicator.style.left = (rect.left - parentRect.left) + "px";
    tabIndicator.style.width = rect.width + "px";
  }

  function activateTab(name) {
    let activeTab = null;
    tabs.forEach(t => {
      const on = t.dataset.tab === name;
      t.classList.toggle("active", on);
      if (on) activeTab = t;
    });
    panes.forEach(p => p.classList.toggle("active", p.id === (name === "original" ? "pane-original" : "pane-compare")));
    if (activeTab) moveIndicator(activeTab);
  }

  tabs.forEach(tab => {
    tab.addEventListener("click", () => activateTab(tab.dataset.tab));
  });
  // 初始指示器位置（等布局完成）
  requestAnimationFrame(() => {
    const active = document.querySelector(".viz-tab.active");
    if (active) moveIndicator(active);
  });
  window.addEventListener("resize", () => {
    const active = document.querySelector(".viz-tab.active");
    if (active) moveIndicator(active);
  });

  // ── 状态刷新 ──
  async function refreshStatus() {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      applyStatus(data);
      if (data.loaded) {
        await loadOriginalTrajectory();
        await Counterfactual.refreshList();
        await Counterfactual.setOriginalFromLoaded();
      }
    } catch (e) {
      console.error("状态查询失败", e);
    }
  }

  function applyStatus(data) {
    loaded = !!data.loaded;
    if (loaded) {
      statusBadge.className = "status-badge online";
      statusBadge.textContent = "已加载";
      statusDetail.textContent = `${data.prefix} · ${data.player_count}人 · ${data.duration_seconds}s · ${data.person_file} + ${data.ball_file}`;
    } else {
      statusBadge.className = "status-badge offline";
      statusBadge.textContent = "未加载";
      statusDetail.textContent = "";
    }
  }

  // ── 原始轨迹加载 ──
  async function loadOriginalTrajectory() {
    try {
      const res = await fetch("/api/trajectory/original");
      if (!res.ok) { vizEmpty.style.display = "block"; return; }
      const data = await res.json();
      viz.pause(); // 重载前先停止旧播放，避免 RAF 残留导致状态错乱
      playing = false;
      playBtn.textContent = "播放";
      viz.load(data);
      frameSlider.min = data.frame_min;
      frameSlider.max = data.frame_max;
      frameSlider.value = data.frame_min;
      frameSlider.disabled = false;
      playBtn.disabled = false;
      vizEmpty.style.display = "none";
      updateFrameLabel();
    } catch (e) {
      console.error("轨迹加载失败", e);
      vizEmpty.style.display = "block";
    }
  }

  function updateFrameLabel() {
    frameLabel.textContent = `帧 ${Math.round(viz.frame)} / ${viz.frameMax}`;
  }

  // ── 滑块与播放 ──
  frameSlider.addEventListener("input", () => {
    viz.setFrame(parseInt(frameSlider.value));
    updateFrameLabel();
  });
  viz.onFrameChange = (f) => {
    frameSlider.value = Math.round(f);
    updateFrameLabel();
  };

  playBtn.addEventListener("click", () => {
    // toggle() 返回实际播放状态，确保按钮文字与真实状态同步
    const isPlaying = viz.toggle();
    playing = isPlaying;
    playBtn.textContent = playing ? "暂停" : "播放";
  });
  // 播放到末尾时自动恢复按钮文字为"播放"
  viz.onPlayEnd = () => {
    playing = false;
    playBtn.textContent = "播放";
  };

  // ── 快捷指令 ──
  document.querySelectorAll(".chip[data-prompt]").forEach(chip => {
    chip.addEventListener("click", () => {
      const prompt = chip.dataset.prompt;
      const input = document.getElementById("chat-input");
      input.value = prompt;
      input.focus();
    });
  });

  // ── 上传成功回调 ──
  function onDataLoaded(data) {
    applyStatus({ loaded: true, ...data });
    loadOriginalTrajectory();
    Counterfactual.refreshList();
    Counterfactual.setOriginalFromLoaded();
  }

  function onDataCleared() {
    loaded = false;
    statusBadge.className = "status-badge offline";
    statusBadge.textContent = "未加载";
    statusDetail.textContent = "";
    viz.clear();
    frameSlider.min = 0; frameSlider.max = 0; frameSlider.value = 0; frameSlider.disabled = true;
    playBtn.disabled = true;
    vizEmpty.style.display = "block";
    if (window.Counterfactual) window.Counterfactual.clear();
  }

  // ── 启动 ──
  refreshStatus();

  return { onDataLoaded, onDataCleared, isLoaded: () => loaded, refreshStatus, activateTab };
})();
window.App = App;
