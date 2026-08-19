/* ═══════════════════════════════════════════════════════════
   viz.js — 足球场轨迹可视化（Canvas 逐帧渲染，含缺帧插值）
   ═══════════════════════════════════════════════════════════ */

const TEAM_COLORS = { A: "#ef4444", B: "#3b82f6", C: "#22c55e" };
// 标准球场尺寸
const FIELD_W = 1200, FIELD_H = 700;
// 绘图画布扩展边距：球场四周留白，确保越界坐标（界外球/发球）可见
const FIELD_MARGIN = 60;
const CANVAS_W = FIELD_W + FIELD_MARGIN * 2;  // 1320
const CANVAS_H = FIELD_H + FIELD_MARGIN * 2;  // 820
// 坐标转换：原始坐标(0~1200, 0~700) → 画布像素(+margin 偏移)
const OX = FIELD_MARGIN, OY = FIELD_MARGIN;

class TrajectoryViz {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.data = null;
    this.frame = 0;
    this.frameMin = 0;
    this.frameMax = 0;
    this.playing = false;
    this._raf = null;
    this._lastTick = 0;
    this._speed = 1.0; // 倍速
    this.onFrameChange = null;
  }

  load(data) {
    this.data = data;
    if (!data) { this.clear(); return; }
    this.frameMin = data.frame_min;
    this.frameMax = data.frame_max;
    this.frame = this.frameMin;
    this.render();
  }

  clear() {
    this.data = null;
    this.frame = 0;
    this.frameMin = 0;
    this.frameMax = 0;
    this.pause();
    const { ctx } = this;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this._drawField();
  }

  setFrame(f) {
    if (!this.data) return;
    // 保留浮点值，不 Math.round 覆盖——否则 play() 循环里每帧增量 <0.5
    // 会被 round 截掉并写回 this.frame，导致 frame 永远卡在 0、画面不动。
    // render() 内部自行 round 取索引，此处只做范围 clamp。
    this.frame = Math.max(this.frameMin, Math.min(this.frameMax, f));
    this.render();
    if (this.onFrameChange) this.onFrameChange(this.frame);
  }

  play() {
    if (!this.data || this.playing) return;
    // 帧范围无效（单帧或未加载）时不启动播放，避免 RAF 空转
    if (this.frameMax <= this.frameMin) return;
    // 起始帧已到末尾时，先回到起点再播放，否则第一帧就触发结束
    if (this.frame >= this.frameMax) {
      this.frame = this.frameMin;
      this.setFrame(this.frame);
    }
    this.playing = true;
    // 用独立浮点变量累积播放进度，避免 setFrame 覆盖 this.frame 导致增量丢失
    let playFrame = this.frame;
    this._lastTick = performance.now();
    const step = (ts) => {
      if (!this.playing) return;
      const dt = (ts - this._lastTick) / 1000;
      this._lastTick = ts;
      // 30 fps 推进（playFrame 是浮点，每帧增量约 0.5，不会丢失）
      playFrame += dt * 30 * this._speed;
      if (playFrame >= this.frameMax) {
        // 播放到末尾停止（不静默循环），让按钮文字同步回"播放"
        playFrame = this.frameMax;
        this.setFrame(playFrame);
        this.pause();
        if (this.onPlayEnd) this.onPlayEnd();
        return;
      }
      this.setFrame(playFrame);
      this._raf = requestAnimationFrame(step);
    };
    this._raf = requestAnimationFrame(step);
  }

  pause() {
    this.playing = false;
    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = null;
  }

  toggle() {
    // 返回实际播放状态，供调用方同步按钮文字
    if (this.playing) {
      this.pause();
      return false;
    }
    this.play();
    return this.playing;
  }

  idx(frame) { return Math.round(frame) - this.frameMin; }

  render() {
    if (!this.data) return;
    const { ctx, data } = this;
    this._drawField();
    const fi = this.idx(this.frame);

    // 球员轨迹 + 当前位置（所有坐标 +OX/+OY 偏移到画布坐标系）
    for (const p of data.players) {
      const color = TEAM_COLORS[p.color] || "#9ca3af";
      // 轨迹线
      if (fi > 0) {
        ctx.beginPath();
        ctx.moveTo(p.xs[0] + OX, p.ys[0] + OY);
        for (let i = 1; i <= fi && i < p.xs.length; i++) {
          ctx.lineTo(p.xs[i] + OX, p.ys[i] + OY);
        }
        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.45;
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
      // 当前位置
      const ci = Math.min(fi, p.xs.length - 1);
      if (ci >= 0) {
        const x = p.xs[ci] + OX, y = p.ys[ci] + OY;
        ctx.beginPath();
        ctx.arc(x, y, 9, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        // 球衣号标签
        ctx.fillStyle = "#fff";
        ctx.font = "bold 10px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(p.track_id, x, y);
      }
    }

    // 球轨迹 + 当前位置
    const b = data.ball;
    if (b && b.xs && b.xs.length) {
      if (fi > 0) {
        ctx.beginPath();
        ctx.moveTo(b.xs[0] + OX, b.ys[0] + OY);
        for (let i = 1; i <= fi && i < b.xs.length; i++) {
          ctx.lineTo(b.xs[i] + OX, b.ys[i] + OY);
        }
        ctx.strokeStyle = "#ffffff";
        ctx.globalAlpha = 0.85;
        ctx.lineWidth = 2.5;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
      const ci = Math.min(fi, b.xs.length - 1);
      if (ci >= 0) {
        ctx.beginPath();
        ctx.arc(b.xs[ci] + OX, b.ys[ci] + OY, 7, 0, Math.PI * 2);
        ctx.fillStyle = "#ffffff";
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#000";
        ctx.stroke();
      }
    }
  }

  _drawField() {
    const { ctx } = this;
    // 外围深色背景（覆盖整个 canvas，含边距区，越界点在此背景上可见）
    ctx.fillStyle = "#1a2e0d";
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
    // 草地背景（标准球场区域，偏移到画布中央）
    ctx.fillStyle = "#2d5016";
    ctx.fillRect(OX, OY, FIELD_W, FIELD_H);
    // 草地条纹（修饰）
    ctx.fillStyle = "rgba(255,255,255,0.03)";
    for (let x = 0; x < FIELD_W; x += 80) {
      ctx.fillRect(OX + x, OY, 40, FIELD_H);
    }
    // 场线（球场边界，偏移后绘制）
    ctx.strokeStyle = "rgba(255,255,255,0.55)";
    ctx.lineWidth = 2;
    ctx.strokeRect(OX + 20, OY + 20, FIELD_W - 40, FIELD_H - 40);
    // 中线
    ctx.beginPath();
    ctx.moveTo(OX + FIELD_W / 2, OY + 20);
    ctx.lineTo(OX + FIELD_W / 2, OY + FIELD_H - 20);
    ctx.stroke();
    // 中圈
    ctx.beginPath();
    ctx.arc(OX + FIELD_W / 2, OY + FIELD_H / 2, 60, 0, Math.PI * 2);
    ctx.stroke();
    // 禁区（上下）
    const boxW = 140, boxH = 320;
    ctx.strokeRect(OX + 20, OY + (FIELD_H - boxH) / 2, boxW, boxH);
    ctx.strokeRect(OX + FIELD_W - 20 - boxW, OY + (FIELD_H - boxH) / 2, boxW, boxH);
    // 球门
    ctx.fillStyle = "rgba(255,255,255,0.7)";
    ctx.fillRect(OX + 8, OY + FIELD_H / 2 - 40, 12, 80);
    ctx.fillRect(OX + FIELD_W - 20, OY + FIELD_H / 2 - 40, 12, 80);
  }
}

/* 叠加渲染：在同一画布上绘制原始 + 反事实
   opts.primary: "original" | "counterfactual" — 主要轨迹（实线+位置点）
   opts.highlightSecondary: bool — 高亮次要轨迹（提亮+位置点） */
function renderOverlay(canvas, origData, cfData, frame, opts = {}) {
  const primary = opts.primary || "original";
  const highlightSecondary = !!opts.highlightSecondary;
  const ctx = canvas.getContext("2d");
  // 外围深色背景 + 草地（与 TrajectoryViz._drawField 一致的偏移逻辑）
  ctx.fillStyle = "#1a2e0d";
  ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
  ctx.fillStyle = "#2d5016";
  ctx.fillRect(OX, OY, FIELD_W, FIELD_H);
  ctx.fillStyle = "rgba(255,255,255,0.03)";
  for (let x = 0; x < FIELD_W; x += 80) ctx.fillRect(OX + x, OY, 40, FIELD_H);
  ctx.strokeStyle = "rgba(255,255,255,0.55)";
  ctx.lineWidth = 2;
  ctx.strokeRect(OX + 20, OY + 20, FIELD_W - 40, FIELD_H - 40);
  ctx.beginPath();
  ctx.moveTo(OX + FIELD_W / 2, OY + 20); ctx.lineTo(OX + FIELD_W / 2, OY + FIELD_H - 20);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(OX + FIELD_W / 2, OY + FIELD_H / 2, 60, 0, Math.PI * 2);
  ctx.stroke();
  const boxW = 140, boxH = 320;
  ctx.strokeRect(OX + 20, OY + (FIELD_H - boxH) / 2, boxW, boxH);
  ctx.strokeRect(OX + FIELD_W - 20 - boxW, OY + (FIELD_H - boxH) / 2, boxW, boxH);

  // style: "primary"（实线+位置点） / "secondary"（虚线+淡） / "highlighted"（虚线+提亮+位置点）
  const drawDataset = (data, frame, style) => {
    if (!data) return;
    const dashed = style !== "primary";
    const showDots = style === "primary" || style === "highlighted";
    const fi = Math.round(frame) - data.frame_min;
    for (const p of data.players) {
      const color = TEAM_COLORS[p.color] || "#9ca3af";
      if (fi > 0) {
        ctx.beginPath();
        ctx.setLineDash(dashed ? [6, 4] : []);
        ctx.moveTo(p.xs[0] + OX, p.ys[0] + OY);
        for (let i = 1; i <= fi && i < p.xs.length; i++) ctx.lineTo(p.xs[i] + OX, p.ys[i] + OY);
        ctx.strokeStyle = color;
        ctx.globalAlpha = style === "highlighted" ? 0.75 : (style === "primary" ? 0.45 : 0.22);
        ctx.lineWidth = style === "highlighted" ? 2.0 : (style === "primary" ? 1.5 : 1.2);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
      }
      if (showDots) {
        const ci = Math.min(fi, p.xs.length - 1);
        if (ci >= 0) {
          ctx.beginPath();
          ctx.arc(p.xs[ci] + OX, p.ys[ci] + OY, 9, 0, Math.PI * 2);
          ctx.fillStyle = color; ctx.fill();
          ctx.fillStyle = "#fff";
          ctx.font = "bold 10px sans-serif";
          ctx.textAlign = "center"; ctx.textBaseline = "middle";
          ctx.fillText(p.track_id, p.xs[ci] + OX, p.ys[ci] + OY);
        }
      }
    }
    const b = data.ball;
    if (b && b.xs && b.xs.length) {
      if (fi > 0) {
        ctx.beginPath();
        ctx.setLineDash(dashed ? [6, 4] : []);
        ctx.moveTo(b.xs[0] + OX, b.ys[0] + OY);
        for (let i = 1; i <= fi && i < b.xs.length; i++) ctx.lineTo(b.xs[i] + OX, b.ys[i] + OY);
        ctx.strokeStyle = dashed ? "#fbbf24" : "#ffffff";
        ctx.globalAlpha = style === "highlighted" ? 0.9 : (dashed ? 0.6 : 0.85);
        ctx.lineWidth = style === "highlighted" ? 3.0 : 2.5; ctx.stroke();
        ctx.setLineDash([]); ctx.globalAlpha = 1;
      }
      if (showDots) {
        const ci = Math.min(fi, b.xs.length - 1);
        if (ci >= 0) {
          ctx.beginPath();
          ctx.arc(b.xs[ci] + OX, b.ys[ci] + OY, 7, 0, Math.PI * 2);
          ctx.fillStyle = "#fff"; ctx.fill();
          ctx.lineWidth = 2; ctx.strokeStyle = "#000"; ctx.stroke();
        }
      }
    }
  };
  // 次要轨迹先画（底层），主要轨迹后画（上层）；高亮时次要提到上层
  const primaryData = primary === "counterfactual" ? cfData : origData;
  const secondaryData = primary === "counterfactual" ? origData : cfData;
  if (highlightSecondary) {
    drawDataset(primaryData, frame, "primary");
    drawDataset(secondaryData, frame, "highlighted");
  } else {
    drawDataset(secondaryData, frame, "secondary");
    drawDataset(primaryData, frame, "primary");
  }
}

window.TrajectoryViz = TrajectoryViz;
window.renderOverlay = renderOverlay;
window.TEAM_COLORS = TEAM_COLORS;
