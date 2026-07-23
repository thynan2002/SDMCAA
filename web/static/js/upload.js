/* ═══════════════════════════════════════════════════════════
   upload.js — 数据上传模块
   ═══════════════════════════════════════════════════════════ */

const Upload = (() => {
  let personFile = null, ballFile = null;
  const personInput = document.getElementById("person-input");
  const ballInput = document.getElementById("ball-input");
  const personName = document.getElementById("person-name");
  const ballName = document.getElementById("ball-name");
  const uploadBtn = document.getElementById("upload-btn");
  const statusEl = document.getElementById("upload-status");

  function refreshBtn() {
    uploadBtn.disabled = !(personFile && ballFile);
  }

  personInput.addEventListener("change", (e) => {
    personFile = e.target.files[0] || null;
    if (personFile) {
      personName.textContent = personFile.name;
      personInput.closest(".upload-item").querySelector(".upload-slot").classList.add("loaded");
    }
    refreshBtn();
  });

  ballInput.addEventListener("change", (e) => {
    ballFile = e.target.files[0] || null;
    if (ballFile) {
      ballName.textContent = ballFile.name;
      ballInput.closest(".upload-item").querySelector(".upload-slot").classList.add("loaded");
    }
    refreshBtn();
  });

  async function upload() {
    if (!personFile || !ballFile) return;
    uploadBtn.disabled = true;
    statusEl.className = "upload-status";
    statusEl.textContent = "上传中…";

    const fd = new FormData();
    fd.append("person_file", personFile);
    fd.append("ball_file", ballFile);

    try {
      const res = await fetch("/api/upload", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) {
        statusEl.className = "upload-status err";
        statusEl.textContent = data.error || "上传失败";
        uploadBtn.disabled = false;
        return null;
      }
      statusEl.className = "upload-status ok";
      statusEl.textContent = `已加载：${data.player_count} 名球员（门将 ${data.goalkeeper_count}），时长 ${data.duration_seconds}s`;
      if (window.App) window.App.onDataLoaded(data);
      return data;
    } catch (err) {
      statusEl.className = "upload-status err";
      statusEl.textContent = "网络错误：" + err.message;
      uploadBtn.disabled = false;
      return null;
    }
  }

  uploadBtn.addEventListener("click", upload);

  return { upload, getPersonFile: () => personFile, getBallFile: () => ballFile };
})();
window.Upload = Upload;
