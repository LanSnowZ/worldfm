const STATUS_TEXT = {
  not_started: "未开始",
  initializing: "生成中",
  ready: "可探索",
  rendering: "生成下一帧中",
  out_of_bounds: "超出边界",
  error: "出错",
};

const KEY_TO_ACTION = {
  KeyW: "move_forward",
  KeyS: "move_backward",
  KeyA: "move_left",
  KeyD: "move_right",
  ArrowLeft: "turn_left",
  ArrowRight: "turn_right",
  ArrowUp: "look_up",
  ArrowDown: "look_down",
};

const state = {
  sessionId: window.localStorage.getItem("worldfm-session-id"),
  session: null,
  inFlight: false,
  queuedAction: null,
  pendingTimerId: null,
  lastActionAt: 0,
};

const imageInput = document.getElementById("imageInput");
const generateButton = document.getElementById("generateButton");
const statusText = document.getElementById("statusText");
const messageText = document.getElementById("messageText");
const frameImage = document.getElementById("frameImage");
const framePlaceholder = document.getElementById("framePlaceholder");

const poseFields = {
  x: document.getElementById("poseX"),
  y: document.getElementById("poseY"),
  z: document.getElementById("poseZ"),
  yaw_deg: document.getElementById("poseYaw"),
  pitch_deg: document.getElementById("posePitch"),
  radius: document.getElementById("poseRadius"),
};

const limitFields = {
  moveStep: document.getElementById("limitMoveStep"),
  turnStep: document.getElementById("limitTurnStep"),
  radius: document.getElementById("limitRadius"),
  pitch: document.getElementById("limitPitch"),
  interval: document.getElementById("limitInterval"),
};

generateButton.addEventListener("click", () => {
  void createScene();
});

window.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLButtonElement) {
    return;
  }

  const action = KEY_TO_ACTION[event.code];
  if (!action) {
    return;
  }
  if (!state.session) {
    return;
  }

  event.preventDefault();
  queueAction(action);
});

void restoreSession();

async function restoreSession() {
  if (!state.sessionId) {
    return;
  }

  try {
    const session = await fetchJson(`/api/sessions/${state.sessionId}`, {
      method: "GET",
    });
    applySession(session);
  } catch (error) {
    clearSession();
    setStatus("not_started", "未开始", "历史 session 已失效, 请重新生成场景。");
  }
}

async function createScene() {
  const file = imageInput.files?.[0];
  if (!file) {
    setStatus("error", "出错", "请先选择一张图片。");
    return;
  }

  state.inFlight = true;
  state.queuedAction = null;
  setStatus("initializing", "生成中", "正在初始化 panorama、depth、renderer 和 WorldFM 服务。");
  setGenerateDisabled(true);

  try {
    const session = await fetchJson("/api/sessions", {
      method: "POST",
      headers: {
        "Content-Type": file.type || "application/octet-stream",
        "X-Filename": encodeURIComponent(file.name),
      },
      body: file,
    });
    state.lastActionAt = Date.now();
    applySession(session);
  } catch (error) {
    handleError(error);
  } finally {
    state.inFlight = false;
    setGenerateDisabled(false);
  }
}

function queueAction(action) {
  state.queuedAction = action;
  void flushActionQueue();
}

async function flushActionQueue() {
  if (state.inFlight || !state.queuedAction || !state.session) {
    return;
  }

  const minIntervalMs = state.session.limits?.min_request_interval_ms ?? 0;
  const elapsed = Date.now() - state.lastActionAt;
  if (elapsed < minIntervalMs) {
    window.clearTimeout(state.pendingTimerId);
    state.pendingTimerId = window.setTimeout(() => {
      void flushActionQueue();
    }, minIntervalMs - elapsed);
    return;
  }

  const action = state.queuedAction;
  state.queuedAction = null;
  state.inFlight = true;
  setStatus("rendering", "生成下一帧中", "上一帧保留显示, 正在生成最新视角。");

  try {
    const session = await fetchJson(`/api/sessions/${state.session.session_id}/actions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ action }),
    });
    state.lastActionAt = Date.now();
    applySession(session);
  } catch (error) {
    state.lastActionAt = Date.now();
    handleError(error);
  } finally {
    state.inFlight = false;
    if (state.queuedAction) {
      void flushActionQueue();
    }
  }
}

function applySession(session) {
  state.session = session;
  state.sessionId = session.session_id;
  window.localStorage.setItem("worldfm-session-id", session.session_id);

  renderFrame(session.frame_url);
  updatePose(session.pose);
  updateLimits(session.limits);
  setStatus(session.status, STATUS_TEXT[session.status] ?? session.status, session.message);
}

function renderFrame(frameUrl) {
  if (!frameUrl) {
    return;
  }
  frameImage.src = frameUrl;
  frameImage.style.display = "block";
  framePlaceholder.style.display = "none";
}

function updatePose(pose) {
  if (!pose) {
    return;
  }
  poseFields.x.textContent = pose.x.toFixed(2);
  poseFields.y.textContent = pose.y.toFixed(2);
  poseFields.z.textContent = pose.z.toFixed(2);
  poseFields.yaw_deg.textContent = `${pose.yaw_deg.toFixed(2)}°`;
  poseFields.pitch_deg.textContent = `${pose.pitch_deg.toFixed(2)}°`;
  poseFields.radius.textContent = pose.radius.toFixed(2);
}

function updateLimits(limits) {
  if (!limits) {
    return;
  }
  limitFields.moveStep.textContent = `${limits.move_step.toFixed(2)} m`;
  limitFields.turnStep.textContent = `${limits.turn_step_deg.toFixed(1)}°`;
  limitFields.radius.textContent = `${limits.max_radius.toFixed(2)} m`;
  limitFields.pitch.textContent = `${limits.min_pitch_deg.toFixed(0)}° ~ ${limits.max_pitch_deg.toFixed(0)}°`;
  limitFields.interval.textContent = `${limits.min_request_interval_ms} ms`;
}

function setStatus(statusKey, label, message) {
  statusText.textContent = label;
  statusText.dataset.status = statusKey;
  messageText.textContent = message;
}

function clearSession() {
  state.session = null;
  state.sessionId = null;
  window.localStorage.removeItem("worldfm-session-id");
}

function handleError(error) {
  const message = error?.message ?? "请求失败";
  const code = error?.code ?? "error";

  if (code === "session_not_found") {
    clearSession();
  }

  const statusKey = code === "out_of_bounds" ? "out_of_bounds" : "error";
  setStatus(statusKey, STATUS_TEXT[statusKey], message);
}

function setGenerateDisabled(disabled) {
  generateButton.disabled = disabled;
  imageInput.disabled = disabled;
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => {
    return {};
  });

  if (!response.ok) {
    const error = new Error(data.message || "请求失败");
    error.code = data.code || "request_failed";
    throw error;
  }
  return data;
}
