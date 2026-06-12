const state = {
  classes: {},
  selectedClassIds: new Set(),
  similar: {},
  selectedSimilarId: null,
  openaiConfigured: false,
  rewards: null,
  chatRoom: "",
  chatName: "",
  chatTimer: null,
  localStream: null,
  videoSocket: null,
  videoRoom: "",
  videoPeerId: "",
  peers: new Map(),
};

const el = (id) => document.getElementById(id);

function setStatus(message, type = "info") {
  const node = el("status");
  node.textContent = message;
  node.dataset.type = type;
}

function showError(error) {
  const message = error?.message || String(error);
  setStatus(message, "error");
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function displayTime(ms) {
  if (!ms) return "";
  return new Date(ms).toLocaleString();
}

async function apiJson(url, options = {}) {
  const response = await fetch(url, options);
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    const detail = payload?.detail || payload || response.statusText;
    throw new Error(detail);
  }
  return payload;
}

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((tab) => tab.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.remove("active"));
    button.classList.add("active");
    el(`tab-${button.dataset.tab}`).classList.add("active");
  });
});

function renderPreview(path, type) {
  const box = el("detectedPreview");
  box.classList.remove("empty");
  box.innerHTML = "";

  if (type === "video") {
    const video = document.createElement("video");
    video.src = path;
    video.controls = true;
    video.playsInline = true;
    box.appendChild(video);
  } else {
    const img = document.createElement("img");
    img.src = path;
    img.alt = "Detected output";
    box.appendChild(img);
  }
}

function renderClasses() {
  const list = el("classesList");
  list.innerHTML = "";
  state.selectedClassIds.clear();

  const entries = Object.entries(state.classes);
  if (!entries.length) {
    list.className = "check-list empty";
    list.textContent = "No classes detected. Try another clearer image.";
    el("similarBtn").disabled = true;
    return;
  }

  list.className = "check-list";
  for (const [label, id] of entries) {
    const item = document.createElement("label");
    item.className = "check-row";
    item.innerHTML = `<input type="checkbox" value="${id}"><span>${escapeHtml(label)}</span>`;
    item.querySelector("input").addEventListener("change", (event) => {
      const value = Number(event.target.value);
      if (event.target.checked) state.selectedClassIds.add(value);
      else state.selectedClassIds.delete(value);
      el("similarBtn").disabled = state.selectedClassIds.size === 0;
    });
    list.appendChild(item);
  }
}

function selectSimilar(id) {
  state.selectedSimilarId = id === null || id === undefined ? null : Number(id);
  const item = state.selectedSimilarId === null ? null : state.similar[id];
  document.querySelectorAll(".catalog-card").forEach((card) => {
    card.classList.toggle("selected", state.selectedSimilarId !== null && Number(card.dataset.id) === state.selectedSimilarId);
  });
  el("selectedLabel").textContent = item ? `${item.class_name} #${id}` : "None selected";
  el("askBtn").disabled = !item || !state.openaiConfigured;
  el("tryOnBtn").disabled = !item || !state.openaiConfigured;
}

function renderSimilar() {
  const grid = el("similarGrid");
  grid.innerHTML = "";
  const entries = Object.entries(state.similar);

  if (!entries.length) {
    grid.className = "catalog-grid empty";
    grid.textContent = "No similar catalog items found.";
    return;
  }

  grid.className = "catalog-grid";
  for (const [id, item] of entries) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "catalog-card";
    card.dataset.id = id;
    card.innerHTML = `
      <img src="${item.image_file_name}" alt="${escapeHtml(item.class_name)}">
      <span>${escapeHtml(item.class_name)}</span>
      <small>score ${Number(item.distance).toFixed(3)}</small>
    `;
    card.addEventListener("click", () => selectSimilar(id));
    grid.appendChild(card);
  }
  el("catalogHint").textContent = `${entries.length} results`;
}

el("detectForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const file = el("detectFile").files[0];
    if (!file) return;
    setStatus("Detecting fashion items...", "info");

    const form = new FormData();
    form.append("file", file);
    const data = await apiJson("/detect_classes", { method: "POST", body: form });
    state.classes = data.classes || {};
    state.similar = {};
    state.selectedSimilarId = null;

    el("fileTypeLabel").textContent = data.file_type || "image";
    renderPreview(data.image, data.file_type);
    renderClasses();
    renderSimilar();
    selectSimilar(null);
    setStatus("Detection complete", "ok");
  } catch (error) {
    showError(error);
  }
});

el("similarBtn").addEventListener("click", async () => {
  try {
    setStatus("Finding similar catalog items...", "info");
    const ids = Array.from(state.selectedClassIds);
    const data = await apiJson("/fetch_similar_images", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(ids),
    });
    state.similar = data.similar_images || {};
    renderSimilar();
    setStatus("Catalog matches ready", "ok");
  } catch (error) {
    showError(error);
  }
});

el("askForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.selectedSimilarId === null) return;
  try {
    setStatus("Asking AI...", "info");
    const data = await apiJson("/ask_me", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        selected_image_id: state.selectedSimilarId,
        text: el("questionText").value,
      }),
    });
    const answer = el("answerBox");
    answer.classList.remove("empty");
    answer.textContent = data.answer || "No answer returned.";
    setStatus("Answer ready", "ok");
  } catch (error) {
    showError(error);
  }
});

el("tryOnForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.selectedSimilarId === null) return;
  try {
    const file = el("tryOnFile").files[0];
    if (!file) return;
    setStatus("Generating try-on image...", "info");
    const form = new FormData();
    form.append("selected_image_id", state.selectedSimilarId);
    form.append("file", file);
    const data = await apiJson("/imerse_image", { method: "POST", body: form });

    const box = el("tryOnPreview");
    box.classList.remove("empty");
    box.innerHTML = data.image_details
      ? `<img src="${data.image_details}" alt="Generated try-on">`
      : "The selected item class was not detected in the try-on image.";
    setStatus("Try-on complete", "ok");
  } catch (error) {
    showError(error);
  }
});

function renderRewards() {
  const data = state.rewards || { challenges: {}, uploads: [] };
  const challenges = Object.values(data.challenges || {});
  const select = el("rewardChallenge");
  select.innerHTML = challenges.map((challenge) => (
    `<option value="${escapeHtml(challenge.id)}">${escapeHtml(challenge.title)}</option>`
  )).join("");

  el("challengeList").className = "challenge-list";
  el("challengeList").innerHTML = challenges.map((challenge) => {
    const count = (data.uploads || []).filter((upload) => upload.challenge_id === challenge.id).length;
    return `<div class="challenge-row"><strong>${escapeHtml(challenge.title)}</strong><span>${count} looks</span></div>`;
  }).join("");

  const uploads = data.uploads || [];
  const coins = uploads.reduce((total, upload) => total + 10 + Number(upload.likes || 0) + (upload.comments || []).length, 0);
  el("coinBalance").textContent = `${coins} coins`;

  const feed = el("rewardFeed");
  if (!uploads.length) {
    feed.className = "feed empty";
    feed.textContent = "No uploads yet.";
    return;
  }

  feed.className = "feed";
  feed.innerHTML = uploads.map((upload) => {
    const comments = (upload.comments || []).map((comment) => (
      `<div class="comment"><strong>${escapeHtml(comment.author)}</strong> ${escapeHtml(comment.text)}</div>`
    )).join("");
    return `
      <article class="feed-card" data-upload-id="${escapeHtml(upload.id)}">
        <img src="${escapeHtml(upload.image_url)}" alt="${escapeHtml(upload.title || "Challenge upload")}">
        <div class="feed-body">
          <h3>${escapeHtml(upload.title || "Untitled look")}</h3>
          <p class="muted">By ${escapeHtml(upload.author)} · ${displayTime(upload.created_at)}</p>
          <div class="feed-actions">
            <button type="button" data-like="${escapeHtml(upload.id)}">Like (${Number(upload.likes || 0)})</button>
          </div>
          <div class="comments">${comments || '<span class="muted">No comments yet.</span>'}</div>
          <form class="comment-form" data-comment-form="${escapeHtml(upload.id)}">
            <input type="text" name="comment" maxlength="500" placeholder="Add a comment">
            <button type="submit">Comment</button>
          </form>
        </div>
      </article>
    `;
  }).join("");

  feed.querySelectorAll("[data-like]").forEach((button) => {
    button.addEventListener("click", async () => {
      await apiJson(`/api/rewards/uploads/${button.dataset.like}/like`, { method: "POST" });
      await loadRewards();
    });
  });

  feed.querySelectorAll("[data-comment-form]").forEach((form) => {
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const input = form.elements.comment;
      const text = input.value.trim();
      if (!text) return;
      await apiJson(`/api/rewards/uploads/${form.dataset.commentForm}/comments`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ author: el("rewardAuthor").value || "Guest", text }),
      });
      input.value = "";
      await loadRewards();
    });
  });
}

async function loadRewards() {
  state.rewards = await apiJson("/api/rewards");
  renderRewards();
}

el("rewardForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const form = new FormData();
    form.append("challenge_id", el("rewardChallenge").value);
    form.append("author", el("rewardAuthor").value || "Guest");
    form.append("title", el("rewardTitle").value);
    form.append("image_url", el("rewardUrl").value);
    const file = el("rewardFile").files[0];
    if (file) form.append("file", file);
    await apiJson("/api/rewards/uploads", { method: "POST", body: form });
    el("rewardTitle").value = "";
    el("rewardUrl").value = "";
    el("rewardFile").value = "";
    await loadRewards();
    setStatus("Challenge upload saved", "ok");
  } catch (error) {
    showError(error);
  }
});

el("refreshRewardsBtn").addEventListener("click", () => loadRewards().catch(showError));

async function loadChat() {
  if (!state.chatRoom) return;
  const data = await apiJson(`/api/chat/rooms/${encodeURIComponent(state.chatRoom)}/messages`);
  const box = el("chatMessages");
  const messages = data.messages || [];
  if (!messages.length) {
    box.className = "messages empty";
    box.textContent = "No messages yet.";
    return;
  }
  box.className = "messages";
  box.innerHTML = messages.map((message) => `
    <div class="message ${message.sender === state.chatName ? "mine" : ""}">
      <strong>${escapeHtml(message.sender)}</strong>
      <span>${escapeHtml(message.text)}</span>
      <small>${displayTime(message.created_at)}</small>
    </div>
  `).join("");
  box.scrollTop = box.scrollHeight;
}

el("chatSetupForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  state.chatName = el("chatName").value.trim() || "Guest";
  state.chatRoom = el("chatRoom").value.trim() || "style-room";
  el("chatRoomTitle").textContent = `Messages: ${state.chatRoom}`;
  el("chatMessageInput").disabled = false;
  el("sendChatBtn").disabled = false;
  clearInterval(state.chatTimer);
  await loadChat();
  state.chatTimer = setInterval(() => loadChat().catch(() => {}), 3000);
  setStatus("Chat room opened", "ok");
});

el("chatMessageForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = el("chatMessageInput");
  const text = input.value.trim();
  if (!text || !state.chatRoom) return;
  await apiJson(`/api/chat/rooms/${encodeURIComponent(state.chatRoom)}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sender: state.chatName || "Guest", text }),
  });
  input.value = "";
  await loadChat();
});

el("refreshChatBtn").addEventListener("click", () => loadChat().catch(showError));

function setVideoControls(connected) {
  el("joinVideoBtn").disabled = connected;
  el("leaveVideoBtn").disabled = !connected;
  el("toggleMicBtn").disabled = !connected;
  el("toggleCamBtn").disabled = !connected;
  el("videoStatus").textContent = connected ? `Connected to ${state.videoRoom}` : "Not connected";
}

function peerConnection(peerId) {
  if (state.peers.has(peerId)) return state.peers.get(peerId);
  const pc = new RTCPeerConnection({
    iceServers: [{ urls: ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"] }],
  });
  state.localStream?.getTracks().forEach((track) => pc.addTrack(track, state.localStream));
  pc.onicecandidate = (event) => {
    if (event.candidate) sendSignal({ type: "ice", to: peerId, candidate: event.candidate });
  };
  pc.ontrack = (event) => attachRemote(peerId, event.streams[0]);
  pc.onconnectionstatechange = () => {
    if (["failed", "closed", "disconnected"].includes(pc.connectionState)) removePeer(peerId);
  };
  state.peers.set(peerId, pc);
  return pc;
}

function attachRemote(peerId, stream) {
  let tile = document.querySelector(`[data-peer-tile="${CSS.escape(peerId)}"]`);
  if (!tile) {
    tile = document.createElement("div");
    tile.className = "video-tile";
    tile.dataset.peerTile = peerId;
    tile.innerHTML = `<video autoplay playsinline></video><span>${escapeHtml(peerId)}</span>`;
    el("remoteVideos").appendChild(tile);
  }
  tile.querySelector("video").srcObject = stream;
}

function removePeer(peerId) {
  const pc = state.peers.get(peerId);
  if (pc) pc.close();
  state.peers.delete(peerId);
  document.querySelector(`[data-peer-tile="${CSS.escape(peerId)}"]`)?.remove();
}

function sendSignal(payload) {
  if (state.videoSocket?.readyState === WebSocket.OPEN) {
    state.videoSocket.send(JSON.stringify(payload));
  }
}

async function createOfferFor(peerId) {
  const pc = peerConnection(peerId);
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  sendSignal({ type: "offer", to: peerId, sdp: offer });
}

async function handleSignal(payload) {
  const peerId = payload.from;
  if (!peerId || peerId === state.videoPeerId) return;
  if (payload.type === "peer-joined") {
    await createOfferFor(peerId);
    return;
  }
  if (payload.type === "peer-left") {
    removePeer(peerId);
    return;
  }
  if (payload.type === "offer") {
    const pc = peerConnection(peerId);
    await pc.setRemoteDescription(new RTCSessionDescription(payload.sdp));
    const answer = await pc.createAnswer();
    await pc.setLocalDescription(answer);
    sendSignal({ type: "answer", to: peerId, sdp: answer });
    return;
  }
  if (payload.type === "answer") {
    const pc = peerConnection(peerId);
    await pc.setRemoteDescription(new RTCSessionDescription(payload.sdp));
    return;
  }
  if (payload.type === "ice" && payload.candidate) {
    const pc = peerConnection(peerId);
    await pc.addIceCandidate(new RTCIceCandidate(payload.candidate));
  }
}

async function joinVideo(room, name) {
  state.videoRoom = room || "style-room";
  state.videoPeerId = `${name || "guest"}-${Math.random().toString(16).slice(2, 8)}`.replace(/[^a-zA-Z0-9_-]/g, "-");
  state.localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
  el("localVideo").srcObject = state.localStream;

  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  state.videoSocket = new WebSocket(`${protocol}://${window.location.host}/ws/video/${encodeURIComponent(state.videoRoom)}/${encodeURIComponent(state.videoPeerId)}`);
  state.videoSocket.onopen = () => setVideoControls(true);
  state.videoSocket.onmessage = (event) => handleSignal(JSON.parse(event.data)).catch(showError);
  state.videoSocket.onclose = () => leaveVideo(false);
}

function leaveVideo(closeSocket = true) {
  if (closeSocket && state.videoSocket) state.videoSocket.close();
  state.videoSocket = null;
  state.localStream?.getTracks().forEach((track) => track.stop());
  state.localStream = null;
  el("localVideo").srcObject = null;
  state.peers.forEach((pc) => pc.close());
  state.peers.clear();
  el("remoteVideos").innerHTML = "";
  setVideoControls(false);
}

el("videoSetupForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await joinVideo(el("videoRoom").value.trim(), el("videoName").value.trim() || "guest");
    setStatus("Video room joined", "ok");
  } catch (error) {
    showError(error);
  }
});

el("leaveVideoBtn").addEventListener("click", () => leaveVideo(true));
el("toggleMicBtn").addEventListener("click", () => {
  state.localStream?.getAudioTracks().forEach((track) => { track.enabled = !track.enabled; });
});
el("toggleCamBtn").addEventListener("click", () => {
  state.localStream?.getVideoTracks().forEach((track) => { track.enabled = !track.enabled; });
});

async function boot() {
  try {
    const health = await apiJson("/api/health");
    state.openaiConfigured = Boolean(health.openai_configured);
    const modelNote = health.fashion_detector_loaded ? "" : ", fallback detection";
    setStatus(
      `${health.catalog_images} catalog images, ${health.embeddings} embeddings${modelNote}${state.openaiConfigured ? "" : ", AI generation disabled"}`,
      state.openaiConfigured && health.fashion_detector_loaded ? "ok" : "warn"
    );
    await loadRewards();
    setVideoControls(false);
  } catch (error) {
    showError(error);
  }
}

boot();
