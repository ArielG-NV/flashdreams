// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const connectButton = document.getElementById("connectButton")
const statusText = document.getElementById("statusText")
const remoteVideo = document.getElementById("remoteVideo")
const viewLabel = document.getElementById("viewLabel")
const controlHint = document.getElementById("controlHint")
const messageText = document.getElementById("messageText")
const averageFpsValue = document.getElementById("averageFpsValue")
const fpsValue = document.getElementById("fpsValue")
const latencyValue = document.getElementById("latencyValue")
const stepValue = document.getElementById("stepValue")
const resolutionValue = document.getElementById("resolutionValue")
const modelValue = document.getElementById("modelValue")
const playerGrid = document.getElementById("playerGrid")
const controlGrid = document.getElementById("controlGrid")
const copyLinkButton = document.getElementById("copyLinkButton")

const keySources = new Map()
const allowedKeys = new Set()
const keyAliases = new Map()
let playerCards = []
let controlButtons = []
let modelConfig = null
let peerConnection = null
let controlChannel = null
let heartbeatTimer = null
let frameRateTimer = null
let initialDecodedFrames = null
let initialStatsTimestamp = null
let previousDecodedFrames = null
let previousStatsTimestamp = null
let connected = false
let connecting = false
let selectedSeat = null
let pendingSeat = null

function normalizeKey(rawKey) {
  const key = String(rawKey || "").toLowerCase()
  return keyAliases.get(key) || key.trim()
}

function eventKey(event) {
  const fromCode = normalizeKey(event.code)
  return allowedKeys.has(fromCode) ? fromCode : normalizeKey(event.key)
}

function setStatus(label, state) {
  statusText.textContent = label
  document.body.dataset.status = state
}

function setMessage(message) {
  messageText.textContent = message
}

function updateSessionButton() {
  const label = connectButton.querySelector("span")
  if (!connected) label.textContent = "Reconnect preview"
  else if (selectedSeat === null) label.textContent = "Disconnect Preview"
  else label.textContent = "End Session"
}

function copyTextFallback(text) {
  const textArea = document.createElement("textarea")
  textArea.value = text
  textArea.setAttribute("readonly", "")
  textArea.style.position = "fixed"
  textArea.style.opacity = "0"
  textArea.style.pointerEvents = "none"
  document.body.appendChild(textArea)
  textArea.select()
  textArea.setSelectionRange(0, text.length)
  const copied = document.execCommand("copy")
  textArea.remove()
  return copied
}

async function copyInviteLink() {
  const link = window.location.href
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(link)
    return true
  }
  return copyTextFallback(link)
}

function groupInputs(inputs) {
  const groups = new Map()
  for (const input of inputs) {
    if (!groups.has(input.group)) {
      groups.set(input.group, {
        name: input.group,
        description: input.groupDescription,
        inputs: [],
      })
    }
    groups.get(input.group).inputs.push(input)
  }
  return Array.from(groups.values())
}

function buildConfiguredUI(config) {
  modelValue.textContent = config.displayName
  resolutionValue.textContent = `${config.video.width} × ${config.video.height}`
  playerGrid.style.setProperty("--player-columns", config.previewGrid.columns)
  playerGrid.replaceChildren()
  for (let seat = 0; seat < config.playerCount; seat += 1) {
    const card = document.createElement("button")
    card.className = "playerCard"
    card.type = "button"
    card.dataset.playerSeat = String(seat)
    const number = document.createElement("b")
    number.textContent = String(seat + 1).padStart(2, "0")
    const label = document.createElement("span")
    label.textContent = `Player ${seat + 1}`
    const availability = document.createElement("small")
    availability.textContent = "AVAILABLE"
    card.append(number, label, availability)
    card.addEventListener("click", () => selectPlayer(seat))
    playerGrid.appendChild(card)
  }
  playerCards = Array.from(playerGrid.querySelectorAll("[data-player-seat]"))

  allowedKeys.clear()
  keyAliases.clear()
  controlGrid.replaceChildren()
  for (const input of config.inputs) {
    allowedKeys.add(input.key)
    keyAliases.set(input.key, input.key)
    for (const alias of input.aliases) keyAliases.set(String(alias).toLowerCase(), input.key)
  }
  for (const [index, group] of groupInputs(config.inputs).entries()) {
    const article = document.createElement("article")
    article.className = "controlGroup"
    const number = document.createElement("span")
    number.className = "groupNumber"
    number.textContent = String(index + 1).padStart(2, "0")
    const copy = document.createElement("div")
    const heading = document.createElement("h3")
    heading.textContent = group.name
    const description = document.createElement("p")
    description.textContent = group.description
    copy.append(heading, description)
    const keys = document.createElement("div")
    keys.className = "keys configuredKeys"
    for (const input of group.inputs) {
      const button = document.createElement("button")
      button.type = "button"
      button.dataset.controlKey = input.key
      button.setAttribute("aria-label", input.action)
      button.title = input.action
      const keyLabel = document.createElement("small")
      keyLabel.textContent = input.label
      const actionLabel = document.createElement("span")
      actionLabel.textContent = input.action
      button.append(keyLabel, actionLabel)
      keys.appendChild(button)
    }
    article.append(number, copy, keys)
    controlGrid.appendChild(article)
  }
  controlButtons = Array.from(controlGrid.querySelectorAll("[data-control-key]"))
  wireControlButtons()
  setControlsEnabled(false)
}

function setControlsEnabled(enabled) {
  for (const button of controlButtons) button.disabled = !enabled
  controlHint.textContent = enabled
    ? `Inputs are being sent to player ${selectedSeat + 1}.`
    : "Choose an available player to enable controls."
}

async function refreshRoom() {
  try {
    const response = await fetch("/api/mira/room", { cache: "no-store" })
    if (!response.ok) return
    const room = await response.json()
    for (const card of playerCards) {
      const seat = Number(card.dataset.playerSeat)
      const occupied = Boolean(room.players?.find((player) => player.seat === seat)?.occupied)
      const yours = selectedSeat === seat
      card.disabled = (occupied && !yours) || pendingSeat !== null || yours
      card.classList.toggle("is-occupied", occupied && !yours)
      card.classList.toggle("is-selected", yours)
      card.querySelector("small").textContent = yours ? "YOU" : occupied ? "CONTROLLED" : "AVAILABLE"
    }
  } catch {
    // Occupancy polling is advisory; the server still claims seats atomically.
  }
}

function selectPlayer(seat) {
  if (!connected || !controlChannel || controlChannel.readyState !== "open") {
    setMessage("The all-player preview is still connecting.")
    return
  }
  if (selectedSeat !== null || pendingSeat !== null) return
  pendingSeat = seat
  sendMessage({ type: "claim", seat })
  setMessage(`Claiming player ${seat + 1}…`)
  refreshRoom()
}

function formatMilliseconds(value) {
  const number = Number(value)
  if (!Number.isFinite(number)) return "—"
  return number >= 1000 ? `${(number / 1000).toFixed(1)} s` : `${Math.round(number)} ms`
}

async function sampleActualFrameRate() {
  const pc = peerConnection
  if (!pc || pc.connectionState !== "connected") {
    averageFpsValue.textContent = "—"
    fpsValue.textContent = "—"
    return
  }
  try {
    const stats = await pc.getStats()
    let inboundVideo = null
    stats.forEach((report) => {
      const video = report.kind === "video" || report.mediaType === "video"
      if (report.type === "inbound-rtp" && video) inboundVideo = report
    })
    if (!inboundVideo) {
      averageFpsValue.textContent = "—"
      fpsValue.textContent = "—"
      return
    }
    const decodedFrames = Number(inboundVideo.framesDecoded)
    const timestamp = Number(inboundVideo.timestamp)
    let actualFps = Number(inboundVideo.framesPerSecond)
    let averageFps = Number.NaN
    if (
      Number.isFinite(decodedFrames) &&
      Number.isFinite(timestamp)
    ) {
      if (initialDecodedFrames === null || initialStatsTimestamp === null) {
        initialDecodedFrames = decodedFrames
        initialStatsTimestamp = timestamp
      } else if (timestamp > initialStatsTimestamp) {
        averageFps = (decodedFrames - initialDecodedFrames) * 1000 / (timestamp - initialStatsTimestamp)
      }
    }
    if (
      !Number.isFinite(actualFps) &&
      Number.isFinite(decodedFrames) &&
      Number.isFinite(timestamp) &&
      previousDecodedFrames !== null &&
      previousStatsTimestamp !== null &&
      timestamp > previousStatsTimestamp
    ) {
      actualFps = (decodedFrames - previousDecodedFrames) * 1000 / (timestamp - previousStatsTimestamp)
    }
    if (Number.isFinite(decodedFrames)) previousDecodedFrames = decodedFrames
    if (Number.isFinite(timestamp)) previousStatsTimestamp = timestamp
    averageFpsValue.textContent = Number.isFinite(averageFps) ? `${averageFps.toFixed(1)} fps` : "—"
    fpsValue.textContent = Number.isFinite(actualFps) ? `${actualFps.toFixed(1)} fps` : "—"
  } catch {
    averageFpsValue.textContent = "—"
    fpsValue.textContent = "—"
  }
}

function startFrameRateMonitor() {
  if (frameRateTimer !== null) window.clearInterval(frameRateTimer)
  initialDecodedFrames = null
  initialStatsTimestamp = null
  previousDecodedFrames = null
  previousStatsTimestamp = null
  sampleActualFrameRate()
  frameRateTimer = window.setInterval(sampleActualFrameRate, 1000)
}

function stopFrameRateMonitor() {
  if (frameRateTimer !== null) window.clearInterval(frameRateTimer)
  frameRateTimer = null
  initialDecodedFrames = null
  initialStatsTimestamp = null
  previousDecodedFrames = null
  previousStatsTimestamp = null
  averageFpsValue.textContent = "—"
  fpsValue.textContent = "—"
}

function sendMessage(payload) {
  if (!controlChannel || controlChannel.readyState !== "open") return false
  controlChannel.send(JSON.stringify(payload))
  return true
}

function sendAction(event, key) {
  if (!sendMessage({ type: "action", action: { event, key } })) return
  setStatus("Generating", "streaming")
  setMessage(`${event === "keydown" ? "Pressed" : "Released"} ${key.toUpperCase()}`)
}

function activeKeys() {
  return Array.from(keySources.entries())
    .filter(([, sources]) => sources.size > 0)
    .map(([key]) => key)
}

function updateControlHighlights() {
  const active = new Set(activeKeys())
  for (const button of controlButtons) {
    const held = active.has(button.dataset.controlKey)
    button.classList.toggle("is-active", held)
    button.setAttribute("aria-pressed", held ? "true" : "false")
  }
}

function setKeyHeld(key, source, held) {
  const normalized = normalizeKey(key)
  if (!allowedKeys.has(normalized) || selectedSeat === null) return
  let sources = keySources.get(normalized)
  if (!sources) {
    sources = new Set()
    keySources.set(normalized, sources)
  }
  const wasHeld = sources.size > 0
  if (held) sources.add(source)
  else sources.delete(source)
  const isHeld = sources.size > 0
  updateControlHighlights()
  if (!wasHeld && isHeld) sendAction("keydown", normalized)
  if (wasHeld && !isHeld) sendAction("keyup", normalized)
}

function releaseAllKeys() {
  for (const [key, sources] of keySources.entries()) {
    if (sources.size === 0) continue
    sources.clear()
    sendAction("keyup", key)
  }
  updateControlHighlights()
}

function updateMetrics(payload) {
  latencyValue.textContent = formatMilliseconds(payload.gen_ms)
  if (Number.isFinite(Number(payload.chunk_index))) stepValue.textContent = String(payload.chunk_index)
  if (payload.model) modelValue.textContent = payload.model
  if (payload.resolution?.width && payload.resolution?.height) {
    resolutionValue.textContent = `${payload.resolution.width} × ${payload.resolution.height}`
  }
}

function handleControlMessage(rawMessage) {
  let payload
  try {
    payload = JSON.parse(rawMessage)
  } catch {
    setMessage("Received an invalid server message.")
    return
  }
  if (payload.type === "seat_claimed") {
    selectedSeat = Number(payload.seat)
    pendingSeat = null
    document.body.dataset.viewMode = "player"
    setControlsEnabled(true)
    setStatus("Ready", "ready")
    viewLabel.textContent = `PLAYER ${selectedSeat + 1} · LIVE WORLD`
    setMessage(`Controlling player ${selectedSeat + 1}. The world is generating now.`)
    updateSessionButton()
    refreshRoom()
    return
  }
  if (payload.type === "seat_released") {
    selectedSeat = null
    pendingSeat = null
    keySources.clear()
    updateControlHighlights()
    setControlsEnabled(false)
    document.body.dataset.viewMode = "preview"
    setStatus("Preview", "ready")
    viewLabel.textContent = "ALL PLAYERS · LIVE PREVIEW"
    setMessage("Session ended. Live all-player preview restored.")
    connectButton.disabled = false
    updateSessionButton()
    refreshRoom()
    return
  }
  if (payload.type === "seat_claim_failed") {
    pendingSeat = null
    setMessage(payload.message || "That player is no longer available.")
    refreshRoom()
    return
  }
  if (payload.type === "error") {
    setStatus("Error", "error")
    setMessage(payload.message || "The runtime reported an error.")
    return
  }
  if (payload.type === "chunk_done") updateMetrics(payload)
}

async function waitForIceGatheringComplete(pc) {
  if (pc.iceGatheringState === "complete") return
  await new Promise((resolve) => {
    const onStateChange = () => {
      if (pc.iceGatheringState !== "complete") return
      pc.removeEventListener("icegatheringstatechange", onStateChange)
      resolve()
    }
    pc.addEventListener("icegatheringstatechange", onStateChange)
  })
}

function wirePeerConnection(pc) {
  pc.addTransceiver("video", { direction: "recvonly" })
  pc.addEventListener("track", (event) => {
    connected = true
    remoteVideo.srcObject = event.streams[0] || new MediaStream([event.track])
    remoteVideo.play().catch(() => {})
    document.body.classList.add("has-video")
    viewLabel.textContent = selectedSeat === null ? "ALL PLAYERS · LIVE PREVIEW" : `PLAYER ${selectedSeat + 1} · LIVE WORLD`
    updateSessionButton()
  })
  pc.addEventListener("connectionstatechange", () => {
    if (pc.connectionState === "connected") {
      connected = true
      startFrameRateMonitor()
      setStatus("Preview", "ready")
      setMessage("Live all-player preview connected. Choose an available player.")
      updateSessionButton()
    }
    if (["failed", "disconnected", "closed"].includes(pc.connectionState)) disconnect(false)
  })
}

function wireControlChannel(channel) {
  channel.addEventListener("open", () => {
    connected = true
    updateSessionButton()
    heartbeatTimer = window.setInterval(() => sendMessage({ type: "heartbeat" }), 2000)
    refreshRoom()
  })
  channel.addEventListener("message", (event) => handleControlMessage(event.data))
  channel.addEventListener("close", () => disconnect(false))
}

async function connectPreview() {
  if (connecting || connected) return
  connecting = true
  connectButton.disabled = true
  setStatus("Negotiating", "idle")
  setMessage("Connecting the live all-player preview…")
  try {
    peerConnection = new RTCPeerConnection()
    wirePeerConnection(peerConnection)
    controlChannel = peerConnection.createDataChannel("mira-controls", { ordered: true })
    wireControlChannel(controlChannel)
    const offer = await peerConnection.createOffer()
    await peerConnection.setLocalDescription(offer)
    await waitForIceGatheringComplete(peerConnection)
    const response = await fetch("/api/mira/offer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(peerConnection.localDescription.toJSON()),
    })
    if (!response.ok) {
      const detail = await response.text()
      throw new Error(detail || `Offer failed (${response.status}).`)
    }
    await peerConnection.setRemoteDescription(await response.json())
  } catch (error) {
    disconnect(false)
    setStatus("Error", "error")
    setMessage(error instanceof Error ? error.message : String(error))
  } finally {
    connecting = false
    connectButton.disabled = false
  }
}

function disconnect(notifyServer = true) {
  if (notifyServer) {
    releaseAllKeys()
    sendMessage({ type: "disconnect" })
  }
  connected = false
  selectedSeat = null
  pendingSeat = null
  stepValue.textContent = "—"
  latencyValue.textContent = "—"
  stopFrameRateMonitor()
  if (heartbeatTimer !== null) window.clearInterval(heartbeatTimer)
  heartbeatTimer = null
  const channel = controlChannel
  const peer = peerConnection
  controlChannel = null
  peerConnection = null
  if (channel) channel.close()
  if (peer) peer.close()
  remoteVideo.srcObject = null
  document.body.classList.remove("has-video")
  document.body.dataset.viewMode = "preview"
  viewLabel.textContent = "WORLD STREAM STANDBY"
  keySources.clear()
  updateControlHighlights()
  setControlsEnabled(false)
  setStatus("Offline", "idle")
  updateSessionButton()
  refreshRoom()
}

function endSession() {
  if (selectedSeat === null) return
  releaseAllKeys()
  if (!sendMessage({ type: "release" })) {
    setMessage("The session control channel is unavailable.")
    return
  }
  connectButton.disabled = true
  setMessage(`Ending player ${selectedSeat + 1} session…`)
}

function wireControlButtons() {
  for (const button of controlButtons) {
    const key = button.dataset.controlKey
    button.addEventListener("pointerdown", (event) => {
      event.preventDefault()
      button.setPointerCapture(event.pointerId)
      setKeyHeld(key, `pointer:${event.pointerId}`, true)
    })
    const releasePointer = (event) => setKeyHeld(key, `pointer:${event.pointerId}`, false)
    button.addEventListener("pointerup", releasePointer)
    button.addEventListener("pointercancel", releasePointer)
    button.addEventListener("lostpointercapture", releasePointer)
  }
}

async function initialize() {
  try {
    const response = await fetch("/api/mira/config", { cache: "no-store" })
    if (!response.ok) throw new Error(`Config failed (${response.status}).`)
    modelConfig = await response.json()
    buildConfiguredUI(modelConfig)
    document.body.dataset.viewMode = "preview"
    await refreshRoom()
    await connectPreview()
  } catch (error) {
    setStatus("Error", "error")
    setMessage(error instanceof Error ? error.message : String(error))
  }
}

connectButton.addEventListener("click", () => {
  if (!connected) connectPreview()
  else if (selectedSeat === null) disconnect(true)
  else endSession()
})

copyLinkButton.addEventListener("click", async () => {
  try {
    const copied = await copyInviteLink()
    if (!copied) throw new Error("Browser rejected the copy command.")
    const previousLabel = copyLinkButton.textContent
    copyLinkButton.textContent = "LINK COPIED"
    setMessage("Invite link copied. Your friend can claim any available player.")
    window.setTimeout(() => {
      copyLinkButton.textContent = previousLabel
    }, 1600)
  } catch {
    setMessage(`Copy failed. Share this link manually: ${window.location.href}`)
  }
})

window.addEventListener("keydown", (event) => {
  const key = eventKey(event)
  if (!allowedKeys.has(key)) return
  event.preventDefault()
  if (!event.repeat) setKeyHeld(key, "keyboard", true)
})

window.addEventListener("keyup", (event) => {
  const key = eventKey(event)
  if (!allowedKeys.has(key)) return
  event.preventDefault()
  setKeyHeld(key, "keyboard", false)
})

window.addEventListener("blur", releaseAllKeys)
window.addEventListener("beforeunload", () => disconnect(true))
window.setInterval(refreshRoom, 1000)
initialize()
