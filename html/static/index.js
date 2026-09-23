const cameraImage = document.querySelector('#camera-image');
const arrayCanvas = document.querySelector('#array-canvas');
const arrayContext = arrayCanvas.getContext('2d');
const histogramCanvas = document.querySelector('#histogram-canvas');
const histogramContext = histogramCanvas.getContext('2d');
const imageDownload = document.querySelector('#image-download');
const arrayDownload = document.querySelector('#array-download');
const status = document.querySelector('#status');
const statusText = document.querySelector('#status-text');
const cameraForm = document.querySelector('#camera-form');
const cameraStateText = document.querySelector('#camera-state');
const cameraMessage = document.querySelector('#camera-message');
const cameraFormat = document.querySelector('#camera-format');
const cameraHeight = document.querySelector('#camera-height');
const cameraHeightRange = document.querySelector('#camera-height-range');
const cameraWidth = document.querySelector('#camera-width');
const cameraWidthRange = document.querySelector('#camera-width-range');
const cameraExposure = document.querySelector('#camera-exposure');
const cameraExposureRange = document.querySelector('#camera-exposure-range');
const cameraFrameRate = document.querySelector('#camera-frame-rate');
const roiDisplay = document.querySelector('#roi-display');
const roiRow = document.querySelector('#roi-row');
const roiCol = document.querySelector('#roi-col');
const roiHeight = document.querySelector('#roi-height');
const roiWidth = document.querySelector('#roi-width');
const arrayHeight = document.querySelector('#array-height');
const arrayWidth = document.querySelector('#array-width');
const cameraActions = document.querySelector('.camera-actions');
const cameraToggle = document.querySelector('#camera-toggle');
const cameraToggleLabel = document.querySelector('#camera-toggle-label');
const cameraStop = document.querySelector('#camera-stop');
const cameraApplyButtons = [...document.querySelectorAll('.apply-button')];
const viewTabs = document.querySelector('.view-tabs');
const previewStage = document.querySelector('#preview-stage');
const cameraFields = [
  cameraFormat,
  cameraHeight,
  cameraHeightRange,
  cameraWidth,
  cameraWidthRange,
  cameraExposure,
  cameraExposureRange,
  cameraFrameRate,
  roiDisplay,
  roiRow,
  roiCol,
  roiHeight,
  roiWidth,
  arrayHeight,
  arrayWidth,
];
const cameraSliders = [
  { number: cameraHeight, range: cameraHeightRange, scale: 'linear' },
  { number: cameraWidth, range: cameraWidthRange, scale: 'linear' },
  { number: cameraExposure, range: cameraExposureRange, scale: 'log' },
];
let objectUrl = null;
let latestImageSource = null;
let latestArray = null;
let cameraStatus = null;
let cameraPending = false;
let cameraDirty = false;
let cameraRevision = 0;

function setConnectionState(connected, text = connected ? '数据已连接' : '等待数据') {
  status.dataset.state = connected ? 'connected' : 'disconnected';
  statusText.textContent = text;
}

function cameraErrorMessage(payload, fallback) {
  const detail = payload?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail.map((item) => item.msg || String(item)).join('; ');
  }
  return fallback;
}

async function cameraRequest(path, options = {}) {
  const response = await fetch(path, {
    cache: 'no-store',
    ...options,
  });
  const contentType = response.headers.get('Content-Type') || '';
  let payload = null;

  if (contentType.includes('application/json')) {
    payload = await response.json();
  } else {
    const text = await response.text();
    payload = text ? { detail: text } : null;
  }

  if (!response.ok) {
    throw new Error(cameraErrorMessage(payload, `请求失败 (${response.status})`));
  }
  return payload;
}

function setCameraMessage(text = '', error = false) {
  cameraMessage.textContent = text;
  cameraMessage.dataset.state = error ? 'error' : 'normal';
}

function syncRangeFromNumber(control) {
  const value = Number(control.number.value);
  if (!Number.isFinite(value)) return;

  const minimum = Number(control.number.min);
  const maximum = Number(control.number.max);
  const clamped = Math.max(minimum, Math.min(maximum, value));
  if (control.scale === 'log') {
    const ratio = Math.log(clamped / minimum) / Math.log(maximum / minimum);
    const rangeMinimum = Number(control.range.min);
    const rangeMaximum = Number(control.range.max);
    control.range.value = rangeMinimum + ratio * (rangeMaximum - rangeMinimum);
  } else {
    control.range.value = clamped;
  }
}

function syncNumberFromRange(control) {
  if (control.scale === 'log') {
    const rangeMinimum = Number(control.range.min);
    const rangeMaximum = Number(control.range.max);
    const ratio = (Number(control.range.value) - rangeMinimum) /
      (rangeMaximum - rangeMinimum);
    const minimum = Number(control.number.min);
    const maximum = Number(control.number.max);
    control.number.value = Math.round(minimum * (maximum / minimum) ** ratio);
  } else {
    control.number.value = control.range.value;
  }
}

for (const control of cameraSliders) {
  control.range.addEventListener('input', () => syncNumberFromRange(control));
  control.number.addEventListener('input', () => syncRangeFromNumber(control));
}

function normalizeFrameRate() {
  if (cameraFrameRate.value.trim() === '') return;
  const value = Number(cameraFrameRate.value);
  if (Number.isFinite(value)) cameraFrameRate.value = value.toFixed(2);
}

cameraFrameRate.addEventListener('change', normalizeFrameRate);

function validateProcessingGeometry() {
  for (const field of [roiRow, roiCol, roiHeight, roiWidth, arrayHeight, arrayWidth]) {
    field.setCustomValidity('');
  }

  const imageHeight = Number(cameraHeight.value);
  const imageWidth = Number(cameraWidth.value);
  const row = Number(roiRow.value);
  const col = Number(roiCol.value);
  const height = Number(roiHeight.value);
  const width = Number(roiWidth.value);
  const rows = Number(arrayHeight.value);
  const cols = Number(arrayWidth.value);

  if (![imageHeight, imageWidth, row, col, height, width, rows, cols].every(Number.isFinite)) {
    return;
  }

  roiRow.max = Math.max(0, imageHeight - 1);
  roiCol.max = Math.max(0, imageWidth - 1);
  roiHeight.max = Math.max(1, imageHeight - row);
  roiWidth.max = Math.max(1, imageWidth - col);
  arrayHeight.max = Math.max(1, height);
  arrayWidth.max = Math.max(1, width);

  if (row + height > imageHeight) {
    roiHeight.setCustomValidity('ROI 高度超出图像范围');
  }
  if (col + width > imageWidth) {
    roiWidth.setCustomValidity('ROI 宽度超出图像范围');
  }
  if (rows > height) {
    arrayHeight.setCustomValidity('阵列高度不能大于 ROI 高度');
  }
  if (cols > width) {
    arrayWidth.setCustomValidity('阵列宽度不能大于 ROI 宽度');
  }

}

function updateCameraControls() {
  const ready = cameraStatus !== null;
  for (const field of cameraFields) {
    field.disabled = !ready || cameraPending;
  }
  if (ready) {
    const toggleLabel = cameraPending
      ? '处理中…'
      : cameraStatus.running
        ? '暂停'
        : cameraStatus.opened
          ? '继续'
          : '启动';
    cameraToggle.dataset.action = cameraPending
      ? 'pending'
      : cameraStatus.running ? 'pause' : 'start';
    cameraToggleLabel.textContent = toggleLabel;
    cameraToggle.setAttribute('aria-label', toggleLabel);
    cameraToggle.title = toggleLabel;
  }
  cameraToggle.disabled = !ready || cameraPending;
  cameraActions.dataset.mode = ready && cameraStatus.opened ? 'session' : 'start';
  cameraStop.hidden = !ready || !cameraStatus.opened;
  cameraStop.disabled = !ready || cameraPending || !cameraStatus.opened;
  for (const button of cameraApplyButtons) {
    button.disabled = !ready || cameraPending || !cameraDirty;
  }
}

function renderCameraStatus(nextStatus, syncForm = true) {
  cameraStatus = nextStatus;

  if (nextStatus.running) {
    cameraStateText.dataset.state = 'running';
    cameraStateText.textContent = '运行中';
  } else if (nextStatus.opened) {
    cameraStateText.dataset.state = 'paused';
    cameraStateText.textContent = '已暂停';
  } else {
    cameraStateText.dataset.state = 'closed';
    cameraStateText.textContent = '已结束';
  }

  if (syncForm) {
    cameraFormat.value = nextStatus.format;
    [cameraHeight.value, cameraWidth.value] = nextStatus.size;
    cameraExposure.value = nextStatus.exposure;
    cameraFrameRate.value = Number(nextStatus.frame_rate).toFixed(2);
    roiDisplay.checked = nextStatus.roi_display;
    [roiRow.value, roiCol.value] = nextStatus.roi_origin;
    [roiHeight.value, roiWidth.value] = nextStatus.roi_size;
    [arrayHeight.value, arrayWidth.value] = nextStatus.array_size;
    for (const control of cameraSliders) syncRangeFromNumber(control);
    validateProcessingGeometry();
    cameraDirty = false;
  }
  updateCameraControls();
}

function renderCameraError(error) {
  cameraStateText.dataset.state = 'error';
  cameraStateText.textContent = '状态异常';
  setCameraMessage(error.message || String(error), true);
}

async function refreshCameraStatus(showError = false) {
  if (cameraPending) return;
  const revision = cameraRevision;

  try {
    const nextStatus = await cameraRequest('/api/camera');
    if (revision !== cameraRevision || cameraPending) return;
    renderCameraStatus(nextStatus, !cameraDirty);
  } catch (error) {
    if (revision !== cameraRevision || cameraPending) return;
    if (showError) renderCameraError(error);
  }
}

async function runCameraAction(path, options = {}, behavior = {}) {
  const {
    syncForm = true,
    successMessage = '操作成功',
  } = behavior;
  let successful = false;

  cameraRevision += 1;
  cameraPending = true;
  setCameraMessage('正在执行…');
  updateCameraControls();

  try {
    const nextStatus = await cameraRequest(path, options);
    renderCameraStatus(nextStatus, syncForm);
    setCameraMessage(successMessage);
    successful = true;
  } catch (error) {
    try {
      const nextStatus = await cameraRequest('/api/camera');
      renderCameraStatus(nextStatus, false);
      setCameraMessage(error.message || String(error), true);
    } catch {
      renderCameraError(error);
    }
  } finally {
    cameraPending = false;
    updateCameraControls();
  }
  return successful;
}

cameraForm.addEventListener('input', (event) => {
  if (event.target === roiDisplay) return;
  validateProcessingGeometry();
  cameraDirty = true;
  setCameraMessage('参数尚未应用');
  updateCameraControls();
});

roiDisplay.addEventListener('change', async () => {
  const hadPendingChanges = cameraDirty;
  const successful = await runCameraAction('/api/camera/config', {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ roi_display: roiDisplay.checked }),
  }, {
    syncForm: false,
    successMessage: hadPendingChanges
      ? 'ROI 显示已更新；其他参数尚未应用'
      : 'ROI 显示已更新',
  });

  if (!successful && cameraStatus) {
    roiDisplay.checked = cameraStatus.roi_display;
  }
});

cameraForm.addEventListener('submit', (event) => {
  event.preventDefault();
  normalizeFrameRate();
  validateProcessingGeometry();
  if (!cameraForm.reportValidity()) return;

  runCameraAction('/api/camera/config', {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      format: cameraFormat.value,
      size: [Number(cameraHeight.value), Number(cameraWidth.value)],
      exposure: Number(cameraExposure.value),
      frame_rate: Number(cameraFrameRate.value),
      roi_display: roiDisplay.checked,
      roi_origin: [Number(roiRow.value), Number(roiCol.value)],
      roi_size: [Number(roiHeight.value), Number(roiWidth.value)],
      array_size: [Number(arrayHeight.value), Number(arrayWidth.value)],
    }),
  });
});

cameraToggle.addEventListener('click', () => {
  if (!cameraStatus) return;
  const action = cameraStatus.running ? 'pause' : 'start';
  runCameraAction(`/api/camera/${action}`, { method: 'POST' });
});
cameraStop.addEventListener('click', () => {
  runCameraAction('/api/camera/stop', { method: 'POST' });
});

function setImage(source) {
  if (!source) return;
  latestImageSource = source;
  imageDownload.disabled = false;

  if (objectUrl) {
    URL.revokeObjectURL(objectUrl);
    objectUrl = null;
  }

  if (source instanceof Blob) {
    objectUrl = URL.createObjectURL(source);
    cameraImage.src = objectUrl;
  } else {
    cameraImage.src = source;
  }
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function downloadTimestamp() {
  return new Date().toISOString().replace(/[:.]/g, '-');
}

imageDownload.addEventListener('click', async (event) => {
  event.stopPropagation();
  if (!latestImageSource) return;

  try {
    const blob = latestImageSource instanceof Blob
      ? latestImageSource
      : await fetch(latestImageSource).then((response) => response.blob());
    downloadBlob(blob, `ccd-image-${downloadTimestamp()}.jpg`);
  } catch (error) {
    console.error('Image download failed', error);
  }
});

arrayDownload.addEventListener('click', (event) => {
  event.stopPropagation();
  if (!latestArray) return;

  const lines = [];
  for (let row = 0; row < latestArray.rows; row += 1) {
    const begin = row * latestArray.columns;
    lines.push(latestArray.values.slice(begin, begin + latestArray.columns).join(','));
  }
  const blob = new Blob([`\uFEFF${lines.join('\r\n')}`], {
    type: 'text/csv;charset=utf-8',
  });
  downloadBlob(blob, `ccd-array-${downloadTimestamp()}.csv`);
});

function setActiveView(view) {
  previewStage.dataset.view = view;
  for (const tab of viewTabs.querySelectorAll('[data-view]')) {
    tab.setAttribute('aria-selected', String(tab.dataset.view === view));
  }
  window.requestAnimationFrame(drawHistogram);
}

viewTabs.addEventListener('click', (event) => {
  const tab = event.target.closest('[data-view]');
  if (tab) setActiveView(tab.dataset.view);
});

viewTabs.addEventListener('keydown', (event) => {
  if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
  const tabs = [...viewTabs.querySelectorAll('[data-view]')];
  const current = tabs.indexOf(document.activeElement);
  if (current < 0) return;
  event.preventDefault();
  const step = event.key === 'ArrowRight' ? 1 : -1;
  const next = tabs[(current + step + tabs.length) % tabs.length];
  next.focus();
  setActiveView(next.dataset.view);
});

const parulaColors = [
  [53, 42, 135],
  [15, 92, 164],
  [18, 125, 165],
  [7, 156, 154],
  [21, 177, 126],
  [89, 189, 88],
  [165, 190, 51],
  [225, 185, 39],
  [249, 251, 14],
];

function heatColor(value) {
  const normalized = Math.max(0, Math.min(1, Number(value) / 255));
  const position = normalized * (parulaColors.length - 1);
  const index = Math.min(parulaColors.length - 2, Math.floor(position));
  const ratio = position - index;
  const start = parulaColors[index];
  const end = parulaColors[index + 1];
  const color = start.map((channel, channelIndex) =>
    Math.round(channel + (end[channelIndex] - channel) * ratio)
  );
  return `rgb(${color[0]} ${color[1]} ${color[2]})`;
}

const histogramBinCount = 64;
let histogramCounts = new Uint32Array(histogramBinCount);

function drawHistogram() {
  const bounds = histogramCanvas.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return;

  const pixelRatio = window.devicePixelRatio || 1;
  const pixelWidth = Math.max(1, Math.round(bounds.width * pixelRatio));
  const pixelHeight = Math.max(1, Math.round(bounds.height * pixelRatio));
  if (histogramCanvas.width !== pixelWidth) histogramCanvas.width = pixelWidth;
  if (histogramCanvas.height !== pixelHeight) histogramCanvas.height = pixelHeight;

  const width = bounds.width;
  const height = bounds.height;
  const compact = width < 260;
  const plot = compact
    ? { left: 10, right: 5, top: 7, bottom: 13 }
    : { left: 48, right: 16, top: 20, bottom: 34 };
  const plotWidth = Math.max(1, width - plot.left - plot.right);
  const plotHeight = Math.max(1, height - plot.top - plot.bottom);
  const maxCount = Math.max(1, ...histogramCounts);

  histogramContext.setTransform(1, 0, 0, 1, 0, 0);
  histogramContext.clearRect(0, 0, pixelWidth, pixelHeight);
  histogramContext.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);

  histogramContext.strokeStyle = 'rgb(255 255 255 / 7%)';
  histogramContext.lineWidth = 1;
  for (let line = 0; line <= 4; line += 1) {
    const y = plot.top + plotHeight * line / 4 + 0.5;
    histogramContext.beginPath();
    histogramContext.moveTo(plot.left, y);
    histogramContext.lineTo(width - plot.right, y);
    histogramContext.stroke();
  }

  const gap = compact ? 1 : 2;
  const barWidth = plotWidth / histogramBinCount;
  histogramContext.fillStyle = '#029dce';
  histogramCounts.forEach((count, index) => {
    const barHeight = count / maxCount * plotHeight;
    histogramContext.fillRect(
      plot.left + index * barWidth + gap / 2,
      plot.top + plotHeight - barHeight,
      Math.max(1, barWidth - gap),
      barHeight,
    );
  });

  if (!compact) {
    histogramContext.fillStyle = '#8294a2';
    histogramContext.font = '12px system-ui, sans-serif';
    histogramContext.textBaseline = 'top';
    histogramContext.textAlign = 'left';
    histogramContext.fillText('0', plot.left, height - 22);
    histogramContext.textAlign = 'center';
    histogramContext.fillText('128', plot.left + plotWidth / 2, height - 22);
    histogramContext.textAlign = 'right';
    histogramContext.fillText('255', width - plot.right, height - 22);
    histogramContext.textBaseline = 'middle';
    histogramContext.fillText(String(maxCount), plot.left - 8, plot.top);
  }
}

function setHistogram(values) {
  histogramCounts = new Uint32Array(histogramBinCount);
  for (const value of values) {
    const normalized = Math.max(0, Math.min(255, Number(value)));
    const index = Math.min(
      histogramBinCount - 1,
      Math.floor(normalized / 256 * histogramBinCount),
    );
    histogramCounts[index] += 1;
  }

  document.querySelector('#histogram-meta').textContent =
    `${histogramBinCount} bins · ${values.length} samples`;
  drawHistogram();
}

new ResizeObserver(() => {
  window.requestAnimationFrame(drawHistogram);
}).observe(histogramCanvas);

function setArray(values, height = null, width = null) {
  if (!values || values.length === 0) return;

  const nested = Array.isArray(values) && Array.isArray(values[0]);
  const rows = nested ? values : null;
  const flat = nested ? rows.flat().map(Number) : values;
  height ??= nested ? rows.length : 1;
  width ??= nested ? rows[0].length : flat.length;
  latestArray = {
    values: Array.from(flat, Number),
    rows: height,
    columns: width,
  };
  arrayDownload.disabled = false;

  arrayCanvas.width = width;
  arrayCanvas.height = height;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      arrayContext.fillStyle = heatColor(Number(flat[y * width + x]));
      arrayContext.fillRect(x, y, 1, 1);
    }
  }

  const min = Math.min(...flat);
  const max = Math.max(...flat);
  const mean = flat.reduce((sum, value) => sum + value, 0) / flat.length;

  document.querySelector('#array-meta').textContent = `${height} × ${width}`;
  document.querySelector('#array-min').textContent = min.toFixed(2);
  document.querySelector('#array-mean').textContent = mean.toFixed(2);
  document.querySelector('#array-max').textContent = max.toFixed(2);
  setHistogram(flat);
}

function updateView({
  image,
  array,
  arrayRows = null,
  arrayColumns = null,
  frameId = null,
  timestamp = null,
}) {
  setImage(image);
  setArray(array, arrayRows, arrayColumns);
  setConnectionState(true);

  const details = [];
  if (frameId !== null) details.push(`#${frameId}`);
  if (timestamp !== null) details.push(new Date(timestamp).toLocaleTimeString());
  document.querySelector('#image-meta').textContent = details.join(' · ') || 'LIVE';
}

// Integration point for tests and alternate transports.
window.ccdView = { update: updateView, setConnectionState };

function decodePacket(buffer) {
  const headerSize = 20;
  if (buffer.byteLength < headerSize) throw new Error('packet is too short');

  const view = new DataView(buffer);
  const frameId = view.getUint32(0, true);
  const timestamp = view.getFloat64(4, true);
  const rows = view.getUint16(12, true);
  const columns = view.getUint16(14, true);
  const jpegSize = view.getUint32(16, true);
  const arraySize = rows * columns * Float32Array.BYTES_PER_ELEMENT;
  const jpegOffset = headerSize + arraySize;

  if (jpegOffset + jpegSize !== buffer.byteLength) {
    throw new Error('packet size mismatch');
  }

  const array = new Float32Array(buffer.slice(headerSize, jpegOffset));
  const image = new Blob(
    [buffer.slice(jpegOffset, jpegOffset + jpegSize)],
    { type: 'image/jpeg' },
  );

  return {
    image,
    array,
    arrayRows: rows,
    arrayColumns: columns,
    frameId,
    timestamp,
  };
}

function connect() {
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  const socket = new WebSocket(`${scheme}://${location.host}/ws`);
  socket.binaryType = 'arraybuffer';

  socket.addEventListener('open', () => {
    setConnectionState(true, '等待图像');
  });

  socket.addEventListener('message', (event) => {
    try {
      updateView(decodePacket(event.data));
    } catch (error) {
      console.error(error);
      setConnectionState(false, '数据格式错误');
    }
  });

  socket.addEventListener('close', () => {
    setConnectionState(false, '连接断开，正在重试');
    window.setTimeout(connect, 1500);
  });

  socket.addEventListener('error', () => socket.close());
}

function showDemo() {
  const cameraHeight = 3072;
  const cameraWidth = 4096;
  const roiRow = 768;
  const roiColumn = 1024;
  const roiHeight = 1536;
  const roiWidth = 2048;
  const blockSize = 32;
  const grayLevels = [0, 85, 170, 255];
  const maxOffset = 4;
  const noiseAmplitude = 3;
  const fadeWidth = 384;
  const darkLevel = 3;
  const imageBlockHeight = cameraHeight / blockSize;
  const imageBlockWidth = cameraWidth / blockSize;
  const arrayHeight = roiHeight / blockSize;
  const arrayWidth = roiWidth / blockSize;
  const offsetRow = Math.floor(Math.random() * (2 * maxOffset + 1)) - maxOffset;
  const offsetColumn = Math.floor(Math.random() * (2 * maxOffset + 1)) - maxOffset;
  const imageValues = Array.from({ length: imageBlockHeight }, () =>
    Array.from({ length: imageBlockWidth }, () =>
      grayLevels[Math.floor(Math.random() * grayLevels.length)]
    )
  );

  const demoCanvas = document.createElement('canvas');
  demoCanvas.height = cameraHeight;
  demoCanvas.width = cameraWidth;
  const context = demoCanvas.getContext('2d');
  context.fillStyle = `rgb(${darkLevel} ${darkLevel} ${darkLevel})`;
  context.fillRect(0, 0, cameraWidth, cameraHeight);

  const regionRow = roiRow - fadeWidth;
  const regionColumn = roiColumn - fadeWidth;
  const regionHeight = roiHeight + 2 * fadeWidth;
  const regionWidth = roiWidth + 2 * fadeWidth;
  const imageData = context.createImageData(regionWidth, regionHeight);
  const sums = new Float64Array(arrayHeight * arrayWidth);

  function edgeFade(position, begin, end) {
    if (position < begin) {
      const x = (position - begin + fadeWidth) / fadeWidth;
      return x * x * (3 - 2 * x);
    }
    if (position >= end) {
      const x = (end + fadeWidth - position) / fadeWidth;
      return x * x * (3 - 2 * x);
    }
    return 1;
  }

  let pixel = 0;
  for (let localRow = 0; localRow < regionHeight; localRow += 1) {
    const imageRow = regionRow + localRow;
    const fadeY = edgeFade(imageRow, roiRow, roiRow + roiHeight);
    const blockRow = Math.floor((imageRow - offsetRow) / blockSize);

    for (let localColumn = 0; localColumn < regionWidth; localColumn += 1) {
      const imageColumn = regionColumn + localColumn;
      const fadeX = edgeFade(imageColumn, roiColumn, roiColumn + roiWidth);
      const blockColumn = Math.floor((imageColumn - offsetColumn) / blockSize);
      const base = imageValues[blockRow][blockColumn];
      const noise = (2 * Math.random() - 1) * noiseAmplitude;
      const fade = fadeY * fadeX;
      const value = Math.max(
        0,
        Math.min(255, Math.round(darkLevel + (base + noise - darkLevel) * fade)),
      );

      imageData.data[pixel] = value;
      imageData.data[pixel + 1] = value;
      imageData.data[pixel + 2] = value;
      imageData.data[pixel + 3] = 255;
      pixel += 4;

      if (
        imageRow >= roiRow && imageRow < roiRow + roiHeight &&
        imageColumn >= roiColumn && imageColumn < roiColumn + roiWidth
      ) {
        const arrayRow = Math.floor((imageRow - roiRow) / blockSize);
        const arrayColumn = Math.floor((imageColumn - roiColumn) / blockSize);
        sums[arrayRow * arrayWidth + arrayColumn] += value;
      }
    }
  }
  context.putImageData(imageData, regionColumn, regionRow);

  const pixelsPerCell = blockSize * blockSize;
  const values = Array.from({ length: arrayHeight }, (_, row) =>
    Array.from({ length: arrayWidth }, (_, column) =>
      sums[row * arrayWidth + column] / pixelsPerCell
    )
  );

  updateView({
    image: demoCanvas.toDataURL('image/jpeg', 0.9),
    array: values,
    arrayRows: arrayHeight,
    arrayColumns: arrayWidth,
    frameId: 0,
    timestamp: Date.now(),
  });
  setConnectionState(false, '演示数据');
}

if (location.protocol === 'file:') {
  showDemo();
  cameraStateText.dataset.state = 'closed';
  cameraStateText.textContent = '仅服务器可用';
  setCameraMessage('请通过服务地址打开此页面');
} else {
  connect();
  refreshCameraStatus(true);
  window.setInterval(refreshCameraStatus, 3000);
}
