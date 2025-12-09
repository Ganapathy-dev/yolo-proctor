import * as tf from '@tensorflow/tfjs'; // <-- important!
import * as cocoSsd from '@tensorflow-models/coco-ssd';

const video = document.getElementById('webcam') as HTMLVideoElement;
const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const ctx = canvas.getContext('2d')!;
const infoList = document.getElementById('info-list')!;

async function initCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;

  await new Promise(resolve => video.onloadedmetadata = resolve);

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  const wrapper = document.getElementById("video-wrapper")!;
  wrapper.style.width = video.videoWidth + "px";
  wrapper.style.height = video.videoHeight + "px";
}

// Load model
const loadModel = async () => {
  const model = await cocoSsd.load();
  return model;
};

// Run detection
const runDetection = async () => {
  const model = await loadModel();

  video.addEventListener('play', () => {
    const detectFrame = async () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const predictions = await model.detect(video);

      predictions.forEach(pred => {
        const [x, y, w, h] = pred.bbox;

        ctx.strokeStyle = 'lime';
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);

        ctx.font = "16px Arial";
        ctx.fillStyle = 'lime';
        ctx.fillText(`${pred.class} (${pred.score.toFixed(2)})`, x, y - 5);
      });

      requestAnimationFrame(detectFrame);
    };

    detectFrame();
  });
};

// system information
function getWebGLInfo() {
  const tempCanvas = document.createElement('canvas');
  const gl = (tempCanvas.getContext('webgl') ||
              tempCanvas.getContext('experimental-webgl')) as WebGLRenderingContext | null;

  if (!gl) return { supported: false };

  const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
  return {
    supported: true,
    renderer: debugInfo ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : 'Unknown',
    vendor: debugInfo ? gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) : 'Unknown',
    version: gl.getParameter(gl.VERSION),
    shadingLanguage: gl.getParameter(gl.SHADING_LANGUAGE_VERSION),
  };
}

async function gatherSystemInfo(): Promise<Record<string, any>> {
  const info: Record<string, any> = {};

  const ua = navigator.userAgent;
  const browserMatch = ua.match(/(Chrome|Firefox|Safari|Edge|Opera)\/(\d+)/);

  info['Browser'] = browserMatch ? `${browserMatch[1]} ${browserMatch[2]}` : ua.split(' ')[0];
  info['User Agent'] = ua;
  info['Platform'] = navigator.platform;
  info['Language'] = navigator.language;
  info['Cookie Enabled'] = navigator.cookieEnabled ? 'Yes' : 'No';
  info['CPU Cores'] = navigator.hardwareConcurrency || null;
  info['Device Memory'] = (navigator as any).deviceMemory ? `${(navigator as any).deviceMemory} GB` : null;
  info['Screen Resolution'] = `${screen.width}x${screen.height}`;
  info['Screen Color Depth'] = `${screen.colorDepth} bits`;
  info['Window Size'] = `${window.innerWidth}x${window.innerHeight}`;
  info['Device Pixel Ratio'] = window.devicePixelRatio || 1;

  const webglInfo = getWebGLInfo();
  info['WebGL Support'] = webglInfo.supported ? 'Yes' : 'No';
  if (webglInfo.supported) {
    info['WebGL Renderer'] = webglInfo.renderer;
    info['WebGL Vendor'] = webglInfo.vendor;
    info['WebGL Version'] = webglInfo.version;
    info['WebGL Shading Language'] = webglInfo.shadingLanguage;
  }

  info['GPU Delegate'] = 'Not initialized';

  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(d => d.kind === 'videoinput');
    info['Video Input Devices'] = videoDevices.length;
    info['Active Video Device'] = videoDevices[0]?.label || 'Unknown Camera';
  } catch {
    info['Video Input Devices'] = 'Access denied';
  }

  if (video.videoWidth > 0) {
    info['Video Resolution'] = `${video.videoWidth}x${video.videoHeight}`;
    info['Video Aspect Ratio'] = (video.videoWidth / video.videoHeight).toFixed(2);
  } else {
    info['Video Resolution'] = 'Not available yet';
  }

  if ('memory' in performance) {
    const memory = (performance as any).memory;
    info['JS Heap Used'] = `${(memory.usedJSHeapSize / 1048576).toFixed(2)} MB`;
    info['JS Heap Total'] = `${(memory.totalJSHeapSize / 1048576).toFixed(2)} MB`;
    info['JS Heap Limit'] = `${(memory.jsHeapSizeLimit / 1048576).toFixed(2)} MB`;
  }

  if ('connection' in navigator) {
    const conn = (navigator as any).connection;
    info['Connection Type'] = conn?.effectiveType || 'Unknown';
    info['Connection Downlink'] = conn?.downlink ? `${conn.downlink} Mbps` : 'Unknown';
  }

  return info;
}

function renderSystemInfo(info: Record<string, any>) {
  infoList.innerHTML = '';

  const table = document.createElement('table');
  table.className = 'system-info-table';

  for (const key in info) {
    const row = document.createElement('tr');

    const labelCell = document.createElement('td');
    labelCell.textContent = key;
    labelCell.className = 'system-info-label';

    const valueCell = document.createElement('td');
    valueCell.textContent = info[key];
    valueCell.className = 'system-info-value';

    row.appendChild(labelCell);
    row.appendChild(valueCell);
    table.appendChild(row);
  }

  infoList.appendChild(table);
}

// initialize page
const sysInfo = await gatherSystemInfo();
initCamera();
runDetection();
renderSystemInfo(sysInfo);