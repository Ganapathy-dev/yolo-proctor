/**
 * Get WebGL information
 */
export function getWebGLInfo() {
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

/**
 * Gather detailed system information
 */
export async function gatherSystemInfo(video?: HTMLVideoElement): Promise<Record<string, any>> {
  const info: Record<string, any> = {};

  const ua = navigator.userAgent;
  const browserMatch = ua.match(/(Chrome|Firefox|Safari|Edge|Opera)\/(\d+)/);

  info['Browser'] = browserMatch ? `${browserMatch[1]} ${browserMatch[2]}` : ua.split(' ')[0];
  info['User Agent'] = ua;
  info['Platform'] = navigator.platform;
  info['Language'] = navigator.language;
  info['Cookie Enabled'] = navigator.cookieEnabled ? 'Yes' : 'No';
  info['CPU Cores'] = navigator.hardwareConcurrency || 'Unknown';
  info['Device Memory'] = (navigator as any).deviceMemory ? `${(navigator as any).deviceMemory} GB` : 'Unknown';
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

  if (video && video.videoWidth > 0) {
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

/**
 * Render system info in a container element
 */
export function renderSystemInfo(info: Record<string, any>, container: HTMLElement) {
  container.innerHTML = '';

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

  container.appendChild(table);
}

/**
 * Initialize camera and set video stream
 */
export async function initCamera(video: HTMLVideoElement) {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 640 } });
  video.srcObject = stream;
  await new Promise(resolve => video.onloadedmetadata = resolve);
}

/**
 * Preprocess video frame to Float32Array [1,3,640,640]
 */
export function preprocess(video: HTMLVideoElement): Float32Array {
  const canvas = document.createElement('canvas');
  canvas.width = 640;
  canvas.height = 640;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(video, 0, 0, 640, 640);

  const imageData = ctx.getImageData(0, 0, 640, 640);
  const data = imageData.data;
  const floatData = new Float32Array(3 * 640 * 640);

  for (let i = 0; i < 640 * 640; i++) {
    floatData[i] = data[i * 4] / 255; // R
    floatData[i + 640 * 640] = data[i * 4 + 1] / 255; // G
    floatData[i + 2 * 640 * 640] = data[i * 4 + 2] / 255; // B
  }

  return floatData;
}
