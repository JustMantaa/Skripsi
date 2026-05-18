/**
 * Camera and Hand Detection Module
 * Handles webcam capture, frame sending to API, and result display
 */

const CameraApp = {
    // Configuration
    config: {
        pythonApiBaseUrl: 'http://127.0.0.1:5000/predict',
        // pythonApiBaseUrl: '/flask-api', //server api
        sendIntervalMs: 66,
        sendJpegQuality: 0.8,
        processWidth: 640,
        processHeight: 360,
    },

    // DOM Elements
    elements: {},

    // State
    state: {
        currentStream: null,
        cameraActive: false,
        sendTimeoutId: null,
        isSendingFrame: false,
    },

    // Initialize the app
    init() {
        this.cacheElements();
        this.setupEventListeners();
    },

    // Cache DOM elements
    cacheElements() {
        this.elements = {
            video: document.getElementById('video'),
            toggleBtn: document.getElementById('toggle-camera'),
            cameraWrapper: document.getElementById('camera-wrapper'),
            hasilDeteksi: document.getElementById('hasil-deteksi'),
            hasilConfidence: document.getElementById('hasil-confidence'),
            hasilMode: document.getElementById('hasil-mode'),
            canvas: document.createElement('canvas'),
        };
    },

    // Setup event listeners
    setupEventListeners() {
        this.elements.toggleBtn.addEventListener('click', () => this.toggleCamera());
    },

    // Stop sending frames
    stopSendingFrames() {
        if (this.state.sendTimeoutId) {
            clearTimeout(this.state.sendTimeoutId);
            this.state.sendTimeoutId = null;
        }
        this.state.isSendingFrame = false;
    },

    // Send frame to API
    async sendFrameToApi() {
        if (!this.state.cameraActive || this.state.isSendingFrame) {
            return;
        }

        if (!this.elements.video.videoWidth || !this.elements.video.videoHeight) {
            this.state.sendTimeoutId = setTimeout(() => this.sendFrameToApi(), this.config.sendIntervalMs);
            return;
        }

        this.state.isSendingFrame = true;

        try {
            const { canvas } = this.elements;
            const { processWidth, processHeight } = this.config;

            canvas.width = processWidth;
            canvas.height = processHeight;
            const ctx = canvas.getContext('2d');
            ctx.save();
            ctx.scale(-1, 1);
            ctx.drawImage(this.elements.video, -canvas.width, 0, canvas.width, canvas.height);
            ctx.restore();

            const dataUrl = canvas.toDataURL('image/jpeg', this.config.sendJpegQuality);

            const response = await fetch(this.config.pythonApiBaseUrl, {
            //const response = await fetch(`${this.config.pythonApiBaseUrl}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: dataUrl }),
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();
            this.updateUI(data);
        } catch (error) {
            this.updateUI({
                label: 'Belum terhubung',
                confidence: '-',
                mode: 'API offline',
            });
        } finally {
            this.state.isSendingFrame = false;
            if (this.state.cameraActive) {
                this.state.sendTimeoutId = setTimeout(() => this.sendFrameToApi(), this.config.sendIntervalMs);
            }
        }
    },

    // Update UI with API response
    updateUI(data) {
        this.elements.hasilDeteksi.textContent = data.label || '-';
        this.elements.hasilConfidence.textContent = Number.isFinite(data.confidence)
            ? Number(data.confidence).toFixed(4)
            : '-';
        this.elements.hasilMode.textContent = data.mode || '-';
    },

    // Start sending frames
    startSendingFrames() {
        this.stopSendingFrames();
        this.sendFrameToApi();
    },

    // Toggle camera on/off
    toggleCamera() {
        if (!this.state.cameraActive) {
            this.startCamera();
        } else {
            this.stopCamera();
        }
    },

    // Start camera
    startCamera() {
        navigator.mediaDevices
            .getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                },
            })
            .then((stream) => {
                this.elements.video.srcObject = stream;

                // Disable native controls
                try {
                    this.elements.video.disablePictureInPicture = true;
                    this.elements.video.removeAttribute('controls');
                    this.elements.video.controls = false;
                } catch (e) {
                    // Ignore if not supported
                }

                // Prevent right-click context menu
                this.elements.video.addEventListener('contextmenu', (ev) => ev.preventDefault());

                this.state.currentStream = stream;
                this.state.cameraActive = true;
                this.elements.cameraWrapper.classList.add('is-camera-active');
                this.elements.toggleBtn.textContent = 'Nonaktifkan Kamera';
                this.elements.toggleBtn.classList.remove('btn-primary');
                this.elements.toggleBtn.classList.add('btn-danger');

                // Set video size
                const track = stream.getVideoTracks()[0];
                const settings = track.getSettings();
                this.elements.video.width = settings.width || 1280;
                this.elements.video.height = settings.height || 720;

                this.elements.hasilDeteksi.textContent = 'Mendeteksi...';
                this.elements.hasilConfidence.textContent = '-';
                this.elements.hasilMode.textContent = 'Menghubungkan API...';

                this.startSendingFrames();
            })
            .catch((err) => {
                alert(`Tidak dapat mengakses kamera: ${err}`);
            });
    },

    // Stop camera
    stopCamera() {
        if (this.state.currentStream) {
            this.state.currentStream.getTracks().forEach((track) => track.stop());
            this.elements.video.srcObject = null;
            this.state.currentStream = null;
        }

        this.state.cameraActive = false;
        this.stopSendingFrames();
        this.elements.cameraWrapper.classList.remove('is-camera-active');
        this.elements.toggleBtn.textContent = 'Aktifkan Kamera';
        this.elements.toggleBtn.classList.remove('btn-danger');
        this.elements.toggleBtn.classList.add('btn-primary');

        this.elements.hasilDeteksi.textContent = '-';
        this.elements.hasilConfidence.textContent = '-';
        this.elements.hasilMode.textContent = '-';
    },
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    CameraApp.init();
});