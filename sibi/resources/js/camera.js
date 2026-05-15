/**
 * Camera and Hand Detection Module
 */

const CameraApp = {
    config: {
        pythonApiBaseUrl: '/flask-api',
        sendIntervalMs: 66,
        sendJpegQuality: 0.6,
        processWidth: 320,
        processHeight: 240,
        yoloFailLimit: 3,
    },

    elements: {},

    state: {
        currentStream: null,
        cameraActive: false,
        sendTimeoutId: null,
        isSendingFrame: false,

        currentMode: 'LSTM',
        yoloFailCount: 0,
    },

    init() {
        this.cacheElements();
        this.setupEventListeners();
        this.saveScrollState();
    },

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

    saveScrollState() {
        this.originalBodyOverflow = document.body.style.overflow;
        this.originalTouchAction = document.body.style.touchAction;
    },

    setupEventListeners() {
        this.elements.toggleBtn.addEventListener('click', () => this.toggleCamera());
    },

    setScrollLock(locked) {
        document.body.style.overflow = locked ? 'hidden' : (this.originalBodyOverflow || '');
        document.body.style.touchAction = locked ? 'none' : (this.originalTouchAction || '');
    },

    stopSendingFrames() {
        if (this.state.sendTimeoutId) {
            clearTimeout(this.state.sendTimeoutId);
            this.state.sendTimeoutId = null;
        }
        this.state.isSendingFrame = false;
    },

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

            const endpoint = this.state.currentMode === 'LSTM'
                ? '/predict-lstm'
                : '/predict-yolo';

            const response = await fetch(`${this.config.pythonApiBaseUrl}${endpoint}`, {
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
            this.handleSwitching(data);

        } catch (error) {
            console.error('API error:', error);
            this.updateUI({
                label: 'Belum terhubung',
                confidence: '-',
                mode: 'API offline',
            });
        } finally {
            this.state.isSendingFrame = false;

            if (this.state.cameraActive) {
                this.state.sendTimeoutId = setTimeout(
                    () => this.sendFrameToApi(),
                    this.config.sendIntervalMs
                );
            }
        }
    },

    handleSwitching(data) {
        if (this.state.currentMode === 'LSTM') {
            if (data.status === 'rejected') {
                this.state.currentMode = 'YOLO';
                this.state.yoloFailCount = 0;
                this.elements.hasilMode.textContent = 'YOLO';
            }
            return;
        }

        if (this.state.currentMode === 'YOLO') {
            if (data.status === 'not_detected') {
                this.state.yoloFailCount++;

                if (this.state.yoloFailCount >= this.config.yoloFailLimit) {
                    this.state.currentMode = 'LSTM';
                    this.state.yoloFailCount = 0;
                    this.elements.hasilMode.textContent = 'LSTM';
                }
            } else if (data.status === 'detected') {
                this.state.yoloFailCount = 0;
            }
        }
    },

    updateUI(data) {
        this.elements.hasilDeteksi.textContent = data.label || '-';
        this.elements.hasilConfidence.textContent = Number.isFinite(data.confidence)
            ? Number(data.confidence).toFixed(4)
            : '-';
        this.elements.hasilMode.textContent = data.mode || this.state.currentMode || '-';
    },

    startSendingFrames() {
        this.stopSendingFrames();
        this.sendFrameToApi();
    },

    toggleCamera() {
        if (!this.state.cameraActive) {
            this.startCamera();
        } else {
            this.stopCamera();
        }
    },

    startCamera() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert('Browser tidak mendukung akses kamera. Gunakan HTTPS.');
            return;
        }

        navigator.mediaDevices
            .getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                },
            })
            .then((stream) => {
                this.elements.video.srcObject = stream;

                try {
                    this.elements.video.disablePictureInPicture = true;
                    this.elements.video.removeAttribute('controls');
                    this.elements.video.controls = false;
                } catch (e) {}

                this.elements.video.addEventListener('contextmenu', (ev) => ev.preventDefault());

                this.state.currentStream = stream;
                this.state.cameraActive = true;
                this.state.currentMode = 'LSTM';
                this.state.yoloFailCount = 0;

                this.elements.cameraWrapper.classList.add('is-camera-active');
                this.elements.toggleBtn.textContent = 'Nonaktifkan Kamera';
                this.elements.toggleBtn.classList.remove('btn-primary');
                this.elements.toggleBtn.classList.add('btn-danger');
                this.setScrollLock(true);

                const track = stream.getVideoTracks()[0];
                const settings = track.getSettings();
                this.elements.video.width = settings.width || 1280;
                this.elements.video.height = settings.height || 720;

                this.elements.hasilDeteksi.textContent = 'Mendeteksi...';
                this.elements.hasilConfidence.textContent = '-';
                this.elements.hasilMode.textContent = 'LSTM';

                this.startSendingFrames();
            })
            .catch((err) => {
                alert(`Tidak dapat mengakses kamera: ${err}`);
            });
    },

    stopCamera() {
        if (this.state.currentStream) {
            this.state.currentStream.getTracks().forEach((track) => track.stop());
            this.elements.video.srcObject = null;
            this.state.currentStream = null;
        }

        this.state.cameraActive = false;
        this.state.currentMode = 'LSTM';
        this.state.yoloFailCount = 0;

        this.stopSendingFrames();

        this.elements.cameraWrapper.classList.remove('is-camera-active');
        this.elements.toggleBtn.textContent = 'Aktifkan Kamera';
        this.elements.toggleBtn.classList.remove('btn-danger');
        this.elements.toggleBtn.classList.add('btn-primary');
        this.setScrollLock(false);

        this.elements.hasilDeteksi.textContent = '-';
        this.elements.hasilConfidence.textContent = '-';
        this.elements.hasilMode.textContent = '-';
    },
};

document.addEventListener('DOMContentLoaded', () => {
    CameraApp.init();
});