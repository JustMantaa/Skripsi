@extends('layouts.app')

@section('content')
<style>
    .camera-wrapper {
        position: relative;
        overflow: hidden;
    }

    .camera-status {
        position: absolute;
        top: 10px;
        left: 10px;
        z-index: 2;
        background: rgba(255, 255, 255, 0.8);
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: bold;
    }

    .camera-status small {
        display: block;
        font-weight: 500;
        color: #555;
    }

    .camera-placeholder {
        position: absolute;
        inset: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 1;
        pointer-events: none;
    }

    .camera-placeholder-icon {
        font-size: 120px;
        color: rgba(46, 45, 45, 0.95);
        line-height: 1;
    }

    .is-camera-active .camera-placeholder {
        display: none;
    }

    #video {
        width: 100%;
        height: 720px;
        display: block;
        background: #f3f3f3;
        object-fit: cover;
        transform: scaleX(-1);
    }
</style>
<div class="container mt-5">
    </div>
    <div class="row mb-4">
        <div class="col-lg-8 mx-auto p-2">
            <div class="card camera-wrapper" id="camera-wrapper">
                <div class="card-body p-0" style="position: relative;">
                    <div class="camera-status">
                        Hasil Deteksi: <span id="hasil-deteksi">-</span>
                        <small>Confidence: <span id="hasil-confidence">-</span></small>
                        <small>Mode: <span id="hasil-mode">-</span></small>
                    </div>
                    <div id="camera-placeholder" class="camera-placeholder" aria-hidden="true">
                        <i class="fa-solid fa-video-slash camera-placeholder-icon"></i>
                    </div>
                    <video id="video" autoplay playsinline></video>
                    <div class="d-flex justify-content-center p-3 gap-2">
                        <button id="toggle-camera" class="btn btn-primary">Aktifkan Kamera</button>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
<script>
    const video = document.getElementById('video');
    const toggleBtn = document.getElementById('toggle-camera');
    const cameraWrapper = document.getElementById('camera-wrapper');
    const hasilDeteksiEl = document.getElementById('hasil-deteksi');
    const hasilConfidenceEl = document.getElementById('hasil-confidence');
    const hasilModeEl = document.getElementById('hasil-mode');
    const canvas = document.createElement('canvas');

    const pythonApiBaseUrl = 'http://127.0.0.1:5000';
    const sendIntervalMs = 120;
    const sendJpegQuality = 0.55;
    const processWidth = 416;
    const processHeight = 416;

    let currentStream = null;
    let cameraActive = false;
    let sendTimeoutId = null;
    let isSendingFrame = false;

    function stopSendingFrames() {
        if (sendTimeoutId) {
            clearTimeout(sendTimeoutId);
            sendTimeoutId = null;
        }
        isSendingFrame = false;
    }

    async function sendFrameToApi() {
        if (!cameraActive || isSendingFrame) {
            return;
        }

        if (!video.videoWidth || !video.videoHeight) {
            sendTimeoutId = setTimeout(sendFrameToApi, sendIntervalMs);
            return;
        }

        isSendingFrame = true;

        try {
            canvas.width = processWidth;
            canvas.height = processHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            const dataUrl = canvas.toDataURL('image/jpeg', sendJpegQuality);

            const response = await fetch(`${pythonApiBaseUrl}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image: dataUrl })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();

            hasilDeteksiEl.textContent = data.label || '-';
            hasilConfidenceEl.textContent = Number.isFinite(data.confidence)
                ? Number(data.confidence).toFixed(4)
                : '-';
            hasilModeEl.textContent = data.mode || '-';
        } catch (error) {
            hasilModeEl.textContent = 'API offline';
            hasilConfidenceEl.textContent = '-';
            hasilDeteksiEl.textContent = 'Belum terhubung';
        } finally {
            isSendingFrame = false;
            if (cameraActive) {
                sendTimeoutId = setTimeout(sendFrameToApi, sendIntervalMs);
            }
        }
    }

    function startSendingFrames() {
        stopSendingFrames();
        sendFrameToApi();
    }

    toggleBtn.onclick = function() {
        if (!cameraActive) {
            navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } } })
                .then(stream => {
                    video.srcObject = stream;
                    currentStream = stream;
                    cameraActive = true;
                    cameraWrapper.classList.add('is-camera-active');
                    toggleBtn.textContent = 'Nonaktifkan Kamera';
                    toggleBtn.classList.remove('btn-primary');
                    toggleBtn.classList.add('btn-danger');

                    // Set video size to match stream
                    const track = stream.getVideoTracks()[0];
                    const settings = track.getSettings();
                    video.width = settings.width || 1280;
                    video.height = settings.height || 720;

                    hasilDeteksiEl.textContent = 'Mendeteksi...';
                    hasilConfidenceEl.textContent = '-';
                    hasilModeEl.textContent = 'Menghubungkan API...';
                    startSendingFrames();
                })
                .catch(err => {
                    alert('Tidak dapat mengakses kamera: ' + err);
                });
            return;
        }

        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
            video.srcObject = null;
            currentStream = null;
        }
        cameraActive = false;
        stopSendingFrames();
        cameraWrapper.classList.remove('is-camera-active');
        toggleBtn.textContent = 'Aktifkan Kamera';
        toggleBtn.classList.remove('btn-danger');
        toggleBtn.classList.add('btn-primary');
        hasilDeteksiEl.textContent = '-';
        hasilConfidenceEl.textContent = '-';
        hasilModeEl.textContent = '-';
    };
</script>
@endsection
