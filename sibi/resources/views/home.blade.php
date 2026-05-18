@extends('layouts.app')

@section('content')
<style>
    .camera-wrapper {
        position: relative;
        overflow: hidden;
        border-radius: 12px;
    }

    .camera-status {
        position: absolute;
        top: 10px;
        left: 10px;
        z-index: 2;
        background: rgba(255, 255, 255, 0.92);
        padding: 8px 14px;
        border-radius: 10px;
        font-size: 28px;
        font-weight: bold;
        max-width: calc(100% - 20px);
        box-shadow: 0 2px 10px rgba(0,0,0,0.12);
    }

    .camera-status small {
        display: block;
        font-weight: 500;
        color: #555;
        word-break: break-word;
        font-size: 16px;
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
        font-size: 100px;
        color: rgba(46, 45, 45, 0.9);
        line-height: 1;
    }

    .is-camera-active .camera-placeholder {
        display: none;
    }

    .camera-tips {
        background: rgba(255,255,255,0.95);
        border-top-left-radius: 12px;
        border-top-right-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 8px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.06);
        font-size: 15px;
        color: #222;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .camera-tips .tips-col-icon,
    .camera-tips .tips-col-close {
        display: flex;
        align-items: center;
        justify-content: center;
        flex: 0 0 auto;
    }

    .camera-tips .tips-col-text {
        flex: 1 1 auto;
    }

    .camera-tips .info-icon {
        font-size: 18px;
        line-height: 1;
    }

    .camera-tips .close-tips {
        border: none;
        background: transparent;
        font-size: 20px;
        line-height: 1;
        color: #666;
        padding: 4px 6px;
        cursor: pointer;
    }

    #video-container {
    position: relative;
    width: 100%;
    aspect-ratio: 16 / 9;
    max-height: 90vh;
    overflow: hidden;
    background: #f3f3f3;
}

    #video {
        width: 100%;
        height: 100%;
        display: block;
        background: #f3f3f3;
        object-fit: cover;
        transform: scaleX(-1);
    }

    #video:focus {
        outline: none;
    }

    .camera-controls {
        display: flex;
        justify-content: center;
        padding: 16px;
        gap: 12px;
        flex-wrap: wrap;
    }

    .camera-controls .btn {
        min-width: 180px;
        font-weight: 600;
        border-radius: 10px;
        padding: 10px 16px;
    }

    /* Tablet */
@media (max-width: 992px) {
    #video-container {
        height: 90vh;
        min-height: unset;
        max-height: unset;
        aspect-ratio: unset;
    }

    #video {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }

    .camera-placeholder-icon {
        font-size: 80px;
    }
}

/* Mobile */
@media (max-width: 768px) {
    .container-fluid {
        padding-left: 8px !important;
        padding-right: 8px !important;
    }

    .camera-wrapper {
        border-radius: 10px;
    }

    .camera-status {
        top: 8px;
        left: 8px;
        padding: 8px 12px;
        font-size: 14px;
    }

    .camera-status small {
        font-size: 13px;
    }

    #video-container {
        height: 90vh;
        min-height: unset;
        max-height: unset;
        aspect-ratio: unset;
    }

    #video {
        width: 100%;
        height: 100%;
        object-fit: cover;
        border-radius: 10px;
    }

    .camera-placeholder-icon {
        font-size: 70px;
    }

    .camera-controls {
        padding: 12px;
    }

    .camera-controls .btn {
        width: 100%;
        min-width: unset;
    }
}

/* Small Mobile */
@media (max-width: 480px) {
    #video-container {
        height: 90vh;
    }

    .camera-status {
        font-size: 13px;
    }

    .camera-status small {
        font-size: 12px;
    }

    .camera-placeholder-icon {
        font-size: 60px;
    }
}

/* Tips responsive adjustments */
@media (max-width: 768px) {
    .camera-tips {
        padding: 10px 12px;
    }

    .camera-tips .info-icon {
        font-size: 16px;
    }

    .camera-tips .close-tips {
        font-size: 18px;
        padding: 2px 6px;
    }

    .camera-tips .tips-col-text p {
        font-size: 14px;
    }
}

@media (max-width: 480px) {
    .camera-tips {
        padding: 8px 10px;
    }

    .camera-tips .info-icon {
        font-size: 14px;
    }

    .camera-tips .close-tips {
        font-size: 18px;
        padding: 2px 4px;
    }

    .camera-tips .tips-col-text p {
        font-size: 13px;
    }
}
</style>
<div class="container-fluid px-2 px-md-4 mt-3">
    <div class="row mb-4 mx-0">
        <div class="col-12 col-lg-8 mx-auto p-2">
            <div>
                <div id="camera-tips" class="camera-tips" role="region" aria-label="Tips Kamera">
                    <div class="tips-col-icon">
                        <i class="fa-solid fa-circle-info info-icon" aria-hidden="true"></i>
                    </div>
                    <div class="tips-col-text">
                        <p class="mb-0">Gunakan tangan kanan dan lakukan gerakan secara perlahan di depan kamera untuk hasil deteksi yang lebih akurat. Pastikan pencahayaan cukup dan latar belakang tidak terlalu ramai.</p>
                    </div>
                    <div class="tips-col-close">
                        <button class="close-tips" aria-label="Tutup tips" onclick="document.getElementById('camera-tips').style.display='none'">&times;</button>
                    </div>
                </div>
            </div>

            <div class="card camera-wrapper" id="camera-wrapper">
                <div class="card-body p-0" style="position: relative;">
                    <div class="camera-status">
                        <span id="hasil-deteksi">-</span>
                        <small>Confidence: <span id="hasil-confidence">-</span></small>
                        <small>Mode: <span id="hasil-mode">-</span></small>
                    </div>
                    <div id="camera-placeholder" class="camera-placeholder" aria-hidden="true">
                        <i class="fa-solid fa-video-slash camera-placeholder-icon"></i>
                    </div>
                    <div id="video-container">
                        <video id="video" autoplay playsinline></video>
                    </div>
                    <div class="camera-controls">
                        <button id="toggle-camera" class="btn btn-primary">Aktifkan Kamera</button>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
@vite('resources/js/camera.js')
@endsection