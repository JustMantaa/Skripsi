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
        background: linear-gradient(135deg, rgba(255,255,255,0.98), rgba(245,248,255,0.98));
        border: 1px solid rgba(49, 91, 161, 0.16);
        border-radius: 16px;
        padding: 18px 18px 16px;
        margin-bottom: 14px;
        box-shadow: 0 10px 28px rgba(16, 24, 40, 0.08);
        font-size: 15px;
        color: #1f2937;
        display: flex;
        align-items: flex-start;
        gap: 14px;
    }

    .camera-tips:focus {
        outline: 3px solid rgba(59, 130, 246, 0.35);
        outline-offset: 3px;
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

    .camera-tips .tips-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 8px;
        padding: 4px 10px;
        border-radius: 999px;
        background: rgba(59, 130, 246, 0.12);
        color: #1d4ed8;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .camera-tips .tips-heading {
        margin: 0 0 8px;
        font-size: 18px;
        line-height: 1.25;
        font-weight: 800;
        color: #0f172a;
    }

    .camera-tips .tips-copy {
        margin: 0 0 10px;
        color: #334155;
        line-height: 1.6;
    }

    .camera-tips .tips-list {
        margin: 0;
        padding-left: 18px;
        color: #334155;
        line-height: 1.55;
    }

    .camera-tips .tips-list li + li {
        margin-top: 4px;
    }

    .camera-tips .info-icon {
        font-size: 20px;
        line-height: 1;
        color: #1d4ed8;
        margin-top: 2px;
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

    .camera-actions {
        display: flex;
        justify-content: center;
        gap: 12px;
        padding: 16px 16px 20px;
        flex-wrap: nowrap;
    }

    .camera-actions .btn {
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

    .camera-actions {
        padding: 12px;
        gap: 8px;
    }

    .camera-actions .btn {
        flex: 1 1 0;
        min-width: 0;
        width: auto;
        padding: 8px 10px;
        font-size: 14px;
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
                <div id="camera-tips" class="camera-tips" role="region" aria-label="Petunjuk penggunaan kamera" tabindex="-1">
                    <div class="tips-col-icon">
                        <i class="fa-solid fa-circle-info info-icon" aria-hidden="true"></i>
                    </div>
                    <div class="tips-col-text">
                        <div class="tips-eyebrow">Petunjuk Utama</div>
                        <h2 class="tips-heading">Baca ini dulu sebelum mulai</h2>
                        <p class="tips-copy">Agar hasil deteksi lebih akurat, ikuti petunjuk berikut saat berada di depan kamera.</p>
                        <ul class="tips-list">
                            <li>Gunakan tangan kanan dan gerakkan secara perlahan.</li>
                            <li>Pastikan pencahayaan cukup dan latar tidak terlalu ramai.</li>
                            <li>Disarankan memakai laptop atau PC dengan webcam.</li>
                            <li>Klik tombol "Reset" untuk mengulang deteksi.</li>
                        </ul>
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
                    <div class="camera-actions">
                        <button id="toggle-camera" class="btn btn-primary" type="button">Aktifkan Kamera</button>
                        <button id="reset-camera" class="btn btn-primary" type="button">Reset</button>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
@vite('resources/js/camera.js')
@endsection
