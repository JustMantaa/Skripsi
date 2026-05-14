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

    #video:focus {
        outline: none;
    }

    #video-container {
        position: relative;
        width: 100%;
        height: 720px;
    }

    /* bbox overlay removed - not used */
</style>
<div class="container mt-3">
    </div>
    <div class="row mb-4">
        <div class="col-lg-8 mx-auto p-2">
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
                    <div class="d-flex justify-content-center p-3 gap-2">
                        <button id="toggle-camera" class="btn btn-primary">Aktifkan Kamera</button>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
@vite('resources/js/camera.js')
@endsection
