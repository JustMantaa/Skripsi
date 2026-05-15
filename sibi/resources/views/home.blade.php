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
        font-weight: bold;
        max-width: calc(100% - 20px);
        box-shadow: 0 2px 10px rgba(0,0,0,0.12);
    }

    .camera-status small {
        display: block;
        font-weight: 500;
        color: #555;
        word-break: break-word;
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

    #video-container {
        position: relative;
        width: 100%;
        background: #000;
        overflow: hidden;
        border-radius: 12px;
    }

    #video {
        width: 100%;
        height: auto;
        max-height: 80vh;
        display: block;
        background: #f3f3f3;
        object-fit: cover;
        transform: scaleX(-1);
        border-radius: 12px;
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
        #video {
            max-height: 70vh;
        }

        .camera-placeholder-icon {
            font-size: 80px;
        }
    }

    /* Mobile */
    @media (max-width: 768px) {
        .container {
            padding-left: 10px;
            padding-right: 10px;
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
            font-size: 12px;
        }

        #video {
            width: 100%;
            height: auto;
            max-height: 65vh;
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
        #video {
            max-height: 60vh;
        }

        .camera-status {
            font-size: 13px;
        }

        .camera-status small {
            font-size: 11px;
        }

        .camera-placeholder-icon {
            font-size: 60px;
        }
    }
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
