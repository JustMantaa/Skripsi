@extends('layouts.app')

@section('content')
<style>
    .kamus-section {
        padding: 32px 12px 48px;
    }

    .kamus-title {
        font-weight: 700;
        margin-bottom: 28px;
        text-align: center;
    }

    .kamus-card {
        border: none;
        border-radius: 18px;
        overflow: hidden;
        transition: 0.25s ease;
        height: 100%;
        background: #fff;
    }

    .kamus-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    }

    .kamus-image-wrapper {
        width: 100%;
        aspect-ratio: 1 / 1;
        background: #fafafa;

        display: flex;
        align-items: center;
        justify-content: center;

        overflow: hidden;
        padding: 12px;
    }

    .kamus-image {
        width: 100%;
        height: 100%;
        object-fit: contain;
    }

    .kamus-body {
        padding: 18px;
    }

    .kamus-huruf {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .kamus-text {
        font-size: 0.95rem;
        color: #555;
        margin-bottom: 0;
        line-height: 1.6;
    }

    /* Tablet */
    @media (max-width: 992px) {
        .kamus-section {
            padding: 28px 10px 40px;
        }

        .kamus-title {
            font-size: 2rem;
        }

        .kamus-huruf {
            font-size: 1.8rem;
        }

        .kamus-body {
            padding: 16px;
        }
    }

    /* Mobile */
    @media (max-width: 768px) {
        .kamus-section {
            padding: 24px 8px 36px;
        }

        .kamus-title {
            font-size: 1.8rem;
            margin-bottom: 22px;
        }

        .kamus-card {
            border-radius: 16px;
        }

        .kamus-image-wrapper {
            padding: 10px;
        }

        .kamus-huruf {
            font-size: 1.7rem;
        }

        .kamus-text {
            font-size: 0.92rem;
        }

        .kamus-body {
            padding: 14px;
        }
    }

    /* Small Mobile */
    @media (max-width: 480px) {
        .kamus-title {
            font-size: 1.6rem;
        }

        .kamus-huruf {
            font-size: 1.5rem;
        }

        .kamus-text {
            font-size: 0.88rem;
            line-height: 1.5;
        }

        .kamus-image-wrapper {
            padding: 8px;
        }

        .kamus-body {
            padding: 12px;
        }
    }
</style>

<div class="container-fluid kamus-section">
    <div class="container">
        <h2 class="kamus-title">Kamus SIBI</h2>

        <div class="row g-4 justify-content-center">
            @foreach($huruf as $item)
                <div class="col-12 col-sm-6 col-md-4 col-lg-3">
                    <div class="card kamus-card text-center">

                        <div class="kamus-image-wrapper">
                            <img
                                src="{{ asset('img/huruf/' . $item['gambar']) }}"
                                class="kamus-image"
                                alt="Gambar {{ $item['huruf'] }}"
                            >
                        </div>

                        <div class="card-body kamus-body">
                            <h5 class="kamus-huruf">
                                {{ $item['huruf'] }}
                            </h5>

                            <p class="kamus-text">
                                {{ $item['penjelasan'] }}
                            </p>
                        </div>

                    </div>
                </div>
            @endforeach
        </div>
    </div>
</div>
@endsection