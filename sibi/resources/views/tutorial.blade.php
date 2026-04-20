@extends('layouts.app')

@section('content')
<div class="container mt-5">
    <h2 class="mb-4">Kamus SIBI</h2>
    <div class="row row-cols-1 row-cols-md-4 g-4 justify-content-center">
        @foreach($huruf as $item)
        <div class="col">
            <div class="card h-100 text-center p-2 pt-3">
                <img src="{{ asset('img/huruf/' . $item['gambar']) }}" class="card-img-top mx-auto" alt="Gambar {{ $item['huruf'] }}" style="width: 225px; height: 225px; object-fit: contain;">
                <div class="card-body">
                    <h5 class="card-title display-6">{{ $item['huruf'] }}</h5>
                    <p class="card-text">{{ $item['penjelasan'] }}</p>
                </div>
            </div>
        </div>
        @endforeach
    </div>
</div>
@endsection
