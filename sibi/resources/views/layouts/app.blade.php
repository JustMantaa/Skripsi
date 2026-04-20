<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SIBI</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.7.2/css/all.min.css" integrity="sha512-Evv84Mr4kqVGRNSgIGL/F/aIDqQb7xQ2vcrdIwxfjThSH8CSR7PBEakCr51Ck+w+/U6swU2Im1vVX0SVk9ABhg==" crossorigin="anonymous" referrerpolicy="no-referrer" />
    <style>
        body {
            background: #F7F7F7;
            font-family: 'Poppins', sans-serif;
        }
        .navbar {
            background: #D4F6FF !important;
        }
        .navbar .navbar-brand, .navbar .nav-link {
            color: black !important;
        }
        .navbar .nav-link {
            font-size: 1.15rem;
            font-weight: 250;
        }
        .navbar .nav-link.active, .navbar .nav-link:focus, .navbar .nav-link:hover {
            color: #F8B195 !important;
        }
        .card {
            background: #FFFFFF;
            box-shadow: 0 2px 8px rgba(108,99,255,0.08);
            border-radius: 16px;
        }
        .btn-primary {
            background: #F8B195;
            border: none;
            color: #333;
        }
        .btn-primary:hover {
            background: #F6D365;
            color: #333;
        }
        .btn-danger {
            background: #D4F6FF;
            color: #333;
            border: none;
        }
        .btn-danger:hover {
            background: #F8B195;
            color: #333;
        }
    </style>
</head>
<body>
<nav class="navbar navbar-expand-lg navbar-light bg-light px-4">
    <div class="container-fluid">
        <a class="navbar-brand d-flex align-items-center " href="{{ route('home') }}">
            <img src="{{ asset('img/huruf/logo.png') }}" alt="Logo SIBI" width="60" height="60" class="d-inline-block align-text-top" style="object-fit: contain;">
            <span class="fw-bold ms-2 mb-1">Kamus SIBI</span>
        </a>
        <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
            <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="navbarNav">
            <ul class="navbar-nav ms-auto">
                <li class="nav-item">
                    <a class="nav-link" href="{{ route('home') }}">Home</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="{{ route('tutorial') }}">Kamus</a>
                </li>
            </ul>
        </div>
    </div>
</nav>
<div>
    @yield('content')
</div>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
