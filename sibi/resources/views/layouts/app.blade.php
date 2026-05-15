<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kamus SIBI</title>

    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">

    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">

    <link
        rel="stylesheet"
        href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.7.2/css/all.min.css"
        integrity="sha512-Evv84Mr4kqVGRNSgIGL/F/aIDqQb7xQ2vcrdIwxfjThSH8CSR7PBEakCr51Ck+w+/U6swU2Im1vVX0SVk9ABhg=="
        crossorigin="anonymous"
        referrerpolicy="no-referrer"
    />

    <style>
        :root {
            --primary-bg: #F7F7F7;
            --navbar-bg: #D4F6FF;
            --accent: #F8B195;
            --accent-hover: #F6D365;
            --text-main: #222;
            --card-bg: #FFFFFF;
        }

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            min-height: 100vh;
            background: var(--primary-bg);
            font-family: 'Poppins', sans-serif;
            color: var(--text-main);
        }

        .navbar {
            background: var(--navbar-bg) !important;
            padding-top: 0.35rem;
            padding-bottom: 0.35rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        }

        .navbar-brand {
            color: #000 !important;
            font-size: 1.15rem;
            font-weight: 700;
        }

        .navbar-logo {
            width: 58px;
            height: 58px;
            object-fit: contain;
        }

        .navbar .nav-link {
            color: #000 !important;
            font-size: 1rem;
            font-weight: 500;
            padding-left: 0.9rem !important;
            padding-right: 0.9rem !important;
        }

        .navbar .nav-link.active,
        .navbar .nav-link:focus,
        .navbar .nav-link:hover {
            color: var(--accent) !important;
        }

        .navbar-toggler {
            border: none;
            box-shadow: none !important;
        }

        .card {
            background: var(--card-bg);
            border-radius: 16px;
            box-shadow: 0 2px 8px rgba(108, 99, 255, 0.08);
        }

        .btn-primary {
            background: var(--accent);
            border: none;
            color: #333;
            font-weight: 600;
        }

        .btn-primary:hover,
        .btn-primary:focus {
            background: var(--accent-hover);
            color: #333;
        }

        .btn-danger {
            background: var(--navbar-bg);
            color: #333;
            border: none;
            font-weight: 600;
        }

        .btn-danger:hover,
        .btn-danger:focus {
            background: var(--accent);
            color: #333;
        }

        main {
            width: 100%;
        }

        @media (max-width: 768px) {
            .navbar {
                padding-left: 0.25rem;
                padding-right: 0.25rem;
            }

            .navbar-logo {
                width: 48px;
                height: 48px;
            }

            .navbar-brand {
                font-size: 1rem;
            }

            .navbar .nav-link {
                padding-top: 0.6rem;
                padding-bottom: 0.6rem;
            }
        }
    </style>
</head>

<body>
    <nav class="navbar navbar-expand-lg navbar-light px-3 px-md-4">
        <div class="container-fluid">
            <a class="navbar-brand d-flex align-items-center" href="{{ route('home') }}">
                <img
                    src="{{ asset('img/huruf/logo.png') }}"
                    alt="Logo SIBI"
                    class="navbar-logo d-inline-block align-text-top"
                >
                <span class="ms-2">Kamus SIBI</span>
            </a>

            <button
                class="navbar-toggler"
                type="button"
                data-bs-toggle="collapse"
                data-bs-target="#navbarNav"
                aria-controls="navbarNav"
                aria-expanded="false"
                aria-label="Toggle navigation"
            >
                <span class="navbar-toggler-icon"></span>
            </button>

            <div class="collapse navbar-collapse mt-2 mt-lg-0" id="navbarNav">
                <ul class="navbar-nav ms-auto align-items-lg-center">
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

    <main>
        @yield('content')
    </main>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>