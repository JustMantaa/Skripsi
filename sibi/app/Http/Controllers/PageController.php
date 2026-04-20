<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;

class PageController extends Controller
{
    public function home()
    {
        return view('home');
    }

    public function tutorial()
    {
        $huruf = config('huruf');
        return view('tutorial', compact('huruf'));
    }

    public function about()
    {
        return view('about');
    }
}
