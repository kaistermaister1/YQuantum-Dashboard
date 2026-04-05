param(
    [string]$Output = "qaoa_grid_viewer.exe"
)

$ErrorActionPreference = "Stop"

$gpp = (Get-Command g++ -ErrorAction SilentlyContinue)
if (-not $gpp) {
    throw "g++ was not found on PATH."
}

$pkgConfig = (Get-Command pkg-config -ErrorAction SilentlyContinue)
$pkgFlags = ""
if ($pkgConfig) {
    try {
        $pkgFlags = (& pkg-config --cflags --libs raylib 2>$null)
    } catch {
        $pkgFlags = ""
    }
}

if (-not $pkgFlags) {
    throw "raylib was not found via pkg-config. Install raylib and ensure pkg-config can resolve raylib.pc."
}

$command = "g++ -std=c++17 -O2 -Wall -Wextra main.cpp -o $Output $pkgFlags"
Write-Host $command
Invoke-Expression $command
