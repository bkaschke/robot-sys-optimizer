# CI helper (release.yml on GitHub and GitLab): start the built exe, wait for its "ready" line, load the page in headless Edge,
# check that the Viser client rendered, then stop the exe. Exits 1 with a message on any failure.
#   powershell -File packaging\ci_serve_check.ps1 -Exe dist\RobotCellOptimizer\RobotCellOptimizer.exe [-Port 8765]
# The caller blocks the network (firewall rule) and sets RSO_OFFLINE_GUARD / RSO_DATA; this script only checks.
param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [int]$Port = 8765,
    [int]$ReadyTimeoutSec = 120,
    [int]$VirtualTimeMs = 30000
)
$ErrorActionPreference = "Stop"

function Find-Edge {
    $c = @(@(
        "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe",
        "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe"
    ) | Where-Object { $_ -and (Test-Path $_) })
    if ($c) { return $c[0] }
    $cmd = Get-Command msedge -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

$tmp = Join-Path ([IO.Path]::GetTempPath()) ("rso-serve-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Path $tmp | Out-Null
$out = Join-Path $tmp "out.txt"
$err = Join-Path $tmp "err.txt"
$proc = $null
try {
    $edge = Find-Edge
    if (-not $edge) { throw "Microsoft Edge (msedge.exe) not found on this machine" }
    $proc = Start-Process -FilePath $Exe -ArgumentList @("--no-browser", "--port", "$Port") `
        -RedirectStandardOutput $out -RedirectStandardError $err -PassThru -WindowStyle Hidden
    $deadline = (Get-Date).AddSeconds($ReadyTimeoutSec)
    $ready = $false
    while ((Get-Date) -lt $deadline) {
        if ($proc.HasExited) { throw "the exe exited early with code $($proc.ExitCode)" }
        if ((Test-Path $out) -and (Select-String -Path $out -Pattern "Robot Cell Optimizer ready:" -Quiet)) {
            $ready = $true; break
        }
        Start-Sleep -Milliseconds 500
    }
    if (-not $ready) { throw "no 'Robot Cell Optimizer ready:' line within $ReadyTimeoutSec s" }
    Write-Host "exe ready: $(Select-String -Path $out -Pattern 'ready:' | Select-Object -First 1 -ExpandProperty Line)"

    $edgeProfile = Join-Path $tmp "edge-profile"
    $domFile = Join-Path $tmp "dom.html"
    # Two kinds of proof that the page works offline: (a) the page's own script connected its websocket to
    # the exe (viser logs "Connection opened"), (b) Edge dumped a DOM with the React root and a canvas.
    # On the GitHub runner (no GPU, 2026-11-19) Edge's software WebGL never let it dump the DOM, not even
    # with --timeout, while (a) came within seconds; locally both come in about 5 s. Either one passes.
    $edgeArgs = @("--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
        "--user-data-dir=`"$edgeProfile`"", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
        "--virtual-time-budget=$VirtualTimeMs", "--timeout=$(2 * $VirtualTimeMs)", "--dump-dom",
        "http://localhost:$Port/")
    $t0 = Get-Date
    $edgeProc = Start-Process -FilePath $edge -ArgumentList $edgeArgs -RedirectStandardOutput $domFile `
        -RedirectStandardError (Join-Path $tmp "edge-err.txt") -PassThru -WindowStyle Hidden
    $connectedAt = $null
    $deadline = $t0.AddSeconds(180)
    while ((Get-Date) -lt $deadline -and -not $edgeProc.HasExited) {
        if (-not $connectedAt -and (Select-String -Path $out -Pattern "Connection opened" -Quiet)) {
            $connectedAt = Get-Date
            $deadline = $connectedAt.AddSeconds(60)  # then give Edge one more minute to write the page
        }
        Start-Sleep -Milliseconds 500
    }
    $exited = $edgeProc.HasExited
    $secs = [int]((Get-Date) - $t0).TotalSeconds
    $conn = if ($connectedAt) { "after $([int]($connectedAt - $t0).TotalSeconds) s" } else { "no" }
    Write-Host "headless Edge: exited $exited after $secs s; page connected to the exe: $conn"
    if (-not $exited) { & taskkill /PID $edgeProc.Id /T /F | Out-Null; Start-Sleep -Seconds 2 }
    $dom = if (Test-Path $domFile) { Get-Content $domFile -Raw } else { "" }
    # the Viser client mounts React into #root; once connected it adds a canvas and the control panel
    $hasRoot = [bool]$dom -and ($dom -match '<div id="root"[^>]*>\s*<')
    $hasCanvas = [bool]$dom -and ($dom -match '<canvas')
    Write-Host "DOM: $($dom.Length) chars, root populated: $hasRoot, canvas: $hasCanvas"
    if (-not (($hasRoot -and $hasCanvas) -or $connectedAt)) {
        if ($dom) { $dom.Substring(0, [Math]::Min(2000, $dom.Length)) | Write-Host }
        throw "the Viser page did not render or connect (root populated: $hasRoot, canvas: $hasCanvas)"
    }
    if (Select-String -Path $out, $err -Pattern "offline-guard\] blocked" -Quiet) {
        throw "the app tried to reach the network (see [offline-guard] lines in its output)"
    }
    Write-Host "serve check: PASS"
}
catch {
    Write-Host "serve check: FAIL: $($_.Exception.Message)"
    foreach ($f in @($out, $err, (Join-Path $tmp "edge-err.txt"))) {
        if (Test-Path $f) { Write-Host "--- $f"; Get-Content $f -Tail 40 }
    }
    $script:failed = $true
}
finally {
    if ($proc -and -not $proc.HasExited) { & taskkill /PID $proc.Id /T /F | Out-Null }
    Start-Sleep -Milliseconds 500
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
if ($script:failed) { exit 1 }
