[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Model,

    [string]$HostOllamaUrl = $(if ($env:OLLAMA_HOST) { $env:OLLAMA_HOST } else { 'http://127.0.0.1:11434' }),

    [ValidateRange(1, 65535)]
    [int]$Port = 3001,

    [string]$ProjectName,

    [string]$CaFile,

    [switch]$ForceDockerOllama
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$modelConfigPath = Join-Path $projectRoot 'src/chemistry_bot/integrations/anythingllm_probe/tutor_models.json'
$modelConfiguration = Get-Content -LiteralPath $modelConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
$allowedModels = @($modelConfiguration.tutor_models)
if ($Model -notin $allowedModels) {
    throw "Unsupported tutor model '$Model'. Choose one of: $($allowedModels -join ', ')."
}

$supportModels = @($modelConfiguration.support_models.PSObject.Properties | ForEach-Object { $_.Value } | Select-Object -Unique)
$requiredModels = @(@($Model) + $supportModels | Select-Object -Unique)
$hostUrl = [Uri]$HostOllamaUrl.TrimEnd('/')
if ($hostUrl.Scheme -notin @('http', 'https')) {
    throw 'HostOllamaUrl must use http or https.'
}
$hostProbeBase = $hostUrl.GetLeftPart([System.UriPartial]::Authority) + $hostUrl.AbsolutePath.TrimEnd('/')
$containerHost = $hostUrl.Host
if ($containerHost -in @('localhost', '127.0.0.1', '::1')) {
    $containerHost = 'host.docker.internal'
}
$containerOllamaUrl = '{0}://{1}:{2}{3}' -f $hostUrl.Scheme, $containerHost, $hostUrl.Port, $hostUrl.AbsolutePath.TrimEnd('/')
$dockerOllamaUrl = 'http://ollama:11434'

$composePrefix = @()
if ($ProjectName) {
    $composePrefix += @('--project-name', $ProjectName)
}
$temporaryComposeOverlay = $null
$temporaryCaCopy = $null
$caDirectory = Join-Path $projectRoot 'data/compose-poc/trust'
$dockerOllamaCaCopy = Join-Path $caDirectory 'docker-ollama-runtime-ca.crt'
$dockerOllamaCaCopyCreated = $false
$dockerOllamaStarted = $false
$null = New-Item -ItemType Directory -Path $caDirectory -Force
if (-not (Test-Path -LiteralPath $dockerOllamaCaCopy)) {
    $null = New-Item -ItemType File -Path $dockerOllamaCaCopy
    $dockerOllamaCaCopyCreated = $true
}
elseif ((Get-Item -LiteralPath $dockerOllamaCaCopy).PSIsContainer) {
    throw "The runtime CA path '$dockerOllamaCaCopy' is a directory. Remove that empty generated directory and retry."
}
if ($CaFile) {
    try {
        $resolvedCa = (Resolve-Path -LiteralPath $CaFile).Path
        if ((Get-Item -LiteralPath $resolvedCa).PSIsContainer) {
            throw 'CaFile must be a file.'
        }
        $composeFile = Join-Path $projectRoot 'compose.yaml'
        $composePrefix += @('-f', $composeFile)
        $localOverride = Join-Path $projectRoot 'compose.override.yaml'
        if (Test-Path -LiteralPath $localOverride) {
            $composePrefix += @('-f', $localOverride)
        }

        # Compose/Buildx on Windows grants filesystem-read access to the build
        # context. Stage the selected public CA under the ignored data path so
        # Bake can read it, then pass it to Dockerfile only as a BuildKit secret.
        $temporaryCaCopy = Join-Path $caDirectory ("tutor-build-ca-$PID-$([Guid]::NewGuid().ToString('N')).crt")
        Copy-Item -LiteralPath $resolvedCa -Destination $temporaryCaCopy
        if ((Get-Item -LiteralPath $dockerOllamaCaCopy).Length -gt 0) {
            $existingCaHash = (Get-FileHash -LiteralPath $dockerOllamaCaCopy -Algorithm SHA256).Hash
            $providedCaHash = (Get-FileHash -LiteralPath $resolvedCa -Algorithm SHA256).Hash
            if ($existingCaHash -ne $providedCaHash) {
                throw "A different runtime CA already exists at '$dockerOllamaCaCopy'. Stop Docker Ollama and replace that generated file before using another CA."
            }
        }
        else {
            Copy-Item -LiteralPath $resolvedCa -Destination $dockerOllamaCaCopy
            $dockerOllamaCaCopyCreated = $true
        }

        $temporaryComposeOverlay = Join-Path $env:TEMP ("chemistry-tutor-ca-$PID-$([Guid]::NewGuid().ToString('N')).yaml")
        $caPathYaml = ConvertTo-Json -InputObject ($temporaryCaCopy -replace '\\', '/') -Compress
        $overlayText = @"
services:
  tutor:
    build:
      secrets:
        - tutor_build_ca
secrets:
  tutor_build_ca:
    file: $caPathYaml
"@
        [System.IO.File]::WriteAllText($temporaryComposeOverlay, $overlayText, [System.Text.UTF8Encoding]::new($false))
        $composePrefix += @('-f', $temporaryComposeOverlay)
    }
    catch {
        if ($temporaryCaCopy -and (Test-Path -LiteralPath $temporaryCaCopy)) {
            Remove-Item -LiteralPath $temporaryCaCopy -Force
        }
        if ($dockerOllamaCaCopyCreated -and $dockerOllamaCaCopy -and (Test-Path -LiteralPath $dockerOllamaCaCopy)) {
            Remove-Item -LiteralPath $dockerOllamaCaCopy -Force
        }
        if ($temporaryComposeOverlay -and (Test-Path -LiteralPath $temporaryComposeOverlay)) {
            Remove-Item -LiteralPath $temporaryComposeOverlay -Force
        }
        throw
    }
}

$environmentNames = @('CHEM_TUTOR_MODEL', 'CHEM_TUTOR_MODEL_OPTIONS', 'OLLAMA_HOST', 'ANYTHINGLLM_BIND', 'COMPOSE_PROFILES')
$previousEnvironment = @{}
$locationPushed = $false
foreach ($name in $environmentNames) {
    $previousEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}

function Invoke-Compose {
    param([string[]]$Arguments)

    $allArguments = @($script:composePrefix) + @($Arguments)
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        # Windows PowerShell 5.1 writes Compose progress to stderr. Capture
        # that stream as ordinary command output so Stop does not abort a
        # successful build; the process exit code remains the failure oracle.
        $ErrorActionPreference = 'Continue'
        $composeOutput = @(& docker compose @allArguments 2>&1)
        $composeExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    foreach ($line in $composeOutput) {
        Write-Output $line
    }
    if ($composeExitCode -ne 0) {
        throw "docker compose $($Arguments -join ' ') failed with exit code $composeExitCode."
    }
}

function Test-HostOllama {
    param([string]$BaseUrl, [string[]]$RequiredTags)

    try {
        $reply = Invoke-RestMethod -Uri "$($BaseUrl.TrimEnd('/'))/api/tags" -TimeoutSec 4
        $installed = @($reply.models | ForEach-Object { if ($_.model) { $_.model } elseif ($_.name) { $_.name } })
        $missing = @($RequiredTags | Where-Object { $_ -notin $installed })
        return [PSCustomObject]@{ Responding = $true; Missing = $missing }
    }
    catch {
        return [PSCustomObject]@{ Responding = $false; Missing = @($RequiredTags) }
    }
}

function Test-TutorContainerOllamaAccess {
    param()

    $probe = "import os,urllib.request; urllib.request.urlopen(os.environ['OLLAMA_HOST'].rstrip('/')+'/api/tags',timeout=3).read()"
    for ($attempt = 0; $attempt -lt 12; $attempt++) {
        try {
            Invoke-Compose @('exec', '-T', 'tutor', 'python', '-c', $probe) | Out-Null
            return $true
        }
        catch {
            Start-Sleep -Seconds 2
        }
    }
    return $false
}

function Start-DockerOllama {
    param([string[]]$Tags)

    [Environment]::SetEnvironmentVariable('COMPOSE_PROFILES', 'docker-ollama', 'Process')
    [Environment]::SetEnvironmentVariable('OLLAMA_HOST', $script:dockerOllamaUrl, 'Process')
    Write-Host 'Starting the isolated Compose Ollama service; models will be stored under data/compose-poc/ollama.'
    if ($script:dockerOllamaCaCopy) {
        if ((Get-Item -LiteralPath $script:dockerOllamaCaCopy).Length -gt 0) {
            Write-Host 'The supplied CA is mounted read-only into Docker Ollama for verified model-registry TLS. Its local trust copy remains under data/compose-poc/trust for container recreation.'
        }
    }
    Invoke-Compose @('up', '-d', 'ollama')
    $script:dockerOllamaStarted = $true

    $ready = $false
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        try {
            Invoke-Compose @('exec', '-T', 'ollama', 'ollama', 'ps') | Out-Null
            $ready = $true
            break
        }
        catch {
            Start-Sleep -Seconds 2
        }
    }
    if (-not $ready) {
        throw 'The Compose Ollama service did not become ready within 120 seconds.'
    }

    foreach ($tag in $Tags) {
        Write-Host "Ensuring Docker Ollama model is installed: $tag"
        Invoke-Compose @('exec', '-T', 'ollama', 'ollama', 'pull', $tag)
    }
}

try {
    Push-Location $projectRoot
    $locationPushed = $true
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw 'Docker CLI was not found on PATH.'
    }
    $dockerInfoErrorAction = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $null = & docker info 2>&1
        $dockerInfoExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $dockerInfoErrorAction
    }
    if ($dockerInfoExitCode -ne 0) {
        throw 'Docker Engine is not available. Start Docker Desktop and retry.'
    }

    [Environment]::SetEnvironmentVariable('CHEM_TUTOR_MODEL', $Model, 'Process')
    [Environment]::SetEnvironmentVariable('CHEM_TUTOR_MODEL_OPTIONS', ($allowedModels -join ','), 'Process')
    [Environment]::SetEnvironmentVariable('ANYTHINGLLM_BIND', "127.0.0.1:$Port", 'Process')

    $hostProbe = Test-HostOllama -BaseUrl $hostProbeBase -RequiredTags $requiredModels
    $useHost = -not $ForceDockerOllama -and $hostProbe.Responding -and $hostProbe.Missing.Count -eq 0
    if ($useHost) {
        Write-Host "Host Ollama has the selected tutor model and fixed support models. Selected model: $Model"
        [Environment]::SetEnvironmentVariable('OLLAMA_HOST', $containerOllamaUrl, 'Process')
        [Environment]::SetEnvironmentVariable('COMPOSE_PROFILES', '', 'Process')
    }
    else {
        if ($ForceDockerOllama) {
            Write-Host 'Docker Ollama was explicitly requested.'
        }
        elseif (-not $hostProbe.Responding) {
            Write-Host 'Host Ollama did not respond; using the Docker fallback.'
        }
        else {
            Write-Host "Host Ollama is missing required model tag(s): $($hostProbe.Missing -join ', '). Using the Docker fallback."
        }
        [Environment]::SetEnvironmentVariable('COMPOSE_PROFILES', 'docker-ollama', 'Process')
        [Environment]::SetEnvironmentVariable('OLLAMA_HOST', $dockerOllamaUrl, 'Process')
    }

    Invoke-Compose @('config', '--quiet')
    Invoke-Compose @('build', 'tutor')

    if ($useHost) {
        Invoke-Compose @('up', '-d', '--wait', 'tutor')
        if (-not (Test-TutorContainerOllamaAccess)) {
            Write-Host 'The tutor container cannot reach host Ollama; switching to the Docker fallback.'
            Invoke-Compose @('--profile', 'docker-ollama', 'stop', 'tutor')
            $useHost = $false
            Start-DockerOllama -Tags $requiredModels
        }
    }
    else {
        Start-DockerOllama -Tags $requiredModels
    }

    if (-not $useHost) {
        Invoke-Compose @('up', '-d', '--wait', 'tutor', 'anythingllm')
    }
    else {
        Invoke-Compose @('up', '-d', '--wait', 'anythingllm')
    }

    $pingUri = "http://127.0.0.1:$Port/api/ping"
    $ready = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        try {
            $ping = Invoke-RestMethod -Uri $pingUri -TimeoutSec 3
            if ($ping.online) { $ready = $true; break }
        }
        catch {
            Start-Sleep -Seconds 2
        }
    }
    if (-not $ready) {
        throw "AnythingLLM did not become ready at $pingUri. Check 'docker compose logs'."
    }

    Write-Host "AnythingLLM is ready at http://localhost:$Port/"
    Write-Host "Active tutor model: $Model"
    Write-Host "Active Ollama backend: $(if ($useHost) { 'host Ollama' } else { 'Docker Ollama' })"
}
finally {
    foreach ($name in $environmentNames) {
        [Environment]::SetEnvironmentVariable($name, $previousEnvironment[$name], 'Process')
    }
    if ($temporaryComposeOverlay -and (Test-Path -LiteralPath $temporaryComposeOverlay)) {
        Remove-Item -LiteralPath $temporaryComposeOverlay -Force
    }
    if ($temporaryCaCopy -and (Test-Path -LiteralPath $temporaryCaCopy)) {
        Remove-Item -LiteralPath $temporaryCaCopy -Force
    }
    if ($dockerOllamaCaCopyCreated -and -not $dockerOllamaStarted -and $dockerOllamaCaCopy -and (Test-Path -LiteralPath $dockerOllamaCaCopy)) {
        Remove-Item -LiteralPath $dockerOllamaCaCopy -Force
    }
    if ($locationPushed) {
        Pop-Location
    }
}
