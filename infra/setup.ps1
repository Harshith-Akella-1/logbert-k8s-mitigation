# Stage 1: start Minikube, deploy Online Boutique and the prober, wait until every pod is healthy.
# Needs Docker Desktop running. Safe to re-run.
$ErrorActionPreference = 'Stop'
$boutique = 'https://raw.githubusercontent.com/GoogleCloudPlatform/microservices-demo/v0.10.3/release/kubernetes-manifests.yaml'

cmd /c "docker info >nul 2>&1"
if ($LASTEXITCODE -ne 0) { throw 'Docker is not running. Start Docker Desktop and wait for "Engine running".' }

# One node, 4 CPUs, 6 GB: fits in Docker Desktop's default share of a 16 GB laptop.
minikube start --driver=docker --cpus=4 --memory=6g
minikube addons enable metrics-server   # the HPA baseline reads CPU usage from this

kubectl apply -f $boutique
kubectl apply -f "$PSScriptRoot\prober.yaml"

Write-Host 'Waiting for pods (the first run downloads about 2 GB of images, allow 10 minutes)...'
kubectl wait --for=condition=Available deployment --all --timeout=900s
kubectl get pods

# The paper quotes the Kubernetes version from this file.
New-Item -ItemType Directory -Force "$PSScriptRoot\..\results" | Out-Null
kubectl version -o json | Out-File "$PSScriptRoot\..\results\cluster.json" -Encoding utf8

# Fault intervals are stamped with the laptop's clock and log lines with the node's clock.
$node = [int](minikube ssh -- date +%s)
$skew = $node - [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
Write-Host "Clock difference node - laptop: $skew s (run 'wsl --shutdown' and restart Docker if this is more than 2 s)"
