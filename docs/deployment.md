# Deploying on Kubernetes

Verified end to end on a local `kind` cluster: image built, model mounted, pod healthy, real
generation served through the Service, metrics scraped, and a rollout completed **without
dropping an in-flight request**.

```bash
kind create cluster --config k8s/kind-cluster.yaml     # mounts models/ into the node
docker build -t microvllm:dev .
kind load docker-image microvllm:dev --name microvllm
kubectl apply -f k8s/
kubectl rollout status deploy/microvllm
```

---

## What was actually verified

| Check | Result |
|---|---|
| Image build | 138 MB local, **35.2 MB** on the node (model excluded) |
| Model mount | 491 MB GGUF visible at `/models` inside the node |
| Pod startup | Ready after model load, 0 restarts |
| Generation via Service | `"The capital of France is"` → `" Paris. It is the largest city in Europe…"` |
| `/metrics` via Service | counters, gauges, and histograms all scraping |
| Structured logging | `ttft_ms: 728.03`, `tpot_ms: 32.47`, `queue_wait_ms: 13.55` |
| **Rollout drain** | 180-token request begun **before** `rollout restart` completed with **HTTP 200** |

---

## Why the manifests look the way they do

Generic Kubernetes advice gets several things wrong for an LLM server. Each of these is a
deliberate departure, not a default.

### The model is not in the image

At 469 MB it would dominate the layer, push half a gigabyte on every rebuild, and pin the
image to one model. It is mounted instead, so the image is 35 MB and model-agnostic. On kind
that is a `hostPath` fed by `extraMounts`; a real cluster would use a ReadOnlyMany PVC or an
initContainer pulling from object storage. The container only ever reads a file at `/models`,
so it cannot tell the difference.

### `GGML_NATIVE=OFF`

The default tunes for the *build* machine's instruction set, which produces an image that
dies with `SIGILL` on any host with a smaller one. Portability beats a few percent for
something meant to be scheduled anywhere.

### Three probes, doing three different jobs

Collapsing them is the usual mistake, and each failure mode is distinct:

- **`startupProbe`** — model load takes ~13 s. Without this the liveness probe fires during
  startup and kills the pod before it can ever serve, producing a crash loop that looks like
  a broken image. The `Startup probe failed: connection refused` warnings during boot are this
  working correctly.
- **`readinessProbe`, deliberately slack** — under load the server returns 503 from a full
  queue while remaining perfectly healthy. An aggressive readiness probe would pull a busy pod
  out of the Service, pushing its load onto peers, which then do the same. That cascade is how
  a whole fleet fails at once.
- **`livenessProbe`, slacker still** — a restart discards every in-flight request and the KV
  cache with them. Reserve it for a genuinely wedged process.

### `terminationGracePeriodSeconds: 90`

The server drains on SIGTERM: stop accepting, finish in-flight work, exit. That takes as long
as the longest running generation, so the grace period must exceed it or Kubernetes SIGKILLs
mid-request on **every rollout**. The entrypoint is exec-form with no shell wrapper so the
process is PID 1 and receives the signal directly — a wrapper would swallow it and the drain
would never run.

This is the one behaviour most worth testing, because it is invisible until a deploy quietly
fails a fraction of live traffic. The drain test above is that test.

### `sessionAffinity: ClientIP`

Default round-robin is actively wrong here, for two reasons:

1. **KV cache is per-pod state.** Prefix sharing — including donors retained past their
   request's retirement — only pays off when a request reaches the pod holding the prefix.
   Round-robin makes that a 1-in-N chance and silently discards a measured 2.25× on
   repeated-prefix traffic. The feature does not break; it stops mattering, which is worse,
   because nothing reports it.
2. **Requests cost wildly different amounts** (16 vs 512 tokens). Round-robin keeps feeding a
   pod that is grinding through long generations — precisely the head-of-line blocking that
   continuous batching removed *inside* one process, reappearing *between* processes.

`ClientIP` is the crude fix available in a plain Service: it buys cache locality but keys on
client address, so it degrades behind a shared NAT and is blind to pod load. The real answer
is a queue-depth-aware proxy routing on a prompt-prefix hash. `/metrics` already publishes the
queue depth such a proxy would need.

### `strategy: Recreate`

A rolling update briefly runs two pods on one node, each loading 463 MiB of weights and
competing for the memory bandwidth the serving pod needs. Since decode is bandwidth-bound,
that is a self-inflicted latency spike on every deploy. Multi-node clusters with pod
anti-affinity can safely switch back to `RollingUpdate`.

### `requests == limits` (Guaranteed QoS)

Bandwidth sensitivity means throttling or a noisy neighbour shows up directly as inter-token
latency, not merely lower throughput. Guaranteed pods are also evicted last under node
pressure.

### Threads must match the CPU limit

The Phase 0 sweep measured 12 threads on 14 hardware threads dropping decode to 43.57 tok/s
against 63.39 at 8 — oversubscription costs real throughput, and under a cgroup limit it is
worse because the kernel throttles rather than descheduling politely.

---

## Scaling

**Replicas are not the first lever.** Decode is memory-bandwidth-bound — the thread sweep
measured 1.75× throughput for 5× the threads — so two pods on one node contend for the same
bandwidth and do not give 2×. Each replica also holds its own weights and its own KV cache;
nothing is shared. Scale across nodes, with anti-affinity, not within one.

**Do not autoscale on CPU.** A saturated LLM server pegs CPU at every load level, so CPU
utilization carries almost no signal. `microvllm_queue_depth` and `microvllm_kv_blocks_used`
are the meaningful ones, both already exported — reaching them from an HPA needs
prometheus-adapter or KEDA.

---

## Not done

- **No Helm chart or Kustomize overlays.** Plain YAML is honest for a single-service deploy;
  templating would be ceremony at this size.
- **No HPA manifest.** It would need a custom-metrics adapter to be anything other than
  wrong (see above), and shipping one wired to CPU would be worse than shipping none.
- **No Ingress/TLS.** The Service is `ClusterIP`; exposure is cluster-specific.
- **Verified on kind only** — a single-node local cluster. Multi-node behaviour (anti-affinity,
  real PVCs, cross-node bandwidth) is untested here.
