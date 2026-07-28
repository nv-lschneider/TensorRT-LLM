# NCCL Window Pool Reuse and CUDA Graph Lifetime Fix Plan

## Objective

Fix registered-window lifetime, device ownership, and cache-identity bugs without
disabling window reuse or adding CUDA graph replay overhead.

This plan does not depend on an NCCL version change. It also does not disable
`NCCL_SYMMETRIC` during capture. The fix belongs in TRT-LLM's registered-window
allocator and CUDA graph ownership model.

All window allocation and NCCL registration remain outside CUDA graph capture.
During capture, TRT-LLM may select an already-registered window and bind its
lifetime to the graph.

## Correctness invariant

Two logical leases may use the same address if and only if every device access
from the old lease happens-before every device access from the new lease.

Tensor destruction ends host ownership, but does not prove that GPU work has
completed. Safe reuse therefore needs both:

1. An address-lifetime proof that prevents deregistration or foreign reuse while
   a graph can still replay the address.
2. An ordering proof between the old and new logical uses.

The common fast path remains safe:

- The old value's final consumer has already been enqueued.
- The next borrower uses the exact same captured stream.
- No outstanding consumer exists on another stream.
- The next logical use overwrites the scratch buffer before reading it.

Same capture ID alone is not sufficient because one capture may contain forked
streams. Cross-stream reuse requires an explicit dependency that joins every old
consumer into the new-use stream.

## Reuse policy

| Situation | Policy |
| --- | --- |
| Same capture, same stream, after the final old consumer is enqueued | Reuse immediately |
| Same capture, different streams | Reuse only after an explicit captured join |
| Repeated replay of one graph exec | Reuse its embedded scratch addresses |
| Different graph execs in one serial runner domain | Share an arena when replay cannot overlap |
| Graph and eager work on the same enforced lane | Allow only for overwrite-before-read scratch |
| Unrelated graphs, domains, devices, or streams | Do not share while graph bindings exist |
| Window-backed value escaping to an unordered consumer | Keep dedicated until the consumer rejoins |

## Allocator model

Replace the current `BufferEntry::{buffer, inUse}` state with independent
resource and lease state:

```text
RegisteredWindow
  communicator identity
  exact CUDA device
  pointer and registered byte range
  stable entry identity

TransientLease
  monotonically increasing generation
  reuse domain
  acquisition/home stream
  capture ID, when captured
  optional retire frontier for explicit cross-stream joins

WindowReuseDomain
  stable domain ID
  communicator and device
  enforced replay/launch lane
  registered high-water arena
  graph binding count
  closing state

GraphBinding
  domain control block
  capture ID
  CUDA user-object reference
  atomic retirement state
```

Ending a `TransientLease` makes a slot eligible for a correctly ordered borrower.
It does not end its `GraphBinding`. A live graph binding keeps the physical
registration unavailable to foreign domains while still allowing temporal reuse
inside its own serial domain.

Tensor deleters carry both the stable entry identity and lease generation. A
stale deleter cannot release a newer lease of the same address.

## Commit plan

Every commit should build and pass its affected unit tests. New behavior should
be enabled only in the commit that also supplies the corresponding cleanup and
failure handling.

### Commit 1: `fix(nccl-window): make allocator device ownership explicit`

Make device identity correct before adding asynchronous graph lifetime state.

Changes:

- Pass the exact `c10::Device` or `DeviceIndex` through
  `allocate_output`, `createNCCLWindowTensor`, and `requestBuffer`.
- Construct `from_blob` tensors as `CUDA:<index>` under a `CUDAGuard`.
- Store the owning device in every registered-window entry.
- Key allocator state by communicator identity and exact device.
- Key communicator lookup by rank group and device, or store the device in the
  cached communicator and assert exact agreement.
- Replace process-global window-support caching with per-device caching.
- Execute allocation, pointer lookup, release, and cleanup under the recorded
  device.
- Make entries address-stable in preparation for asynchronous graph callbacks.

Tests:

- Allocate a device-1 window while device 0 is current.
- Verify tensor device, pointer attributes, stream, and pool entry all identify
  device 1.
- Verify the same rank group on two devices cannot reuse a communicator-cache
  entry.
- Change the current device before release and cleanup and verify the recorded
  device is still used.

### Commit 2: `fix(nccl-window): use generation-safe transient leases`

Separate logical tensor ownership from physical registration.

Changes:

- Return an entry identity and lease generation from `requestBuffer`.
- Capture both in the tensor storage deleter.
- Make `releaseBuffer` succeed only for the currently active generation.
- Store the home stream and reuse domain at acquisition time.
- Preserve immediate, event-free same-stream eager reuse.
- Do not infer safety for arbitrary cross-stream tensor use.
- Add an explicit internal API for known stream use and dependency joins.
- Use separate slots for unordered streams rather than adding implicit waits to
  the normal path.

Tests:

- A stale deleter cannot release a reissued lease.
- Same-stream eager allocate/release/reallocate returns the same pointer
  immediately.
- A different unordered stream cannot borrow that pointer.
- An explicit event/dependency join makes it eligible again.

### Commit 3: `refactor(nccl-window): add graph reuse-domain primitives`

Introduce graph-aware state without changing the active capture policy yet.

Changes:

- Add stable `WindowReuseDomain` and graph-binding control blocks.
- Add a thread-local `WindowCaptureScope` carrying domain, device, capture ID,
  capture stream, and underlying `cudaGraph_t`.
- Obtain capture information once per capture with
  `cudaStreamGetCaptureInfo_v2`, rather than querying on every allocation.
- Add domain-aware eligibility checks to the best-fit allocator.
- Add debug tracing for entry ID, pointer, bytes, device, stream, generation,
  domain, capture ID, and graph-reference count.
- Add capture-abort and domain-closing states.

This commit keeps the existing capture behavior disabled behind an internal
feature switch so that the new state model is independently reviewable.

Tests:

- Domain and stream identity participate in allocator eligibility.
- Domain control blocks remain address-stable.
- Capture-scope nesting and failure cleanup restore thread-local state.
- Debug traces identify every acquire, release, bind, and deferred retirement.

### Commit 4: `fix(nccl-window): retain registered arenas for CUDA graph lifetime`

Bind already-registered storage to graph lifetime and enable safe intra-capture
reuse.

Changes:

- On the first domain use during capture, create a CUDA user object and move one
  reference into the active graph with `cudaGraphRetainUserObject`.
- Attach once per graph/domain, not once per layer or buffer request.
- Let graph instantiation transfer the lifetime reference to the graph exec.
- Keep all NCCL allocation and window registration outside capture.
- Allow a released slot to be reused later in the same capture only when all
  known uses are confined to the exact same captured stream.
- Reject same-capture cross-stream reuse unless an explicit dependency join is
  recorded.
- Keep graph-bound arena storage unavailable to foreign eager callers and
  unrelated captures.
- Make the CUDA user-object destructor perform only an atomic retirement update
  or lock-free notification. It must not call CUDA/NCCL, block, or acquire the
  allocator lock.
- Drain retirement notifications lazily from ordinary host allocator calls.
- Quarantine bindings after capture invalidation until CUDA retires them.

Tests:

- Capture `producer -> consumer`, destroy the intermediate tensor, and allocate
  again later on the same captured stream. Require pointer reuse and correct
  results over at least 1,000 replays.
- Confirm that the normal sequential chain still uses its two-buffer ping-pong.
- Fork an auxiliary capture stream and keep consuming the old value. Reuse must
  be rejected before its join and allowed after it.
- Drop all strong tensor references while retaining the graph. A foreign eager
  allocation must not receive the graph's addresses.
- Assert that allocation and registration counters remain unchanged during
  capture.
- Invalidate capture after binding and verify eventual safe retirement.

### Commit 5: `fix(cuda-graph): share window arenas in serial replay domains`

Connect `CUDAGraphRunner` to the graph-aware allocator without dedicating a
window set to every graph key.

Changes:

- Give each runner a stable `WindowReuseDomain`.
- Enter its `WindowCaptureScope` around warmup/capture.
- Use one enforced replay/launch lane per domain.
- Allow batch-size graph variants in the domain to embed the same arena slots
  when they are never replayed concurrently.
- Require all internal auxiliary-stream work to rejoin the domain lane.
- Require external window consumers to rejoin before the next arena borrower.
- Classify a window-backed graph output as escaping. Keep it dedicated unless
  its consumption is explicitly ordered back into the domain.
- Give unknown capture clients isolated per-graph domains.
- Add debug/test assertions for wrong-device, wrong-stream, and concurrent
  replay. Do not add release-build replay-side allocator calls.

Tests:

- Capture multiple batch-size keys, alternate them for at least 10,000 launches,
  and verify that they share the same high-water arena and remain numerically
  correct.
- Attempt concurrent replay on different streams and require separate domains
  or a debug rejection.
- Alternate graph and eager window operations on the same explicit domain lane.
- Verify that unordered graph/eager alternation cannot borrow the graph arena.
- Verify that an escaping weak graph output is not overwritten before its last
  consumer.

### Commit 6: `fix(nccl-window): make graph-aware teardown deterministic`

Prevent deregistration while a graph or outstanding launch can still use a
window.

Changes:

- Mark the domain and communicator closing and reject new leases.
- Stop new replay and capture for the domain.
- Reset/destroy all graph objects belonging to the runner.
- Synchronize the domain replay lane during teardown only.
- Drain graph user-object retirement notifications.
- Deregister and free windows only after all graph bindings and transient leases
  have retired, while the communicator is still valid.
- Execute cleanup under the window's recorded CUDA device.
- Make runner `clear()` and destructors idempotent.
- Remove assumptions based on global/static destruction order.

Tests:

- Call graph reset while a replay is in flight. The address must not become
  available until completion.
- Attempt communicator cleanup with live graph bindings and verify that cleanup
  defers safely.
- Repeat capture, replay, clear, and recapture and require allocator counts and
  registered bytes to return to baseline.
- Change the current device before teardown.

### Commit 7: `fix(allreduce): preallocate registered windows by byte capacity`

Make preallocation describe the actual resource requirement.

Changes:

- Replace `_prealloc_done`'s `(group, num_tokens)` identity with a per-device,
  per-group, per-domain byte high-water record.
- Include dtype, hidden size, requested bytes, and peak simultaneous slot count
  in capacity planning.
- Remove or roll back the high-water marker if preallocation fails.
- During eager warmup, measure peak simultaneous registered-window demand.
- Materialize any missing arena capacity before capture begins.
- Preserve best-fit reuse for smaller requests.
- Fail capture clearly if prepared capacity is insufficient; never register a
  new window during capture.

Tests:

- Identical token counts with different dtypes, hidden sizes, or devices reserve
  the correct second capacity.
- Smaller later shapes reuse the high-water arena.
- The production linear/all-reduce chain retains its measured two-slot peak
  rather than allocating once per layer or graph.
- Capture performs zero allocation and registration calls.

### Commit 8: `fix(allreduce): align tactic tuning and strategy propagation`

Fix the surrounding cache and configuration defects without changing the
allocator lifetime policy.

Changes:

- Keep `input_uses_nccl_window` in tactic identity because the paths may have
  different costs.
- Warm the actual runtime window/non-window modes and decode shapes so capture
  does not encounter a mode-mismatched cache miss.
- Include exact device, dtype, shape, byte size, and relevant all-reduce mode in
  tuning/preallocation identities.
- Audit and fix `allreduce_strategy` propagation through embeddings, LM head,
  MTP `eh_proj`, and every nested all-reduce runner.
- Add diagnostics that identify the module and complete tactic key on a miss.

Tests:

- Warmup and runtime window modes produce the same intended tactic-cache key.
- MTP decode shapes hit a warmed tactic.
- An explicit all-reduce strategy reaches every nested module.
- Different devices or allocation modes do not collide in caches.

### Commit 9: `test(allreduce): cover window lifetime under serving churn`

Add the end-to-end regression and performance acceptance suite.

Correctness coverage:

- Static and churning online workloads.
- Single and multiple CUDA graph keys.
- Graph/eager alternation.
- Same-stream and auxiliary-stream execution.
- TEP and DEP.
- MTP rejection sampling at full generation length.
- Forced allocator pressure.
- Repeated runner and communicator teardown.
- Full-length per-segment acceptance length and output correctness.

Performance gates:

- Identical CUDA graph node count and kernel sequence before and after the fix.
- No new replay-time event, wait, synchronization, copy, allocation, callback,
  or allocator lookup.
- The normal serial runner keeps the same registered-window high-water count,
  expected to remain two for the common size class.
- Same-stream eager reuse remains an in-memory fast path.
- Decode latency, TPOT, and throughput remain within the agreed measurement
  noise band, initially one percent.
- Cross-stream synchronization cost, if explicitly requested by a caller, is
  reported separately and never charged to the same-stream path.

## Non-goals

The implementation must not:

- Depend on upgrading or downgrading NCCL.
- Disable `NCCL_SYMMETRIC` merely because capture is active.
- Add `cudaDeviceSynchronize()` or stream events to graph replay.
- Retain every captured tensor strongly for the entire graph lifetime.
- Allocate one registered window set per layer or per batch-size graph.
- Treat capture ID equality as proof of cross-stream ordering.

## Completion criteria

The work is complete when:

1. Registered addresses cannot be reused by a foreign domain or deregistered
   while any graph can replay them.
2. Same-capture, same-stream reuse and the existing two-buffer ping-pong remain
   enabled.
3. Serial graph variants share one high-water arena without concurrent aliasing.
4. Device, communicator, generation, cache, and teardown identities are exact.
5. The long-running online MTP reproduction remains numerically correct with
   healthy full-length segment acceptance.
6. CUDA graph replay has no additional synchronization or host-side overhead.
