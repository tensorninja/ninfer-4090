# Turn checkpoint ring

`--turn-checkpoints N` keeps up to N past turn checkpoints per slot in host memory.
A prompt that diverges from the resident session in the middle of its history then
restores at the deepest checkpoint below the divergence point. The server
re-prefills only the suffix after that checkpoint. Without the ring, the same
prompt re-prefills from token zero.

The flag defaults to `0` (off). The recommended production value on Qwen3.8-27B is
`32`. The policy mirrors llama.cpp's `--ctx-checkpoints`, adapted to this engine's
hybrid attention.

## Why the engine needs checkpoints at all

Qwen3.8-27B is a hybrid model: 16 of its 64 layers use full attention, the other
48 use GDN linear attention. The two halves age differently when a prompt edits
history:

- Full-attention KV is positional. The engine can truncate it to any frontier and
  the entries below the edit stay valid.
- Each GDN layer keeps one running recurrent state per slot. That state integrates
  every token in order and cannot rewind. The engine can resume only at a point
  where the state was copied aside.

Without the ring, the engine holds exactly one such copy: the turn checkpoint at
the start of the current turn. A prompt that edits anything before that point
forces a full re-prefill, whatever its depth. The ring retains the older copies
that the single checkpoint used to overwrite.

## What one checkpoint holds

| Component | Content | Size on Qwen3.8-27B |
|---|---|---:|
| Recurrent state | 48 layers x 128x128x48 fp32 | 144 MiB |
| Convolution state | 48 layers x 10240x3 bf16 | 2.9 MiB |
| Boundary hidden | 5120 bf16 | 10 KiB |

One entry therefore costs about 147 MiB of pageable host memory. The attention KV
is not copied. An entry stays valid only while the token ledger below its frontier
is untouched, so the slot's paged KV still holds those positions on the GPU.

## How to run it

Add the flag to any serve line:

```bash
ninfer-serve models/qwen3_8_27b.ninfer ... --turn-checkpoints 32
```

Host memory cost is `N x 147 MiB x --max-concurrency`, plus one fixed pinned
staging entry of 147 MiB per slot. GPU memory is unchanged.

| `--turn-checkpoints` | Host memory per slot | History covered |
|---:|---:|---:|
| 8 | 1.2 GiB | ~33K tokens |
| 32 | 4.6 GiB | ~131K tokens |
| 64 | 9.2 GiB | full 262K context |

The history coverage follows from the compaction spacing described below. Values
above 64 waste memory: the spacing can never retain more than
`context / 4096 = 64` entries.

## Capture and compaction

The engine captures a checkpoint when prefill crosses the current turn boundary,
which is the end of the last user message. The copy to host runs asynchronously on
the engine stream at a point where the stream is idle, so capture does not delay
the first token. The next request for the slot folds the staged copy into the
ring, or discards it when that request rewrote history below the staged frontier.

Ring compaction runs at each fold:

1. An entry with the same frontier is replaced.
2. Entries within 4,096 tokens of a deeper neighbour are folded away
   (`kTurnCheckpointMinStep`). The newest entry always survives.
3. Above capacity, the oldest entry is evicted.

Consequence: retained entries sit at least 4,096 tokens apart, so N entries cover
about `N x 4096` tokens of rewindable history. Short consecutive turns share one
checkpoint; a rewind to an unretained turn restores at the nearest deeper entry
and re-prefills the gap.

## Restore

Reuse planning tries these sources in order:

1. Exact frontier extension (`AppendAtFrontier`).
2. The resident turn checkpoint (`RestoreTurnCheckpoint`).
3. The deepest still-matching host rewind point, which is either a ring entry
   (`RestoreTurnCheckpoint`) or the user-turn anchor (`RestoreUserTurnAnchor`).

Step 3 merges two independent sources. Ring entries sit wherever the checkpoint
policy placed a checkpoint, which is always after the last user message's
content; the user-turn anchor sits at that message's opener. A client that
rewrites the tail of the newest user message therefore invalidates the ring
entry for that turn while the anchor survives, and the two are not redundant.
The ring is ascending by frontier, so the planner walks it in reverse with the
anchor spliced in at its own depth and takes the first candidate that
prefix-matches.

A ring restore uploads the entry back into the device checkpoint slot and then
follows the ordinary `RestoreTurnCheckpoint` path: the KV is truncated to the
entry's frontier, the GDN state is rewound to the copy, and the suffix is
re-prefilled. Generation after a ring restore is identical to a cold prefill of
the same prompt; the end-to-end test verifies greedy-exact output on the real
artifact.

## Observability

`GET /slots` lists each retained slot's checkpoints, oldest first:

```json
"checkpoints": [
  {"frontier": 12480, "session_digest": "a1b2c3d4e5f60718"},
  {"frontier": 30976, "session_digest": "18f7e6d5c4b3a291"}
]
```

`frontier` is the ledger depth the entry rewinds to. `session_digest` is the
FNV-1a 64 hash of the token ledger up to that frontier, in the same encoding as
the slot's `session_digest`. Two slots that report the same checkpoint digest hold
an identical history up to that point.

## Persistence and eviction

Slot snapshots (`--slot-save-path`) carry the ring. The snapshot format is version
6, which records media preprocessing identity, the LoRA adapter's content fingerprint and the ring unconditionally
- an empty ring is a zero count, not an absent section - so one reader shape covers
every image. Earlier versions are rejected. A restore into a server with a smaller
ring keeps the newest entries that fit.

The ring lives and dies with slot residency. Anything that evicts the resident
session also discards its ring:

- admission of a new session when every slot is retained (the cheapest slot is
  evicted: an empty one first, then the shallowest),
- `POST /slots/{id}?action=erase` or a restore over the slot,
- a server restart.

The snapshot path is the tolerance mechanism for slot churn. When clients rotate
more sessions than `--max-concurrency` slots, save each session to disk before it
loses its slot; the ring returns with the restore. A session evicted without a
snapshot starts cold and rebuilds its ring one turn per request.

`--auto-save-evicted` closes the remaining gap: before an involuntary eviction
destroys a retained session, the server spills it (ring included) back to the
slot file it was last saved to or restored from. Sessions that never touched a
slot file are not covered, and an explicit `erase` never auto-saves. The device
snapshot runs on the eviction path and costs a few hundred milliseconds; the
file write runs on a background thread. Requires `--slot-save-path`.

## Limits

- The DFlash backend is not supported. Its cyclic local cache mirrors only the
  resident checkpoint and cannot rebuild older entries.
- Checkpoints exist at turn boundaries only. An edit inside the first turn, or
  before the oldest retained entry, still takes a full re-prefill.
- A ring restore is not free: the suffix between the checkpoint and the new
  frontier is re-prefilled, and the 147 MiB upload from pageable memory costs
  roughly 15 ms before the prefill starts.
- Tool or system-prompt edits diverge near token zero, below every checkpoint.
  The ring cannot help there; keep volatile content out of the prompt prefix
  instead.
