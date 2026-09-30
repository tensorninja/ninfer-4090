# ninfer-web

Observability dashboard for a running `ninfer-serve` instance and both systems it serves, chat
generation and System One decisions. It shows throughput, scheduler occupancy and lane work, TTFT
decomposition, decision latency and state reuse, the shared adapter bank, continuation-cache
behaviour, VRAM budget, board telemetry, and per-request and per-decision detail, either live or
replayed from a request log. A second view, `/playground`, builds and runs System One requests
against the same server and shows each answer's distribution beside what the engine did for that
decision.

`docs/dashboard.md` describes what each panel means and how to read it. This file covers building
and developing the app itself.

## Build and serve

The repository Dockerfile builds this in a dedicated `web` stage and copies `dist/` into the
runtime image at `/opt/ninfer/web`, which the default command serves. The steps below are for
building outside Docker.

```bash
bun install
bun run build
```

Then point the engine at the build output. The dashboard is served same-origin, so no CORS
configuration is involved:

```bash
./build-sm89/apps/ninfer-serve \
  --model models/qwen3_8_27b.ninfer \
  --web-dir apps/web/dist
```

The dashboard is then at the server's root and the playground at `/playground`. Neither works on a
server started with `--api-key`, which gates the static files too. `--request-log-jsonl` is
optional for live use — the same records reach the browser over `/events` either way — but it is
what produces a file the dashboard can replay later.

## Development

```bash
NINFER_BASE_URL=http://127.0.0.1:8080 bun run dev
```

Vite serves on `127.0.0.1:5180` and proxies `/telemetry`, `/events`, `/metrics`, `/slots`,
`/health`, `/v1`, and `/typesafe` to the engine, so the development build sees the same same-origin
layout as production. `NINFER_BASE_URL` is validated rather than defaulted silently: a malformed or
path-bearing origin fails the config instead of quietly pointing at the wrong engine. Proxy
timeouts are disabled because `/events` is an open-ended stream that a long generation can outlast.

| Script                 | Purpose                                        |
| ---------------------- | ---------------------------------------------- |
| `bun run dev`          | Vite dev server with the engine proxy          |
| `bun run build`        | Typecheck, then production bundle into `dist/` |
| `bun run typecheck`    | `tsc -b` only                                  |
| `bun test`             | Derivation and playground tests                |
| `bun run format`       | Prettier write                                 |
| `bun run format:check` | Prettier check, for CI or a pre-commit hook    |

## Where the data comes from

Two engine endpoints, added for this dashboard:

- `GET /telemetry` — a complete snapshot, not a delta: NVML board sensors, scheduler gauges and
  the execution thread's wall-clock split, the `MemorySummary` VRAM budget including the resident
  LoRA bank, per-slot state with each lane's work kind and adapter, the adapter pool with each
  adapter's kind, which adapter each bank slot holds, swap and slot-wait counters, the System One
  alias binding, and cache occupancy paired with the configured capacities. Polled at 1 Hz.
  Because it is complete, the client resynchronizes by simply fetching it again.
- `GET /events` — Server-Sent Events carrying the same schema-22 records that
  `--request-log-jsonl` writes, byte for byte, including the `decision_*` records. A late
  subscriber receives the retained `server_start` plus a bounded backlog, so a browser opened
  mid-run still knows the engine configuration.

The stream is bounded and lossy by design. A subscriber that cannot keep up drops its oldest
records and reports the count rather than applying backpressure to the execution thread; the 1 Hz
poll remains authoritative for current state.

## Replay

Loading a `--request-log-jsonl` file re-derives every record-backed panel offline, System One
included. Board telemetry, live slot occupancy, and bank residency are sampled and never written to
the log, so those panels say so instead of rendering zeros — a reading that was never taken is not
shown as a measurement.

## Layout

| Path                          | Contents                                                              |
| ----------------------------- | --------------------------------------------------------------------- |
| `src/lib/records.ts`          | Schema-22 record types and a tolerant line parser                     |
| `src/lib/telemetry.ts`        | `/telemetry` payload types, cache view from records, Prometheus parse |
| `src/lib/derive.ts`           | Request and decision analytics, adapter pool, cache partitioning      |
| `src/lib/engine-client.ts`    | Poll + stream + file replay, and the state they produce               |
| `src/lib/glossary.ts`         | Definitions behind every tooltip                                      |
| `src/lib/palette.ts`          | Chart colours, mirroring the CSS custom properties                    |
| `src/lib/route.ts`            | Path routes over `pushState`; the hash is left for share links        |
| `src/components/echart.tsx`   | ECharts registration, shared theme, React binding                     |
| `src/components/charts.tsx`   | Band, sample, line, and stacked-bar primitives                        |
| `src/components/tooltip.tsx`  | Portal-positioned tooltips                                            |
| `src/components/topbar.tsx`   | Engine identity, connection, replay, and the view switch              |
| `src/components/decision.tsx` | A decision's phase split, shared by System One panel and playground   |
| `src/views/Dashboard.tsx`     | The dashboard's panel grid                                            |
| `src/panels/`                 | One file per panel                                                    |
| `src/playground/`             | System One playground: request model, checks, answers, engine facts   |

Records are discriminated on `event`, never on `schema_version`, so a log written by an older
schema still replays: fields a later schema added are declared optional and read as absent.

## Conventions worth preserving

**Interval averages and snapshots are drawn differently.** A throughput sample is an average over
its whole `interval_seconds`, so it is drawn as a band filling that interval. The scheduler's
running and waiting counts are read at the instant the report is emitted, so they are drawn as
points. Using a band for them would assert an occupancy that was never measured.

**Reporting intervals are irregular.** The engine suppresses intervals with no activity and folds
the skipped time into the next sample, so series are plotted against real wall-clock time. Plotting
against sample index would compress an idle gap into a straight line.

**`derive.ts` has an oracle.** Its aggregations are a port of the maintainer `cache_health.py`
script: same nearest-rank percentile rule, same denominators, same partitioning of misses and
restore failures. `derive.test.ts` pins the agreement so the dashboard and the script cannot report
different numbers for the same log.

**Terminology lives in `glossary.ts`.** Panels reference entries by key rather than inlining prose,
so a reading and its explanation cannot drift apart.

**The palette is duplicated deliberately.** ECharts renders to canvas and cannot resolve `var()`,
so `palette.ts` mirrors the CSS custom properties in `styles/foundation.css`. Changing a colour
means changing both.
