#pragma once

#include "ninfer/engine.h"
#include "options.h"

namespace ninfer::cli {

// Answers the System One request bodies of `options.systemone_path` with `engine`, printing each
// response (or its {"detail": ...} error body) on its own stdout line and a summary per request on
// stderr. Returns the process exit status: nonzero when any request failed.
[[nodiscard]] int run_systemone(Engine& engine, const Options& options);

} // namespace ninfer::cli
