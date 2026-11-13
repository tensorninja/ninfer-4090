#!/usr/bin/env bash
# Downloads every file in the asset list into $NINFER_HOME and checks its size and SHA-256.
# A file that already passed the check is skipped by its size, mtime, and inode. A file that no
# longer matches its pin, such as the model of an older image, is downloaded again.
set -euo pipefail

home="${NINFER_HOME:-/var/lib/ninfer}"
assets="${NINFER_ASSETS:-/opt/ninfer/assets.txt}"
stamps="$home/.verified"

mkdir -p -- "$stamps"
exec 9>"$home/.fetch.lock"
flock 9

auth=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  auth=(--header "Authorization: Bearer $HF_TOKEN")
fi

identity() {
  printf '%s %s\n' "$1" "$(stat -c '%s %.9Y %i' -- "$2")"
}

matches() {
  [[ "$(stat -c %s -- "$2")" == "$3" ]] && sha256sum --quiet --check --status <<<"$1  $2"
}

while read -r sha size path url; do
  [[ -z "$sha" || "$sha" == \#* ]] && continue
  dest="$home/$path"
  stamp="$stamps/${path//\//_}"

  if [[ -f "$dest" ]]; then
    if [[ -f "$stamp" && "$(<"$stamp")" == "$(identity "$sha" "$dest")" ]]; then
      continue
    fi
    echo "ninfer-fetch: checking $path"
    if matches "$sha" "$dest" "$size"; then
      identity "$sha" "$dest" >"$stamp"
      continue
    fi
    echo "ninfer-fetch: $path does not match SHA-256 $sha, downloading it again"
    rm -f -- "$dest" "$stamp"
  fi

  mkdir -p -- "$(dirname -- "$dest")"
  partial="$dest.partial"
  if [[ ! -f "$partial" || "$(stat -c %s -- "$partial")" -lt "$size" ]]; then
    echo "ninfer-fetch: downloading $path ($((size / 1048576)) MiB) from $url"
    curl --fail --location --retry 10 --retry-all-errors --continue-at - "${auth[@]}" \
      --output "$partial" "$url"
  fi
  echo "ninfer-fetch: checking $path"
  if ! matches "$sha" "$partial" "$size"; then
    rm -f -- "$partial"
    echo "ninfer-fetch: $url does not match SHA-256 $sha and size $size" >&2
    exit 1
  fi
  mv -- "$partial" "$dest"
  identity "$sha" "$dest" >"$stamp"
done <"$assets"
