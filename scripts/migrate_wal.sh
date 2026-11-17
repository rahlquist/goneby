#!/usr/bin/env bash
# migrate_wal.sh — move GoneBy runtime data to the per-profile store.
#
# v1.1.0 pointed every frontend at the shared ~/.goneby. The Hermes catalog
# review (PR #155972) reverted the plugin to <hermes-home>/plugin-data/goneby
# — profiles are kept apart on purpose — so a v1.1.0 store needs carrying
# across. This script copies (never deletes) the old store, so both locations
# stay readable and the move is reversible.
#
# DRY-RUN IS THE DEFAULT. Review the plan, then run with --apply.
# Stop the Hermes gateway first: an append landing mid-copy would be lost.
#
# Usage:
#   scripts/migrate_wal.sh [OLD_DIR]                 # show what would happen
#   scripts/migrate_wal.sh --apply [OLD_DIR]         # copy + checksum verify
set -euo pipefail

APPLY=0
if [ "${1:-}" = "--apply" ]; then
  APPLY=1
  shift
fi

OLD_DIR="${1:-${GONEBY_DATA_DIR:-$HOME/.goneby}}"
NEW_DIR="${HERMES_HOME:-$HOME/.hermes}/plugin-data/goneby"

say() { printf '%s\n' "$*"; }
[ "$APPLY" = 1 ] || say "DRY-RUN (nothing copied; rerun with --apply)"

if [ ! -d "$OLD_DIR" ]; then
  say "old dir $OLD_DIR does not exist — nothing to do"
  exit 0
fi
if [ "$OLD_DIR" = "$NEW_DIR" ]; then
  say "old dir == new dir — nothing to do"
  exit 0
fi

sha_of_tree() {
  # deterministic checksum over relative paths + contents
  ( cd "$1" && find . -type f | LC_ALL=C sort | xargs -r sha256sum | sha256sum | cut -d' ' -f1 )
}

copied=0
shopt -s nullglob dotglob
items=("$OLD_DIR"/*)
if [ ${#items[@]} -eq 0 ]; then
  say "old dir $OLD_DIR is empty — nothing to do"
  exit 0
fi

for src in "${items[@]}"; do
  rel="${src#"$OLD_DIR"/}"
  target="$NEW_DIR/$rel"
  if [ -e "$target" ]; then
    say "SKIP  $rel — target already exists (no overwrite; resolve by hand)"
    continue
  fi
  if [ "$APPLY" = 1 ]; then
    mkdir -p "$(dirname "$target")"
    cp -a "$src" "$target"
    if [ -f "$target" ]; then
      a=$(sha256sum "$src" | cut -d' ' -f1)
      b=$(sha256sum "$target" | cut -d' ' -f1)
    else
      a=$(sha_of_tree "$src")
      b=$(sha_of_tree "$target")
    fi
    [ "$a" = "$b" ] || { say "FAIL checksum mismatch on $rel"; exit 1; }
  fi
  say "COPY  $rel"
  copied=$((copied + 1))
done

if [ "$APPLY" = 1 ]; then
  say "verified by sha256; originals left in place at $OLD_DIR"
  say "post-check (gateway stopped): GONEBY_WAL_PATH=\"$NEW_DIR/wal\" goneby read per project"
else
  say "would copy $copied item(s) $OLD_DIR -> $NEW_DIR"
  say "rerun with --apply AFTER stopping the Hermes gateway"
fi
