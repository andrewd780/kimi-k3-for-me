#!/usr/bin/env bash
# Feed a file as the prompt instead of a command-line string.
#
#   examples/03-long-prompt.sh ~/k3model ~/k3trunk somefile.txt
#
# Use --prompt-file rather than --prompt for anything non-ASCII: argv is re-encoded by
# the shell, so a UTF-8 string can reach the engine as different bytes than you typed.
#
# The prompt is read in passes of --prefill-chunk positions (default 4096, bit-identical to
# one pass), with progress and an estimate of the time to go after each. Every prompt
# position still runs the whole model, so a long file is slow: about 3.6 s per position on
# a 4-vCPU AVX-512 box. --kv-latent keeps the cache at 0.055 MB per position.
set -euo pipefail
MODEL="${1:?usage: 03-long-prompt.sh <model_dir> <trunk_dir> <file>}"
TRUNK="${2:?}"
FILE="${3:?}"

wc -c "$FILE"
# --checkpoint saves the read-in after every chunk: if the run is interrupted (or paused
# with Ctrl-C), the same command continues where it stopped.
./bin/k3 "$MODEL" --trunk "$TRUNK" --preset server \
    --tok "$MODEL" --prompt-file "$FILE" --gen 16 --incremental \
    --checkpoint "$FILE.k3state"
