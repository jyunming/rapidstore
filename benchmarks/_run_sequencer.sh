#!/usr/bin/env bash
# Waits for rerank-only run (PID $1) to finish, then:
#   1. Runs dbpedia full benchmark
#   2. Runs _verify_and_fill.py to catch any missing configs
RERANK_PID=$1
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG="$SCRIPT_DIR/_bench_sequencer.log"

echo "[sequencer] Waiting for rerank-only PID $RERANK_PID to finish..." | tee "$LOG"

while kill -0 "$RERANK_PID" 2>/dev/null; do
    sleep 30
done

echo "[sequencer] Rerank-only done. Starting dbpedia full run..." | tee -a "$LOG"

cd "$REPO_ROOT" || { echo "[sequencer] Failed to cd to $REPO_ROOT" | tee -a "$LOG"; exit 1; }
PYTHONUNBUFFERED=1 python benchmarks/full_config_bench.py --full --datasets dbpedia1536 dbpedia3072 \
    > benchmarks/_dbpedia_full_run.log 2>&1
EXIT1=$?
echo "[sequencer] dbpedia full run finished (exit $EXIT1)." | tee -a "$LOG"

echo "[sequencer] Running completeness verification..." | tee -a "$LOG"
PYTHONUNBUFFERED=1 python benchmarks/_verify_and_fill.py \
    > benchmarks/_verify_fill.log 2>&1
EXIT2=$?
echo "[sequencer] Verification finished (exit $EXIT2)." | tee -a "$LOG"
cat benchmarks/_verify_fill.log | tee -a "$LOG"

echo "[sequencer] All done." | tee -a "$LOG"
