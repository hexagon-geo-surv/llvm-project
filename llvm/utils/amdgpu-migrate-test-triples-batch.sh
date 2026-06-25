#!/bin/bash
#
# Drive a single batch of the AMDGPU test-triple migration.
#
# Given a file listing test paths (one per line, relative to the repo root),
# this:
#   1. rewrites their RUN lines with amdgpu-migrate-test-triples.py,
#   2. runs llvm-lit on the files that actually changed,
#   3. reverts (git checkout) any file that fails lit, so the working tree is
#      left with only passing, migrated tests staged for review.
#
# The codegen produced by the folded triple is byte-identical to the old
# `amdgcn + -mcpu` form except where the triple string itself is emitted, so
# lit is the authoritative safety net: anything that does not pass is backed
# out automatically and can be handled by hand later.
#
# Usage:
#   amdgpu-migrate-test-triples-batch.sh <list-file> [build-dir]
#
# build-dir defaults to build_rel_with_debinfo. It must contain
# bin/llvm-lit built with the AMDGPU target.
#
# Environment is assumed to be the llvm-project repo root.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REWRITE="$HERE/amdgpu-migrate-test-triples.py"

LIST="${1:?usage: $0 <list-file> [build-dir]}"
BUILD="${2:-build_rel_with_debinfo}"
LIT="$BUILD/bin/llvm-lit"

if [ ! -x "$LIT" ]; then
  echo "error: $LIT not found or not executable" >&2
  exit 2
fi

NCAND=$(grep -c . "$LIST")
echo "batch: $NCAND candidate files"

# 1. Rewrite. The script prints each file it changed.
xargs "$REWRITE" < "$LIST" > /tmp/amdgpu_batch_changed.txt
NCHANGED=$(grep -c . /tmp/amdgpu_batch_changed.txt)
echo "rewritten: $NCHANGED files"
if [ "$NCHANGED" -eq 0 ]; then
  echo "nothing to do"
  exit 0
fi

# 2. Run lit on the changed files only.
xargs "$LIT" -q < /tmp/amdgpu_batch_changed.txt > /tmp/amdgpu_batch_lit.txt 2>&1
echo "lit summary:"
grep -E 'Total Discovered|Passed|Failed|Unsupported|Unresolved' /tmp/amdgpu_batch_lit.txt

# 3. Parse failures.  Lit prints e.g.  "  LLVM :: CodeGen/AMDGPU/add.ll"
#    under "Failed Tests".  Map "LLVM :: <suffix>" back to "llvm/test/<suffix>".
#    (Use [[:space:]] / explicit patterns for BSD sed compatibility.)
grep -E '^[[:space:]]+LLVM :: ' /tmp/amdgpu_batch_lit.txt \
  | sed -E 's#^[[:space:]]+LLVM :: #llvm/test/#; s/ \(.*\)$//' \
  | sort -u > /tmp/amdgpu_batch_failed.txt

NFAIL=$(wc -l < /tmp/amdgpu_batch_failed.txt | tr -d ' ')
echo "failed: $NFAIL files (reverting)"
if [ "$NFAIL" -gt 0 ]; then
  cat /tmp/amdgpu_batch_failed.txt
  # Revert just the failing files.
  xargs git checkout -- < /tmp/amdgpu_batch_failed.txt
fi

NKEPT=$(( NCHANGED - NFAIL ))
echo "kept: $NKEPT migrated files staged in working tree"
