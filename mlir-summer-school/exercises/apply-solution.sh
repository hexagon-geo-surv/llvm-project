#!/usr/bin/env bash
# apply-solution.sh -- copy reference solutions over the starter files.
#
# Usage:
#   ./apply-solution.sh 1        # solution for exercise 1
#   ./apply-solution.sh 2        # solutions for exercise 2 (parts A and B)
#   ./apply-solution.sh 3        # solution for exercise 3 (incl. stretch goals)
#   ./apply-solution.sh all      # all of the above
#   ./apply-solution.sh reset    # restore the starter state
#
# Applying a solution is idempotent (it just copies files); rebuild with
# `ninja -C build` afterwards. Exercise 3 touches .td files, so that rebuild
# re-runs TableGen -- this is expected.
#
# Before the first copy, the pristine starter files are snapshotted to
# .starter-backup/ so that `reset` works even when the project does not live
# inside a git checkout.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

STARTER_FILES=(
  lib/School/StrengthReduce.cpp
  lib/School/Peephole.cpp
  lib/School/ConvertSchoolToArith.cpp
  lib/School/SchoolDialect.cpp
  lib/School/SchoolOps.cpp
  include/School/SchoolDialect.td
  include/School/SchoolOps.td
)

backup_starter() {
  # One-time snapshot of the pristine starter, taken before the first copy.
  [[ -d .starter-backup ]] && return
  mkdir -p .starter-backup/lib/School .starter-backup/include/School
  for f in "${STARTER_FILES[@]}"; do
    cp "$f" ".starter-backup/$f"
  done
  echo "Backed up pristine starter files to .starter-backup/."
}

copy() {
  echo "  $1 -> $2"
  cp "$1" "$2"
}

apply1() {
  echo "Applying solution for exercise 1:"
  copy solutions/exercise1/StrengthReduce.cpp lib/School/StrengthReduce.cpp
}

apply2() {
  echo "Applying solutions for exercise 2:"
  copy solutions/exercise2/Peephole.cpp lib/School/Peephole.cpp
  copy solutions/exercise2/ConvertSchoolToArith.cpp \
       lib/School/ConvertSchoolToArith.cpp
}

apply3() {
  echo "Applying solution for exercise 3:"
  copy solutions/exercise3/SchoolDialect.td include/School/SchoolDialect.td
  copy solutions/exercise3/SchoolOps.td include/School/SchoolOps.td
  copy solutions/exercise3/SchoolDialect.cpp lib/School/SchoolDialect.cpp
  copy solutions/exercise3/SchoolOps.cpp lib/School/SchoolOps.cpp
}

case "${1:-}" in
  1) backup_starter; apply1 ;;
  2) backup_starter; apply2 ;;
  3) backup_starter; apply3 ;;
  all) backup_starter; apply1; apply2; apply3 ;;
  reset)
    # Only restores files under THIS directory's include/ and lib/ trees.
    if [[ -d .starter-backup ]]; then
      for f in "${STARTER_FILES[@]}"; do
        copy ".starter-backup/$f" "$f"
      done
      echo "Restored starter state from .starter-backup/."
    elif git ls-files --error-unmatch include/School/SchoolOps.td \
        > /dev/null 2>&1; then
      git checkout -- include/School lib/School
      echo "Restored starter state of include/School/ and lib/School/ from git."
    else
      echo "error: no .starter-backup/ found and the starter files are not" >&2
      echo "       tracked by git -- cannot reset. Re-download the starter." >&2
      exit 1
    fi
    ;;
  *)
    echo "usage: $0 {1|2|3|all|reset}" >&2
    exit 1
    ;;
esac
echo "Done. Rebuild with: ninja -C build"
