# MLIR Summer School — Exercise Project

The hands-on companion to the *Transformations* module (3 sessions). One
self-contained out-of-tree MLIR project, used in all three sessions:

- a tiny **`school` dialect** (C++ namespace `mlir::school`) with two ops:
  - `school.max` — signed maximum of two `i32`
  - `school.mac` — multiply-accumulate, `a*b+c`, three `i32`

  The dialect is **deliberately imperfect** in the starter state: no `Pure`
  trait, no folder, no canonicalization patterns, no constant materializer.
  Fixing that *is* Exercise 3.
- **`school-opt`** — an `mlir-opt`-style tool with the school dialect, the
  upstream transform passes (`-canonicalize`, `-cse`, ...) and the three
  exercise passes registered.
- **pass stubs** with `TODO(exercise N, step K)` markers. The starter always
  compiles; you fill in the bodies.
- **lit/FileCheck tests for every exercise part** — the task sheets map the
  tests to checkpoints — so you always know whether you are on track.

| Exercise | Pass / files you edit | Task sheet |
|---|---|---|
| 1 — Your First Pass | `lib/School/StrengthReduce.cpp` | [exercise1.md](exercise1.md) |
| 2A — Rewrite Patterns | `lib/School/Peephole.cpp` | [exercise2.md](exercise2.md) |
| 2B — Dialect Conversion | `lib/School/ConvertSchoolToArith.cpp` | [exercise2.md](exercise2.md) |
| 3 — The Free Lunch | `include/School/School{Dialect,Ops}.td`, `lib/School/School{Dialect,Ops}.cpp` | [exercise3.md](exercise3.md) |

## Prerequisites

- A **built or installed LLVM/MLIR** (the summer school provides one; any
  reasonably recent build works). You need the path to its
  `lib/cmake/mlir` directory.
- CMake ≥ 3.20, Ninja, a C++17 compiler.
- Python 3 (`llvm-lit` is a Python script; no extra packages needed).

## Building

```bash
cd mlir-summer-school/exercises
cmake -G Ninja -S . -B build \
  -DMLIR_DIR=<your-mlir-build-or-install>/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=Release
ninja -C build
```

Notes:

- Against an MLIR **build tree**, nothing else is needed (lit and FileCheck
  are found automatically). Against an **installed** MLIR, the install must
  include the LLVM utilities (`-DLLVM_INSTALL_UTILS=ON` when LLVM was built)
  and you must pass `-DLLVM_EXTERNAL_LIT=<path-to-llvm-lit>`.
- If the LLVM tree was built with a sanitizer, configure this project with
  the same sanitizer (see the appendix below).
- The first build takes a few minutes (it links against the MLIR static
  libraries); incremental rebuilds while you work on a pass take seconds.

## Running the tests

```bash
# Everything:
ninja -C build check-school

# One exercise (llvm-lit ships with the LLVM build; a stock Python 3 suffices):
<llvm-build>/bin/llvm-lit -v build/test/exercise1

# A single test file:
<llvm-build>/bin/llvm-lit -v build/test/exercise1/strength-reduce.mlir
```

In the starter state, `build/test/dialect` is green and all
`build/test/exercise*` tests are red — each exercise turns its directory
green. The exercise-3 stretch tests (`commutative.mlir`, `reassociate.mlir`)
stay red until the stretch goals are done.

You can also run `school-opt` by hand, e.g.:

```bash
build/bin/school-opt test/dialect/ops.mlir
build/bin/school-opt test/exercise1/strength-reduce.mlir \
  -pass-pipeline="builtin.module(func.func(school-strength-reduce))"
```

## Directory map

```
exercises/
├── CMakeLists.txt          # top-level build (out-of-tree, finds MLIR via MLIR_DIR)
├── include/School/
│   ├── SchoolDialect.td    # dialect definition            (edited in ex. 3)
│   ├── SchoolOps.td        # school.max / school.mac       (edited in ex. 3)
│   ├── SchoolPasses.td     # declarative pass definitions
│   └── *.h                 # thin headers over the generated code
├── lib/School/
│   ├── SchoolDialect.cpp   # dialect init                  (edited in ex. 3)
│   ├── SchoolOps.cpp       # op extras: folder etc.        (edited in ex. 3)
│   ├── StrengthReduce.cpp  # -school-strength-reduce       (edited in ex. 1)
│   ├── Peephole.cpp        # -school-peephole              (edited in ex. 2A)
│   └── ConvertSchoolToArith.cpp # -convert-school-to-arith (edited in ex. 2B)
├── school-opt/             # the opt tool (you don't edit this)
├── test/
│   ├── dialect/            # round-trip test, green from day one
│   ├── exercise1/ exercise2/ exercise3/   # per-checkpoint FileCheck tests
│   └── lit.cfg.py, lit.site.cfg.py.in    # lit wiring
├── solutions/              # reference solutions (see apply-solution.sh)
├── exercise{1,2,3}.md      # the task sheets
└── apply-solution.sh
```

## apply-solution.sh

Copies the reference solutions over the starter files:

```bash
./apply-solution.sh 1      # or 2, 3, all
ninja -C build && ninja -C build check-school

./apply-solution.sh reset  # back to the starter state (restores from the
                           # .starter-backup/ snapshot taken on first apply,
                           # falling back to git checkout in a git clone)
```

The solutions are exemplary implementations using current MLIR APIs
(`OpTy::create(builder, loc, ...)`, `applyPatternsGreedily`,
`notifyMatchFailure`, ...). Exercise 3's solution includes all stretch goals, so
after `apply-solution.sh all` every test in the suite is green.

## Appendix: setup on the summer-school preparation machine

The prebuilt tree on this machine is **Debug + assertions + AddressSanitizer**
and was compiled with clang. Out-of-tree projects linking it must match the
sanitizer and should match the compiler:

```bash
cmake -G Ninja -S . -B build \
  -DMLIR_DIR=/home/mspringer/llvm-project/build/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_USE_SANITIZER=Address \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
ninja -C build
ninja -C build check-school
# single exercise:
/home/mspringer/llvm-project/build/bin/llvm-lit -v build/test/exercise1
```

Without `-DLLVM_USE_SANITIZER=Address` the link fails with
`undefined reference to '__asan_report_load...'`. For the class itself, use a
plain Release+assertions LLVM build and drop the sanitizer/compiler flags.
