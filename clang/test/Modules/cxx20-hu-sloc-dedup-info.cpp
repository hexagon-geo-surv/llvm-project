// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header %t/hu.h \
// RUN:   -I%t -o %t/hu.pcm
// RUN: llvm-bcanalyzer --dump --disable-histogram %t/hu.pcm \
// RUN:   | FileCheck %s --check-prefix=SLOC-INFO

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/M.cppm \
// RUN:   -o %t/M.pcm
// RUN: llvm-bcanalyzer --dump --disable-histogram %t/M.pcm \
// RUN:   | FileCheck %s --check-prefix=SLOC-INFO

// SLOC-INFO: <SLOC_ENTRY_DEDUP_INFO
// SLOC-INFO-NEXT: <SOURCE_LOCATION_OFFSETS

//--- shared.h
#define SHARED_VALUE 1
int shared;

//--- hu.h
#include "shared.h"
int hu;

//--- M.cppm
export module M;
export int f();
