//===- SchoolOps.cpp - School dialect ops -----------------------*- C++ -*-===//
//
// Out-of-line definitions for the school ops. In the starter state there is
// nothing here besides the generated op definitions -- Exercise 3 adds a
// folder and (as a stretch goal) a canonicalization pattern.
//
//===----------------------------------------------------------------------===//

#include "School/SchoolOps.h"
#include "School/SchoolDialect.h"

// The generated op definitions (SchoolOps.cpp.inc) need the full builder
// types, not just the forward declarations the headers use.
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;
using namespace mlir::school;

// TODO(exercise 3.2): Define MaxOp::fold here (after declaring the hook in
// SchoolOps.td). See exercise3.md, checkpoint 2.

// TODO(exercise 3.5, stretch): Define MaxOp::getCanonicalizationPatterns here
// (after declaring the hook in SchoolOps.td). See exercise3.md.

#define GET_OP_CLASSES
#include "School/SchoolOps.cpp.inc"
