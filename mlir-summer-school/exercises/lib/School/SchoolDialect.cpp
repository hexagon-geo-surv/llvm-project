//===- SchoolDialect.cpp - School dialect -----------------------*- C++ -*-===//
//
// Dialect initialization: registers the ops with the dialect.
//
//===----------------------------------------------------------------------===//

#include "School/SchoolDialect.h"
#include "School/SchoolOps.h"

using namespace mlir;
using namespace mlir::school;

#include "School/SchoolOpsDialect.cpp.inc"

void SchoolDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "School/SchoolOps.cpp.inc"
      >();
}

// TODO(exercise 3.3): Define SchoolDialect::materializeConstant here (after
// declaring the hook in SchoolDialect.td). See exercise3.md, checkpoint 3.
