//===- SchoolDialect.cpp - School dialect -----------------------*- C++ -*-===//
//
// Reference solution for Exercise 3 (checkpoint 3): dialect initialization
// plus the constant materializer that turns Attribute fold results back into
// constant ops.
//
//===----------------------------------------------------------------------===//

#include "School/SchoolDialect.h"
#include "School/SchoolOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using namespace mlir::school;

#include "School/SchoolOpsDialect.cpp.inc"

void SchoolDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "School/SchoolOps.cpp.inc"
      >();
}

/// When a fold returns an Attribute, the folding infrastructure asks the
/// *dialect of the folded op* to turn that attribute into a constant op.
/// The school dialect has no constant op of its own, so we borrow
/// arith.constant (which is exactly what its `materialize` helper builds --
/// or returns null if the attribute/type combination is not supported,
/// which tells the caller to keep the original op).
Operation *SchoolDialect::materializeConstant(OpBuilder &builder,
                                              Attribute value, Type type,
                                              Location loc) {
  return arith::ConstantOp::materialize(builder, value, type, loc);
}
