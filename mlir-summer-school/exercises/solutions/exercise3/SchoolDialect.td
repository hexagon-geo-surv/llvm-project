//===- SchoolDialect.td - School dialect definition --------*- tablegen -*-===//
//
// Reference solution for Exercise 3 (checkpoint 3): the dialect declares a
// constant materializer so that Attribute results returned from fold methods
// can be turned back into constant ops.
//
//===----------------------------------------------------------------------===//

#ifndef SCHOOL_DIALECT
#define SCHOOL_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// School dialect definition.
//===----------------------------------------------------------------------===//

def School_Dialect : Dialect {
  let name = "school";
  let summary = "The MLIR summer school exercise dialect.";
  let description = [{
    A tiny dialect with two integer operations, `school.max` and `school.mac`.
    It exists so that the exercises have something to rewrite, lower, and
    canonicalize.
  }];
  let cppNamespace = "::mlir::school";

  // Exercise 3, checkpoint 3: declares
  //   Operation *materializeConstant(OpBuilder &, Attribute, Type, Location);
  // on the dialect class (defined in SchoolDialect.cpp). Without it, folds
  // that return an Attribute are silently dropped.
  let hasConstantMaterializer = 1;

  // The materializer creates arith.constant ops, so arith must be loaded
  // whenever the school dialect is.
  let dependentDialects = ["::mlir::arith::ArithDialect"];
}

//===----------------------------------------------------------------------===//
// Base school operation definition.
//===----------------------------------------------------------------------===//

class School_Op<string mnemonic, list<Trait> traits = []> :
    Op<School_Dialect, mnemonic, traits>;

#endif // SCHOOL_DIALECT
