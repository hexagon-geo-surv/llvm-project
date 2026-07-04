//===- SchoolDialect.td - School dialect definition --------*- tablegen -*-===//
//
// The `school` dialect: two toy integer ops used throughout the MLIR summer
// school "Transformations" module. The dialect is deliberately bare-bones in
// the starter state -- making it a well-behaved citizen is Exercise 3.
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

  // TODO(exercise 3.3): Edit this dialect definition so that constants
  // produced by folds can be materialized. See exercise3.md, checkpoint 3.
}

//===----------------------------------------------------------------------===//
// Base school operation definition.
//===----------------------------------------------------------------------===//

class School_Op<string mnemonic, list<Trait> traits = []> :
    Op<School_Dialect, mnemonic, traits>;

#endif // SCHOOL_DIALECT
