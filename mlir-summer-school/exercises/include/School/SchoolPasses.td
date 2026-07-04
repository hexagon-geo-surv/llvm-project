//===- SchoolPasses.td - School exercise passes ------------*- tablegen -*-===//
//
// Declarative definitions of the three exercise passes. TableGen generates
// the pass boilerplate (base class, command-line flag, registration,
// statistics) from these entries; the C++ files under lib/School/ only
// implement runOnOperation().
//
//===----------------------------------------------------------------------===//

#ifndef SCHOOL_PASSES
#define SCHOOL_PASSES

include "mlir/Pass/PassBase.td"

// Exercise 1: a hand-written walk, anchored on func.func. The pass manager
// runs it on every function in the module (in parallel).
def SchoolStrengthReduce : Pass<"school-strength-reduce", "::mlir::func::FuncOp"> {
  let summary = "Rewrite muli-by-power-of-two into shli (hand-written walk)";
  let description = [{
    Rewrites `arith.muli %x, %c` where `%c` is a constant power of two into
    `arith.shli %x, %log2(c)`. Implemented as a hand-written IR walk --
    Exercise 1 of the summer school "Transformations" module.
  }];
  let dependentDialects = ["::mlir::arith::ArithDialect"];
  let statistics = [
    Statistic<"numRewrites", "num-rewrites",
              "Number of muli ops rewritten to shli">
  ];
}

// Exercise 2A: the same rewrite (plus a shift-merging one) expressed as
// rewrite patterns, run by the greedy driver.
def SchoolPeephole : Pass<"school-peephole", "::mlir::func::FuncOp"> {
  let summary = "Pattern-based peephole optimizations (greedy driver)";
  let description = [{
    Runs two `OpRewritePattern`s to a fixpoint with `applyPatternsGreedily`:
    `MulByPow2ToShl` (muli-by-power-of-two -> shli) and `MergeConsecutiveShl`
    (shli-of-shli -> one shli). Exercise 2, part A.
  }];
  let dependentDialects = ["::mlir::arith::ArithDialect"];
}

// Exercise 2B: a dialect conversion. Unanchored (an any-op pass), so it can
// run on the module or on individual functions.
def ConvertSchoolToArith : Pass<"convert-school-to-arith"> {
  let summary = "Lower the school dialect to arith (dialect conversion)";
  let description = [{
    Lowers `school.max` to `arith.cmpi sgt` + `arith.select` and `school.mac`
    to `arith.muli` + `arith.addi` using the dialect conversion framework
    (`ConversionTarget` + `applyPartialConversion`). Exercise 2, part B.
  }];
  let dependentDialects = ["::mlir::arith::ArithDialect"];
}

#endif // SCHOOL_PASSES
