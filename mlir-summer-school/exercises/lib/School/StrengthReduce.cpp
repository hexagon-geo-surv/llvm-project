//===- StrengthReduce.cpp - Exercise 1: hand-written IR surgery -*- C++ -*-===//
//
// Exercise 1: rewrite `arith.muli %x, %c` (with %c a constant power of two)
// into `arith.shli %x, %log2(c)` -- by hand, with a walk, a builder, and
// explicit use replacement. No pattern framework yet; that is Exercise 2.
//
// The pass is registered under the flag -school-strength-reduce and is
// anchored on func.func. Run it with:
//
//   school-opt in.mlir -pass-pipeline="builtin.module(func.func(school-strength-reduce))"
//
//===----------------------------------------------------------------------===//

#include "School/SchoolPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::school {
#define GEN_PASS_DEF_SCHOOLSTRENGTHREDUCE
#include "School/SchoolPasses.h.inc"

namespace {
struct SchoolStrengthReduce
    : public impl::SchoolStrengthReduceBase<SchoolStrengthReduce> {
  void runOnOperation() override {
    // `getOperation()` is the func.func this pass instance runs on.
    //
    // TODO(exercise 1, step 1): Find the candidates.
    //   Walk the function with a typed callback
    //   (`getOperation()->walk([&](arith::MulIOp op) { ... })`) and collect
    //   every muli whose right-hand side is a constant power of two into a
    //   `SmallVector<arith::MulIOp>`.
    //   Useful APIs: matchPattern + m_ConstantInt (mlir/IR/Matchers.h),
    //   APInt::isPowerOf2.
    //   For checkpoint 1, just print each candidate: `llvm::errs() << op
    //   << "\n";` (stderr, so the FileCheck test on stdout keeps working).
    //
    // TODO(exercise 1, step 2): Build the replacement ops.
    //   For each candidate, create an `arith::ConstantOp` holding log2(c)
    //   (APInt::logBase2) and an `arith::ShLIOp`, inserted right before the
    //   muli (`OpBuilder b(op);`).
    //   Remember: ops are created with `OpTy::create(b, loc, ...)`, and new
    //   ops should reuse the location of the op they replace (op.getLoc()).
    //
    // TODO(exercise 1, step 3): Replace and erase.
    //   Redirect all uses of the muli to the new shli, then erase the muli
    //   -- in that order (Operation::replaceAllUsesWith, Operation::erase).
    //   Count each rewrite in the `numRewrites` pass statistic.
  }
};
} // namespace
} // namespace mlir::school
