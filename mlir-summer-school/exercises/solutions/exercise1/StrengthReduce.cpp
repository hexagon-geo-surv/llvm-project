//===- StrengthReduce.cpp - Exercise 1: hand-written IR surgery -*- C++ -*-===//
//
// Reference solution for Exercise 1.
//
// Rewrites `arith.muli %x, %c` (with %c a constant power of two) into
// `arith.shli %x, %log2(c)` using a hand-written walk. Compare with the
// pattern-based version in Exercise 2A: this file does by hand what the
// greedy driver will do for us there (traversal, revisiting, cleanup).
//
//===----------------------------------------------------------------------===//

#include "School/SchoolPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"

namespace mlir::school {
#define GEN_PASS_DEF_SCHOOLSTRENGTHREDUCE
#include "School/SchoolPasses.h.inc"

namespace {
struct SchoolStrengthReduce
    : public impl::SchoolStrengthReduceBase<SchoolStrengthReduce> {
  void runOnOperation() override {
    // Step 1: collect the candidates. We do not mutate while walking:
    // erasing ops the walk has not visited yet invalidates the traversal.
    // Collect-then-mutate is the robust idiom.
    SmallVector<arith::MulIOp> worklist;
    getOperation()->walk([&](arith::MulIOp op) { worklist.push_back(op); });

    for (arith::MulIOp op : worklist) {
      // Match: the rhs must be a constant power of two. matchPattern
      // handles the "is it defined by a constant?" dance and binds the
      // value; block arguments and non-constant values simply fail to
      // match (no crash on `muli %x, %y`).
      APInt rhsValue;
      if (!matchPattern(op.getRhs(), m_ConstantInt(&rhsValue)) ||
          !rhsValue.isPowerOf2())
        continue;

      // Step 2: build the replacement right before the muli. Reuse the
      // muli's location so diagnostics and debug info survive the rewrite.
      OpBuilder builder(op); // Insertion point: directly before `op`.
      Value shiftAmount = arith::ConstantOp::create(
          builder, op.getLoc(),
          builder.getIntegerAttr(op.getType(), rhsValue.logBase2()));
      Value shifted = arith::ShLIOp::create(builder, op.getLoc(), op.getLhs(),
                                            shiftAmount);

      // Step 3: reroute all uses to the new value, then erase the muli.
      // The order matters: erasing an op that still has uses is a fatal
      // error (in assert builds; memory corruption otherwise).
      op->replaceAllUsesWith(ValueRange{shifted});
      op->erase();
      ++numRewrites; // Pass statistic, see --mlir-pass-statistics.

      // Note: the original constant (e.g. %c8) may now be dead. We leave
      // it behind on purpose -- Session 3 shows which standard passes
      // clean this up (and why passes should not duplicate that work).
    }
  }
};
} // namespace
} // namespace mlir::school
