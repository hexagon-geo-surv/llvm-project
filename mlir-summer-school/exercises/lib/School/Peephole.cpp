//===- Peephole.cpp - Exercise 2A: rewrite patterns -------------*- C++ -*-===//
//
// Exercise 2, part A: express rewrites as OpRewritePatterns and let the
// greedy driver (applyPatternsGreedily) orchestrate them. Two patterns:
//
//   MulByPow2ToShl:      muli(x, C)            -> shli(x, log2(C))
//   MergeConsecutiveShl: shli(shli(x, C1), C2) -> shli(x, C1 + C2)
//
// Neither pattern alone reduces `(x * 4) * 8` to a single shift -- together,
// run to a fixpoint, they do. That composition is the point of this exercise.
//
// Run with:
//
//   school-opt in.mlir -pass-pipeline="builtin.module(func.func(school-peephole))"
//
//===----------------------------------------------------------------------===//

#include "School/SchoolPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::school {
#define GEN_PASS_DEF_SCHOOLPEEPHOLE
#include "School/SchoolPasses.h.inc"

namespace {

/// muli(x, C) -> shli(x, log2(C)) when C is a constant power of two.
struct MulByPow2ToShl : public OpRewritePattern<arith::MulIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::MulIOp op,
                                PatternRewriter &rewriter) const override {
    // TODO(exercise 2A, step 1): Port your Exercise 1 rewrite to a pattern.
    //   - Match: rhs must be a constant power of two (matchPattern +
    //     m_ConstantInt, APInt::isPowerOf2). Return failure() *before*
    //     touching the IR -- the matchAndRewrite contract.
    //   - Rewrite: create the shift-amount constant with
    //     arith::ConstantOp::create(rewriter, ...), then replace the muli via
    //     rewriter.replaceOpWithNewOp<arith::ShLIOp>(...). Every mutation
    //     must go through `rewriter`; never call op->erase() yourself.
    return rewriter.notifyMatchFailure(op, "pattern not implemented yet");
  }
};

/// shli(shli(x, C1), C2) -> shli(x, C1 + C2) when both shift amounts are
/// constants and the combined amount still fits the type.
struct MergeConsecutiveShl : public OpRewritePattern<arith::ShLIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ShLIOp op,
                                PatternRewriter &rewriter) const override {
    // TODO(exercise 2A, step 2): Implement the merge.
    //   - Match: op's lhs must be produced by another arith.shli
    //     (Value::getDefiningOp<arith::ShLIOp>()), and both rhs operands
    //     must be constants (matchPattern + m_ConstantInt).
    //   - Guard: if C1 + C2 >= bit width of the type, do NOT merge --
    //     shifting by >= the bit width is poison. (Checkpoint 2's
    //     @no_merge_overflow tests this.)
    //   - Rewrite: build the combined constant, then
    //     rewriter.replaceOpWithNewOp<arith::ShLIOp>(op, innerX, newAmount).
    return rewriter.notifyMatchFailure(op, "pattern not implemented yet");
  }
};

struct SchoolPeephole : public impl::SchoolPeepholeBase<SchoolPeephole> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MulByPow2ToShl, MergeConsecutiveShl>(&getContext());
    // Run the patterns to a fixpoint. The greedy driver also folds and
    // erases trivially dead ops along the way -- your patterns don't have
    // to clean up dead constants.
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::school
