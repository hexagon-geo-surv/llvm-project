//===- Peephole.cpp - Exercise 2A: rewrite patterns -------------*- C++ -*-===//
//
// Reference solution for Exercise 2, part A.
//
// The Exercise 1 rewrite, re-expressed as an OpRewritePattern, plus a second
// pattern that merges consecutive shifts. applyPatternsGreedily runs both to
// a fixpoint, so the two patterns compose: ((x*4)*8) becomes a single
// shli-by-5 even though no single pattern performs that rewrite.
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
    // All match checks come first: if we return failure(), we must not
    // have touched the IR (the matchAndRewrite contract). notifyMatchFailure
    // documents *why* we bailed out -- visible under
    // --debug-only=greedy-rewriter.
    APInt rhsValue;
    if (!matchPattern(op.getRhs(), m_ConstantInt(&rhsValue)))
      return rewriter.notifyMatchFailure(op, "rhs is not a constant integer");
    if (!rhsValue.isPowerOf2())
      return rewriter.notifyMatchFailure(op, "rhs is not a power of two");

    // Rewrite: every mutation goes through the rewriter so the driver can
    // track it (that is how new ops end up back on the worklist).
    Value shiftAmount = arith::ConstantOp::create(
        rewriter, op.getLoc(),
        rewriter.getIntegerAttr(op.getType(), rhsValue.logBase2()));
    rewriter.replaceOpWithNewOp<arith::ShLIOp>(op, op.getLhs(), shiftAmount);
    return success();
  }
};

/// shli(shli(x, C1), C2) -> shli(x, C1 + C2) when both shift amounts are
/// constants and C1 + C2 is still smaller than the bit width.
struct MergeConsecutiveShl : public OpRewritePattern<arith::ShLIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ShLIOp op,
                                PatternRewriter &rewriter) const override {
    auto inner = op.getLhs().getDefiningOp<arith::ShLIOp>();
    if (!inner)
      return rewriter.notifyMatchFailure(op, "lhs is not another shli");

    APInt outerAmount, innerAmount;
    if (!matchPattern(op.getRhs(), m_ConstantInt(&outerAmount)) ||
        !matchPattern(inner.getRhs(), m_ConstantInt(&innerAmount)))
      return rewriter.notifyMatchFailure(op, "shift amount is not a constant");

    // Guard: shifting by >= the bit width is poison, so merging would turn
    // two well-defined shifts into an ill-defined one. Also guards
    // convergence: the pattern strictly shrinks the shli chain only when it
    // actually fires.
    bool overflow = false;
    APInt total = innerAmount.uadd_ov(outerAmount, overflow);
    if (overflow || total.uge(op.getType().getIntOrFloatBitWidth()))
      return rewriter.notifyMatchFailure(op, "total shift amount too large");

    Value newAmount = arith::ConstantOp::create(
        rewriter, op.getLoc(), rewriter.getIntegerAttr(op.getType(), total));
    rewriter.replaceOpWithNewOp<arith::ShLIOp>(op, inner.getLhs(), newAmount);
    return success();
  }
};

struct SchoolPeephole : public impl::SchoolPeepholeBase<SchoolPeephole> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MulByPow2ToShl, MergeConsecutiveShl>(&getContext());
    // Fixpoint iteration; the driver also folds and DCEs along the way, so
    // the dead constants left behind by the rewrites vanish for free.
    // failure() here means "did not converge", not "nothing matched".
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::school
