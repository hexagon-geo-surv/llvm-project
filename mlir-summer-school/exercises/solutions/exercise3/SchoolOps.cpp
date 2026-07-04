//===- SchoolOps.cpp - School dialect ops -----------------------*- C++ -*-===//
//
// Reference solution for Exercise 3: the MaxOp folder (checkpoints 2+3) and
// the reassociation canonicalization pattern (stretch goal b).
//
//===----------------------------------------------------------------------===//

#include "School/SchoolOps.h"
#include "School/SchoolDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::school;

//===----------------------------------------------------------------------===//
// MaxOp folder (exercise 3, checkpoints 2 + 3)
//===----------------------------------------------------------------------===//

// The fold contract: no new ops, no IR mutation. We may only return an
// existing Value, an Attribute (a constant -- materialized by the dialect's
// materializeConstant hook), or {} for "no fold".
OpFoldResult MaxOp::fold(FoldAdaptor adaptor) {
  // max(x, x) -> x: replace the op with an existing value.
  if (getLhs() == getRhs())
    return getLhs();

  // max(c1, c2) -> the larger constant. The adaptor holds an Attribute for
  // every operand defined by a constant op, and a *null* Attribute
  // otherwise -- hence dyn_cast_if_present, never a plain cast.
  auto lhsCst = dyn_cast_if_present<IntegerAttr>(adaptor.getLhs());
  auto rhsCst = dyn_cast_if_present<IntegerAttr>(adaptor.getRhs());
  if (lhsCst && rhsCst)
    return lhsCst.getValue().sgt(rhsCst.getValue()) ? lhsCst : rhsCst;

  // No fold applies.
  return {};
}

//===----------------------------------------------------------------------===//
// MaxOp canonicalization (exercise 3, stretch goal b)
//===----------------------------------------------------------------------===//

namespace {
/// max(max(x, c1), c2) -> max(x, c3) with c3 = max(c1, c2).
///
/// This cannot be a fold: it creates a new constant op. It relies on the
/// Commutative trait having moved constant operands to the right, so only
/// the rhs of each op needs checking -- that is the point of canonical
/// forms: every pattern downstream matches one shape instead of four.
struct ReassociateConstantMax : public OpRewritePattern<MaxOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MaxOp op,
                                PatternRewriter &rewriter) const override {
    APInt outerCst;
    if (!matchPattern(op.getRhs(), m_ConstantInt(&outerCst)))
      return rewriter.notifyMatchFailure(op, "rhs is not a constant");
    auto inner = op.getLhs().getDefiningOp<MaxOp>();
    if (!inner)
      return rewriter.notifyMatchFailure(op, "lhs is not another school.max");
    APInt innerCst;
    if (!matchPattern(inner.getRhs(), m_ConstantInt(&innerCst)))
      return rewriter.notifyMatchFailure(op, "inner rhs is not a constant");

    // Each application removes one max from the chain, so repeated
    // application converges (a hard requirement for canonicalization
    // patterns).
    APInt merged = innerCst.sgt(outerCst) ? innerCst : outerCst;
    Value mergedCst = arith::ConstantOp::create(
        rewriter, op.getLoc(), rewriter.getIntegerAttr(op.getType(), merged));
    rewriter.replaceOpWithNewOp<MaxOp>(op, op.getType(), inner.getLhs(),
                                       mergedCst);
    return success();
  }
};
} // namespace

void MaxOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<ReassociateConstantMax>(context);
}

#define GET_OP_CLASSES
#include "School/SchoolOps.cpp.inc"
