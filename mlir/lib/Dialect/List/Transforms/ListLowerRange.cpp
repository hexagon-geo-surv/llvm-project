//===- ListLowerRange.cpp - Lower list.range to an scf.for loop -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/List/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/List/IR/List.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace list {

#define GEN_PASS_DEF_LISTLOWERRANGE
#include "mlir/Dialect/List/Transforms/Passes.h.inc"

namespace {

/// Build the list of a `list.range` operation one element at a time:
///
/// ```mlir
/// %li = list.range %lb to %ub : !list.list<i32>
/// ```
///
/// becomes:
///
/// ```mlir
/// %empty = list.from_elements : () -> !list.list<i32>
/// %c1 = arith.constant 1 : i32
/// %li = scf.for %i = %lb to %ub step %c1
///     iter_args(%collected = %empty) -> (!list.list<i32>) {
///   %longer = list.push_back %collected, %i : !list.list<i32>
///   scf.yield %longer : !list.list<i32>
/// }
/// ```
///
/// The bounds of an `scf.for` may be signless integers, so the induction
/// variable is the element to append and no conversion to `index` is needed. An
/// `scf.for` whose upper bound is not greater than its lower bound runs zero
/// times, which is exactly the empty list that `list.range` produces in that
/// case.
struct LowerRangeToForLoop : OpRewritePattern<RangeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RangeOp rangeOp,
                                PatternRewriter &rewriter) const override {
    Location loc = rangeOp.getLoc();
    Value lower = rangeOp.getLower();
    Type resultType = rangeOp.getResult().getType();

    // The induction variable of the loop has the type of the bounds and is
    // appended to the list as is, so it has to be an element of that list.
    Type elementType = cast<ListType>(resultType).getElementType();
    if (elementType != lower.getType())
      return rewriter.notifyMatchFailure(
          rangeOp, "elements of the list are not of the type of the bounds");

    // The loop carries the list built so far, which starts out empty, and
    // appends one element per iteration.
    Value empty = FromElementsOp::create(rewriter, loc, resultType,
                                         /*elements=*/ValueRange{});
    Value step =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(1));
    auto forOp = scf::ForOp::create(rewriter, loc, lower, rangeOp.getUpper(),
                                    step, /*initArgs=*/ValueRange{empty});

    // The builder of `scf.for` leaves the body of a loop with iteration
    // arguments empty, because only the caller knows what to yield from it.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(forOp.getBody());
    Value longer = PushBackOp::create(rewriter, loc, forOp.getRegionIterArg(0),
                                      forOp.getInductionVar());
    scf::YieldOp::create(rewriter, loc, longer);

    rewriter.replaceOp(rangeOp, forOp.getResult(0));
    return success();
  }
};

struct ListLowerRangePass
    : public impl::ListLowerRangeBase<ListLowerRangePass> {

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerRangeToForLoop>(patterns.getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace list
} // namespace mlir
