//===- ListSimplifyElements.cpp - Simplify element operations -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/List/Transforms/Passes.h"

#include "mlir/Dialect/List/IR/List.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace list {

#define GEN_PASS_DEF_LISTSIMPLIFYELEMENTS
#include "mlir/Dialect/List/Transforms/Passes.h.inc"

namespace {

/// Extract the last element of a `list.get_elements` operation with a
/// `list.peek_back` and take it off the list with a `list.pop_back`, leaving a
/// `list.get_elements` for the remaining elements:
///
/// ```mlir
/// %a, %b = list.get_elements %x : (!list.list<i32>) -> (i32, i32)
/// ```
///
/// becomes:
///
/// ```mlir
/// %b = list.peek_back %x : !list.list<i32> -> i32
/// %shorter = list.pop_back %x : !list.list<i32>
/// %a = list.get_elements %shorter : (!list.list<i32>) -> (i32)
/// ```
///
/// Applying this repeatedly replaces every `list.get_elements` with a sequence
/// of `list.peek_back` and `list.pop_back` operations.
struct UnrollGetElements : OpRewritePattern<GetElementsOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(GetElementsOp op,
                                PatternRewriter &rewriter) const override {
    // All elements have been extracted, so nothing is left to do. (Without this
    // case, the pattern would apply forever.)
    if (op.getNumResults() == 0) {
      rewriter.eraseOp(op);
      return success();
    }

    Location loc = op.getLoc();
    Value input = op.getInput();

    // The last result of the operation is the back of the list, the remaining
    // results are the elements of the list without its back.
    Value back = PeekBackOp::create(rewriter, loc, input);
    Value shorter = PopBackOp::create(rewriter, loc, input);
    SmallVector<Type> frontTypes = llvm::to_vector(op.getResultTypes());
    frontTypes.pop_back();
    auto frontOp = GetElementsOp::create(rewriter, loc, frontTypes, shorter);

    SmallVector<Value> results;
    llvm::append_range(results, frontOp.getResults());
    results.push_back(back);
    rewriter.replaceOp(op, results);
    return success();
  }
};

/// Append the item of a `list.push_back` to the `list.from_elements` operation
/// that builds its operand list:
///
/// ```mlir
/// %l = list.from_elements %a, %b : (i32, i32) -> !list.list<i32>
/// %m = list.push_back %l, %c : !list.list<i32>
/// ```
///
/// becomes:
///
/// ```mlir
/// %m = list.from_elements %a, %b, %c : (i32, i32, i32) -> !list.list<i32>
/// ```
struct FoldPushBackIntoFromElements : OpRewritePattern<PushBackOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PushBackOp op,
                                PatternRewriter &rewriter) const override {
    auto fromElements = op.getInput().getDefiningOp<FromElementsOp>();
    if (!fromElements)
      return rewriter.notifyMatchFailure(op,
                                         "operand is not a list.from_elements");

    // Unlike a `list.map` body, the elements are simply operands, so there is
    // nothing to duplicate if `fromElements` has other users. It is left in
    // place for them.
    SmallVector<Value> elements = llvm::to_vector(fromElements.getElements());
    elements.push_back(op.getItem());
    rewriter.replaceOpWithNewOp<FromElementsOp>(op, op.getResult().getType(),
                                                elements);
    return success();
  }
};

struct ListSimplifyElementsPass
    : public impl::ListSimplifyElementsBase<ListSimplifyElementsPass> {

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<UnrollGetElements, FoldPushBackIntoFromElements>(
        patterns.getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace list
} // namespace mlir
