//===- ListLowerMap.cpp - Lower list.map to an scf.while loop -------------===//
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
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace list {

#define GEN_PASS_DEF_LISTLOWERMAP
#include "mlir/Dialect/List/Transforms/Passes.h.inc"

namespace {

/// Turn a `list.map` operation into a loop over the elements of the mapped
/// list:
///
/// ```mlir
/// %result = list.map %input with (%element : i32) -> i64 {
///   %mapped = ...
///   list.yield %mapped : i64
/// }
/// ```
///
/// becomes:
///
/// ```mlir
/// %empty = list.from_elements : () -> !list.list<i64>
/// %loop:2 = scf.while (%remaining = %input, %collected = %empty)
///     : (!list.list<i32>, !list.list<i64>) -> (!list.list<i32>,
///                                              !list.list<i64>) {
///   %length = list.length %remaining : !list.list<i32> -> i32
///   %c0 = arith.constant 0 : i32
///   %not_empty = arith.cmpi ne, %length, %c0 : i32
///   scf.condition(%not_empty) %remaining, %collected
///       : !list.list<i32>, !list.list<i64>
/// } do {
/// ^bb0(%remaining : !list.list<i32>, %collected : !list.list<i64>):
///   %element = list.peek_front %remaining : !list.list<i32> -> i32
///   %rest = list.pop_front %remaining : !list.list<i32>
///   %mapped = ...
///   %longer = list.push_back %collected, %mapped : !list.list<i64>
///   scf.yield %rest, %longer : !list.list<i32>, !list.list<i64>
/// }
/// // %loop#1 replaces %result
/// ```
///
/// The loop carries two lists: the elements that still have to be mapped and
/// the mapped elements collected so far. Taking the elements off the front of
/// the first list and appending them to the back of the second one keeps the
/// order of the elements intact.
struct LowerMapToWhileLoop : OpRewritePattern<MapOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MapOp mapOp,
                                PatternRewriter &rewriter) const override {
    Location loc = mapOp.getLoc();
    Type remainingType = mapOp.getInput().getType();
    Type collectedType = mapOp.getResult().getType();

    // The list of mapped elements starts out empty.
    Value empty = FromElementsOp::create(rewriter, loc, collectedType,
                                         /*elements=*/ValueRange{});

    // Both loop-carried lists are passed on unchanged by the condition of the
    // loop, so the operation results and the arguments of both regions all have
    // the same types.
    SmallVector<Type> loopTypes = {remainingType, collectedType};
    SmallVector<Value> inits = {mapOp.getInput(), empty};
    auto whileOp = scf::WhileOp::create(rewriter, loc, loopTypes, inits,
                                        /*beforeBuilder=*/nullptr,
                                        /*afterBuilder=*/nullptr);

    // The "before" region decides whether another element is left to map.
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(whileOp.getBeforeBody());
      Value remaining = whileOp.getBeforeArguments()[0];
      Value length = LengthOp::create(rewriter, loc, remaining);
      Value zero = arith::ConstantOp::create(rewriter, loc,
                                             rewriter.getI32IntegerAttr(0));
      Value notEmpty = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::ne, length, zero);
      scf::ConditionOp::create(rewriter, loc, notEmpty,
                               whileOp.getBeforeArguments());
    }

    // The "after" region maps a single element.
    {
      OpBuilder::InsertionGuard guard(rewriter);
      Block *afterBody = whileOp.getAfterBody();
      rewriter.setInsertionPointToEnd(afterBody);
      Value remaining = whileOp.getAfterArguments()[0];
      Value collected = whileOp.getAfterArguments()[1];
      Value element = PeekFrontOp::create(rewriter, loc, remaining);
      Value rest = PopFrontOp::create(rewriter, loc, remaining);

      // Move the body of the `list.map` into the loop and let it compute on the
      // element that was just taken off the list. The yield operation is looked
      // up before inlining, but its operand is read afterwards: inlining
      // replaces the block argument of the body, which the yield may use.
      auto yieldOp = cast<list::YieldOp>(mapOp.getBodyBlock().getTerminator());
      rewriter.inlineBlockBefore(&mapOp.getBodyBlock(), afterBody,
                                 afterBody->end(), element);
      Value mapped = yieldOp.getYielded();
      rewriter.eraseOp(yieldOp);

      rewriter.setInsertionPointToEnd(afterBody);
      Value longer = PushBackOp::create(rewriter, loc, collected, mapped);
      scf::YieldOp::create(rewriter, loc, ValueRange{rest, longer});
    }

    // The loop ends with an empty list of remaining elements, which is of no
    // interest; the mapped elements are the result of the `list.map`.
    rewriter.replaceOp(mapOp, whileOp.getResult(1));
    return success();
  }
};

struct ListLowerMapPass : public impl::ListLowerMapBase<ListLowerMapPass> {

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerMapToWhileLoop>(patterns.getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace list
} // namespace mlir
