//===- ListSimplify.cpp - Simplify list operations ------------------------===//
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

namespace mlir {
namespace list {

#define GEN_PASS_DEF_LISTSIMPLIFY
#include "mlir/Dialect/List/Transforms/Passes.h.inc"

namespace {

/// Merge a `list.map` whose operand is produced by another `list.map` into a
/// single `list.map` that applies both computations to every element:
///
/// ```mlir
/// %0 = list.map %x with (%a: i32) -> i32 { ... list.yield %p : i32 }
/// %1 = list.map %0 with (%b: i32) -> i64 { ... list.yield %c : i64 }
/// ```
///
/// becomes:
///
/// ```mlir
/// %1 = list.map %x with (%a: i32) -> i64 {
///   ...              // computes %p from %a
///   ...              // computes %c from %p
///   list.yield %c : i64
/// }
/// ```
struct MergeConsecutiveMaps : OpRewritePattern<MapOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MapOp mapOp,
                                PatternRewriter &rewriter) const override {
    auto producer = mapOp.getInput().getDefiningOp<MapOp>();
    if (!producer)
      return rewriter.notifyMatchFailure(mapOp, "operand is not a list.map");

    // The producer's body is moved into the merged operation, so the producer
    // must not be needed anywhere else. (This also rules out a producer whose
    // result is used inside the body of `mapOp`, which could not be merged.)
    if (!producer.getResult().hasOneUse())
      return rewriter.notifyMatchFailure(producer, "result has other users");

    // The merged operation maps over the producer's operand and yields the
    // element type that `mapOp` yields.
    auto elementType =
        cast<ListType>(mapOp.getResult().getType()).getElementType();
    auto mergedOp = MapOp::create(rewriter, mapOp.getLoc(), producer.getInput(),
                                  elementType);
    Block &mergedBody = mergedOp.getBodyBlock();

    // Move the producer's body into the merged body, computing on the element
    // of the list that the merged operation maps over.
    Block &producerBody = producer.getBodyBlock();
    auto producerYield = cast<YieldOp>(producerBody.getTerminator());
    rewriter.inlineBlockBefore(&producerBody, &mergedBody, mergedBody.end(),
                               mergedBody.getArgument(0));
    // Read the yielded value only now: inlining replaced the argument of the
    // producer's body, which may well be the value that it yields.
    Value producerElement = producerYield.getYielded();
    rewriter.eraseOp(producerYield);

    // Append the body of `mapOp`, which consumes the element that the
    // producer's body computes.
    rewriter.inlineBlockBefore(&mapOp.getBodyBlock(), &mergedBody,
                               mergedBody.end(), producerElement);

    rewriter.replaceOp(mapOp, mergedOp.getResult());
    // The producer is an empty operation without users at this point.
    rewriter.eraseOp(producer);
    return success();
  }
};

/// Compute the length of a mapped list from the list that it is mapped from,
/// because a `list.map` yields exactly one element per element of its operand:
///
/// ```mlir
/// %0 = list.map %x with (%a: i32) -> i64 { ... }
/// %1 = list.length %0 : !list.list<i64> -> i32
/// ```
///
/// becomes:
///
/// ```mlir
/// %1 = list.length %x : !list.list<i32> -> i32
/// ```
struct SkipMapBeforeLength : OpRewritePattern<LengthOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LengthOp lengthOp,
                                PatternRewriter &rewriter) const override {
    auto mapOp = lengthOp.getInput().getDefiningOp<MapOp>();
    if (!mapOp)
      return rewriter.notifyMatchFailure(lengthOp, "operand is not a list.map");

    // The `list.map` itself is left alone: its body may have side effects, and
    // it may have other users.
    rewriter.replaceOpWithNewOp<LengthOp>(lengthOp, mapOp.getInput());
    return success();
  }
};

struct ListSimplifyPass : public impl::ListSimplifyBase<ListSimplifyPass> {

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MergeConsecutiveMaps, SkipMapBeforeLength>(
        patterns.getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace list
} // namespace mlir
