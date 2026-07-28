//===- ListSimplify.cpp - Simplify and lower list operations --------------===//
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
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace list {

#define GEN_PASS_DEF_LISTSIMPLIFY
#include "mlir/Dialect/List/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Simplifications
//===----------------------------------------------------------------------===//

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

/// Reverse a list without any element by doing nothing at all:
///
/// ```mlir
/// %empty = list.empty : !list.list<i32>
/// %reversed = list.reverse %empty : !list.list<i32>
/// ```
///
/// becomes:
///
/// ```mlir
/// %reversed = list.empty : !list.list<i32>
/// ```
///
/// Being a simplification, this is tried before the lowering below, so that a
/// `list.reverse` of an empty list does not become a loop that never runs.
struct ReplaceReverseOfEmpty : OpRewritePattern<ReverseOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReverseOp reverseOp,
                                PatternRewriter &rewriter) const override {
    if (!reverseOp.getInput().getDefiningOp<EmptyOp>())
      return rewriter.notifyMatchFailure(reverseOp,
                                         "operand is not a list.empty");

    rewriter.replaceOpWithNewOp<EmptyOp>(reverseOp,
                                         reverseOp.getResult().getType());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Conversions between lists and their elements
//===----------------------------------------------------------------------===//

/// Extract the first element of a `list.get_elements` operation with a
/// `list.peek_front` and take it off the list with a `list.pop_front`, leaving
/// a `list.get_elements` for the remaining elements:
///
/// ```mlir
/// %a, %b = list.get_elements %x : (!list.list<i32>) -> (i32, i32)
/// ```
///
/// becomes:
///
/// ```mlir
/// %a = list.peek_front %x : !list.list<i32> -> i32
/// %shorter = list.pop_front %x : !list.list<i32>
/// %b = list.get_elements %shorter : (!list.list<i32>) -> (i32)
/// ```
///
/// Applying this repeatedly replaces every `list.get_elements` with a sequence
/// of `list.peek_front` and `list.pop_front` operations.
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

    // The first result of the operation is the front of the list, the remaining
    // results are the elements of the list without its front.
    Value front = PeekFrontOp::create(rewriter, loc, input);
    Value shorter = PopFrontOp::create(rewriter, loc, input);
    SmallVector<Type> restTypes =
        llvm::to_vector(llvm::drop_begin(op.getResultTypes()));
    auto restOp = GetElementsOp::create(rewriter, loc, restTypes, shorter);

    SmallVector<Value> results = {front};
    llvm::append_range(results, restOp.getResults());
    rewriter.replaceOp(op, results);
    return success();
  }
};

/// Append the last element of a `list.from_elements` with a `list.push_back`,
/// leaving a `list.from_elements` for the remaining elements:
///
/// ```mlir
/// %li = list.from_elements %a, %b, %c : (i32, i32, i32) -> !list.list<i32>
/// ```
///
/// becomes:
///
/// ```mlir
/// %shorter = list.from_elements %a, %b : (i32, i32) -> !list.list<i32>
/// %li = list.push_back %shorter, %c : !list.list<i32>
/// ```
///
/// Applying this repeatedly replaces every `list.from_elements` with a
/// `list.from_elements` without any element and a sequence of `list.push_back`
/// operations. The pattern below turns what is left into a `list.empty`.
struct UnrollFromElements : OpRewritePattern<FromElementsOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(FromElementsOp op,
                                PatternRewriter &rewriter) const override {
    // A list without any element is not built up from other operations.
    // (Without this case, the pattern would apply forever.)
    if (op.getElements().empty())
      return rewriter.notifyMatchFailure(op, "list has no element");

    Location loc = op.getLoc();
    Type resultType = op.getResult().getType();

    // The list of all elements is the list of all but the last element with the
    // last element appended to it.
    ValueRange elements = op.getElements();
    auto shorterOp =
        FromElementsOp::create(rewriter, loc, resultType, elements.drop_back());
    rewriter.replaceOpWithNewOp<PushBackOp>(op, shorterOp.getResult(),
                                            elements.back());
    return success();
  }
};

/// Build a list without any element with `list.empty`:
///
/// ```mlir
/// %li = list.from_elements : () -> !list.list<i32>
/// ```
///
/// becomes:
///
/// ```mlir
/// %li = list.empty : !list.list<i32>
/// ```
struct ReplaceEmptyFromElements : OpRewritePattern<FromElementsOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(FromElementsOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getElements().empty())
      return rewriter.notifyMatchFailure(op, "list has elements");

    rewriter.replaceOpWithNewOp<EmptyOp>(op, op.getResult().getType());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lowerings to loops
//===----------------------------------------------------------------------===//

/// Build the condition of a loop that runs as long as `list` holds an element.
/// The `scf.while` loop continues while its condition holds, so it is the
/// negation of the emptiness test that keeps it going.
static Value createNotEmpty(PatternRewriter &rewriter, Location loc,
                            Value list) {
  Value isEmpty = IsEmptyOp::create(rewriter, loc, list);
  Value trueValue =
      arith::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(true));
  return arith::XOrIOp::create(rewriter, loc, isEmpty, trueValue);
}

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
/// %empty = list.empty : !list.list<i64>
/// %loop:2 = scf.while (%remaining = %input, %collected = %empty)
///     : (!list.list<i32>, !list.list<i64>) -> (!list.list<i32>,
///                                              !list.list<i64>) {
///   %is_empty = list.is_empty %remaining : !list.list<i32> -> i1
///   %true = arith.constant true
///   %not_empty = arith.xori %is_empty, %true : i1
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
    Value empty = EmptyOp::create(rewriter, loc, collectedType);

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
      Value notEmpty =
          createNotEmpty(rewriter, loc, whileOp.getBeforeArguments()[0]);
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

/// Build the list of a `list.range` operation one element at a time:
///
/// ```mlir
/// %li = list.range %lb to %ub : !list.list<i32>
/// ```
///
/// becomes:
///
/// ```mlir
/// %empty = list.empty : !list.list<i32>
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
    Value empty = EmptyOp::create(rewriter, loc, resultType);
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

/// Reverse a list by moving its elements one at a time:
///
/// ```mlir
/// %reversed = list.reverse %input : !list.list<i32>
/// ```
///
/// becomes:
///
/// ```mlir
/// %empty = list.empty : !list.list<i32>
/// %loop:2 = scf.while (%remaining = %input, %collected = %empty)
///     : (!list.list<i32>, !list.list<i32>) -> (!list.list<i32>,
///                                              !list.list<i32>) {
///   %is_empty = list.is_empty %remaining : !list.list<i32> -> i1
///   %true = arith.constant true
///   %not_empty = arith.xori %is_empty, %true : i1
///   scf.condition(%not_empty) %remaining, %collected
///       : !list.list<i32>, !list.list<i32>
/// } do {
/// ^bb0(%remaining : !list.list<i32>, %collected : !list.list<i32>):
///   %element = list.peek_front %remaining : !list.list<i32> -> i32
///   %rest = list.pop_front %remaining : !list.list<i32>
///   %longer = list.push_front %collected, %element : !list.list<i32>
///   scf.yield %rest, %longer : !list.list<i32>, !list.list<i32>
/// }
/// // %loop#1 replaces %reversed
/// ```
///
/// Taking the elements off the front of the input list and prepending them to
/// the result list is what turns their order around: the element that is moved
/// first ends up last.
struct LowerReverseToWhileLoop : OpRewritePattern<ReverseOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReverseOp reverseOp,
                                PatternRewriter &rewriter) const override {
    Location loc = reverseOp.getLoc();
    Type listType = reverseOp.getInput().getType();

    // The loop carries the elements that still have to be moved and the
    // reversed elements collected so far, which start out empty.
    Value empty = EmptyOp::create(rewriter, loc, listType);
    SmallVector<Type> loopTypes = {listType, listType};
    SmallVector<Value> inits = {reverseOp.getInput(), empty};
    auto whileOp = scf::WhileOp::create(rewriter, loc, loopTypes, inits,
                                        /*beforeBuilder=*/nullptr,
                                        /*afterBuilder=*/nullptr);

    // The "before" region decides whether another element is left to move.
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(whileOp.getBeforeBody());
      Value notEmpty =
          createNotEmpty(rewriter, loc, whileOp.getBeforeArguments()[0]);
      scf::ConditionOp::create(rewriter, loc, notEmpty,
                               whileOp.getBeforeArguments());
    }

    // The "after" region moves a single element.
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(whileOp.getAfterBody());
      Value remaining = whileOp.getAfterArguments()[0];
      Value collected = whileOp.getAfterArguments()[1];
      Value element = PeekFrontOp::create(rewriter, loc, remaining);
      Value rest = PopFrontOp::create(rewriter, loc, remaining);
      Value longer = PushFrontOp::create(rewriter, loc, collected, element);
      scf::YieldOp::create(rewriter, loc, ValueRange{rest, longer});
    }

    // The loop ends with an empty list of remaining elements, which is of no
    // interest; the moved elements are the result of the `list.reverse`.
    rewriter.replaceOp(reverseOp, whileOp.getResult(1));
    return success();
  }
};

/// Count the elements of a list by taking them off one at a time:
///
/// ```mlir
/// %length = list.length %input : !list.list<i32> -> i32
/// ```
///
/// becomes:
///
/// ```mlir
/// %c0 = arith.constant 0 : i32
/// %loop:2 = scf.while (%remaining = %input, %count = %c0)
///     : (!list.list<i32>, i32) -> (!list.list<i32>, i32) {
///   %is_empty = list.is_empty %remaining : !list.list<i32> -> i1
///   %true = arith.constant true
///   %not_empty = arith.xori %is_empty, %true : i1
///   scf.condition(%not_empty) %remaining, %count : !list.list<i32>, i32
/// } do {
/// ^bb0(%remaining : !list.list<i32>, %count : i32):
///   %rest = list.pop_front %remaining : !list.list<i32>
///   %c1 = arith.constant 1 : i32
///   %next = arith.addi %count, %c1 : i32
///   scf.yield %rest, %next : !list.list<i32>, i32
/// }
/// // %loop#1 replaces %length
/// ```
///
/// Unlike the loops that the other patterns build, this one does not carry two
/// lists but a list and the number of elements that have been taken off it so
/// far, which is the length of the list once nothing is left of it.
struct LowerLengthToWhileLoop : OpRewritePattern<LengthOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LengthOp lengthOp,
                                PatternRewriter &rewriter) const override {
    Location loc = lengthOp.getLoc();
    Type listType = lengthOp.getInput().getType();
    Type countType = lengthOp.getResult().getType();

    // Nothing has been taken off the list yet.
    Value zero =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
    SmallVector<Type> loopTypes = {listType, countType};
    SmallVector<Value> inits = {lengthOp.getInput(), zero};
    auto whileOp = scf::WhileOp::create(rewriter, loc, loopTypes, inits,
                                        /*beforeBuilder=*/nullptr,
                                        /*afterBuilder=*/nullptr);

    // The "before" region decides whether another element is left to count.
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(whileOp.getBeforeBody());
      Value notEmpty =
          createNotEmpty(rewriter, loc, whileOp.getBeforeArguments()[0]);
      scf::ConditionOp::create(rewriter, loc, notEmpty,
                               whileOp.getBeforeArguments());
    }

    // The "after" region takes a single element off the list and counts it.
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(whileOp.getAfterBody());
      Value remaining = whileOp.getAfterArguments()[0];
      Value count = whileOp.getAfterArguments()[1];
      Value rest = PopFrontOp::create(rewriter, loc, remaining);
      Value one = arith::ConstantOp::create(rewriter, loc,
                                            rewriter.getI32IntegerAttr(1));
      Value next = arith::AddIOp::create(rewriter, loc, count, one);
      scf::YieldOp::create(rewriter, loc, ValueRange{rest, next});
    }

    // The loop ends with an empty list, which is of no interest; the number of
    // elements that were taken off it is the result of the `list.length`.
    rewriter.replaceOp(lengthOp, whileOp.getResult(1));
    return success();
  }
};

/// Print the elements of a list one at a time:
///
/// ```mlir
/// list.print %input : !list.list<i32>
/// ```
///
/// becomes:
///
/// ```mlir
/// %loop = scf.while (%remaining = %input)
///     : (!list.list<i32>) -> !list.list<i32> {
///   %is_empty = list.is_empty %remaining : !list.list<i32> -> i1
///   %true = arith.constant true
///   %not_empty = arith.xori %is_empty, %true : i1
///   scf.condition(%not_empty) %remaining : !list.list<i32>
/// } do {
/// ^bb0(%remaining : !list.list<i32>):
///   %element = list.peek_front %remaining : !list.list<i32> -> i32
///   vector.print %element : i32
///   %rest = list.pop_front %remaining : !list.list<i32>
///   scf.yield %rest : !list.list<i32>
/// }
/// ```
///
/// The loop carries a single value, the elements that still have to be printed,
/// and its result is the empty list that is left over. Because `list.print` has
/// no result, nothing is left to replace and the operation is simply erased.
struct LowerPrintToWhileLoop : OpRewritePattern<PrintOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PrintOp printOp,
                                PatternRewriter &rewriter) const override {
    Location loc = printOp.getLoc();
    Value input = printOp.getInput();
    Type listType = input.getType();

    auto whileOp = scf::WhileOp::create(rewriter, loc, listType, input,
                                        /*beforeBuilder=*/nullptr,
                                        /*afterBuilder=*/nullptr);

    // The "before" region decides whether another element is left to print.
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(whileOp.getBeforeBody());
      Value notEmpty =
          createNotEmpty(rewriter, loc, whileOp.getBeforeArguments()[0]);
      scf::ConditionOp::create(rewriter, loc, notEmpty,
                               whileOp.getBeforeArguments());
    }

    // The "after" region prints a single element and takes it off the list.
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(whileOp.getAfterBody());
      Value remaining = whileOp.getAfterArguments()[0];
      Value element = PeekFrontOp::create(rewriter, loc, remaining);
      vector::PrintOp::create(rewriter, loc, element);
      Value rest = PopFrontOp::create(rewriter, loc, remaining);
      scf::YieldOp::create(rewriter, loc, rest);
    }

    rewriter.eraseOp(printOp);
    return success();
  }
};

struct ListSimplifyPass : public impl::ListSimplifyBase<ListSimplifyPass> {

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());

    // The simplifications are given a higher benefit than the lowerings, so
    // that they are tried first: a `list.map` that can be merged into another
    // one is merged before it is turned into a loop, and the length of a mapped
    // list is read through the map before it is counted.
    patterns
        .add<MergeConsecutiveMaps, SkipMapBeforeLength, ReplaceReverseOfEmpty>(
            patterns.getContext(), /*benefit=*/2);
    patterns
        .add<UnrollGetElements, UnrollFromElements, ReplaceEmptyFromElements,
             LowerMapToWhileLoop, LowerRangeToForLoop, LowerReverseToWhileLoop,
             LowerLengthToWhileLoop, LowerPrintToWhileLoop>(
            patterns.getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace list
} // namespace mlir
