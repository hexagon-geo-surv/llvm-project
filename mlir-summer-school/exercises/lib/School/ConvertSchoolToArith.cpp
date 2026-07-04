//===- ConvertSchoolToArith.cpp - Exercise 2B: dialect conversion -*- C++ -*-=//
//
// Exercise 2, part B: lower the school dialect to arith with the dialect
// conversion framework. The three ingredients:
//
//   1. A ConversionTarget: school is illegal, arith is legal.
//   2. Conversion patterns: school.max -> arith.cmpi sgt + arith.select,
//                           school.mac -> arith.muli + arith.addi.
//   3. A driver: applyPartialConversion.
//
// (No TypeConverter here: all types stay i32. Type conversion enters the
// picture only when the lowering changes types.)
//
// Run with:
//
//   school-opt in.mlir -pass-pipeline="builtin.module(convert-school-to-arith)"
//
//===----------------------------------------------------------------------===//

#include "School/SchoolPasses.h"

#include "School/SchoolDialect.h"
#include "School/SchoolOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::school {
#define GEN_PASS_DEF_CONVERTSCHOOLTOARITH
#include "School/SchoolPasses.h.inc"

namespace {

/// school.max %a, %b  -->  arith.cmpi sgt + arith.select.
struct MaxOpLowering : public OpConversionPattern<MaxOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO(exercise 2B, step 2): Implement the lowering.
    //   Create an arith::CmpIOp with predicate arith::CmpIPredicate::sgt,
    //   then rewriter.replaceOpWithNewOp<arith::SelectOp>(...).
    //   Take the operands from `adaptor` (adaptor.getLhs(), ...), not from
    //   `op` -- the adaptor holds the already-converted operands.
    return rewriter.notifyMatchFailure(op, "pattern not implemented yet");
  }
};

// TODO(exercise 2B, step 3): Add a MacOpLowering pattern here
// (school.mac %a, %b, %c --> arith.muli + arith.addi) and register it below.
// Before you write it, run the pass on test/exercise2/convert-mac.mlir and
// read the error the driver gives you.

struct ConvertSchoolToArith
    : public impl::ConvertSchoolToArithBase<ConvertSchoolToArith> {
  void runOnOperation() override {
    // TODO(exercise 2B, step 1): Describe the goal state.
    //   Mark the school dialect illegal and the arith dialect legal on
    //   `target` (ConversionTarget::addIllegalDialect / addLegalDialect).
    //   With an empty target every op is "unknown", and a *partial*
    //   conversion happily leaves unknown ops as they are -- which is why
    //   this starter pass runs and changes nothing.
    ConversionTarget target(getContext());

    RewritePatternSet patterns(&getContext());
    patterns.add<MaxOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::school
