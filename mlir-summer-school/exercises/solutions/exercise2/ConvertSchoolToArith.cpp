//===- ConvertSchoolToArith.cpp - Exercise 2B: dialect conversion -*- C++ -*-=//
//
// Reference solution for Exercise 2, part B.
//
// Lowers the school dialect to arith via the dialect conversion framework:
// a ConversionTarget describes the goal state (school illegal, arith legal),
// OpConversionPatterns describe the individual rewrites, and
// applyPartialConversion drives the process. The driver fails loudly if an
// explicitly-illegal op has no applicable pattern -- exactly what a lowering
// should do.
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
    // Operands come from the adaptor: those are the *already converted*
    // values. (`op.getLhs()` would be the original operand -- with a type
    // converter in play the two can differ. Here the types are all i32,
    // but the habit is worth building early.)
    Value isGreater =
        arith::CmpIOp::create(rewriter, op.getLoc(),
                              arith::CmpIPredicate::sgt, adaptor.getLhs(),
                              adaptor.getRhs());
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, isGreater,
                                                 adaptor.getLhs(),
                                                 adaptor.getRhs());
    return success();
  }
};

/// school.mac %a, %b, %c  -->  arith.muli + arith.addi.
struct MacOpLowering : public OpConversionPattern<MacOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MacOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value product = arith::MulIOp::create(rewriter, op.getLoc(),
                                          adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOpWithNewOp<arith::AddIOp>(op, product, adaptor.getAcc());
    return success();
  }
};

struct ConvertSchoolToArith
    : public impl::ConvertSchoolToArithBase<ConvertSchoolToArith> {
  void runOnOperation() override {
    // The goal state: no school ops may survive; arith ops are fine.
    // Ops we say nothing about (func.func, func.return, ...) are "unknown";
    // a *partial* conversion leaves them untouched.
    ConversionTarget target(getContext());
    target.addIllegalDialect<SchoolDialect>();
    target.addLegalDialect<arith::ArithDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<MaxOpLowering, MacOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::school
