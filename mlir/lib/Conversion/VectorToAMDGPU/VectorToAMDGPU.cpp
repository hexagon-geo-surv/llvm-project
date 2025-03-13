//===- VectorToAMDGPU.cpp - Vector to AMDGPU dialect conversion ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/VectorToAMDGPU/VectorToAMDGPU.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTVECTORTOAMDGPUPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

static LogicalResult
transferPreconditions(PatternRewriter &rewriter,
                      VectorTransferOpInterface xferOp,
                      SmallVector<unsigned> &broadcastedDims,
                      VectorType &unbroadcastedVectorType) {
  if (!xferOp.getMask())
    return rewriter.notifyMatchFailure(xferOp, "Only support masked transfer");
  // TODO reject when source operand is not from buffer address space.

  // Permutations are handled by VectorToSCF or
  // populateVectorTransferPermutationMapLoweringPatterns.
  // We let the 0-d corner case pass-through as it is supported.
  if (!xferOp.getPermutationMap().isMinorIdentityWithBroadcasting(
          &broadcastedDims))
    return rewriter.notifyMatchFailure(xferOp, "not minor identity + bcast");

  auto memRefType = dyn_cast<MemRefType>(xferOp.getShapedType());
  if (!memRefType)
    return rewriter.notifyMatchFailure(xferOp, "not a memref source");

  // Non-unit strides are handled by VectorToSCF.
  if (!memRefType.isLastDimUnitStride())
    return rewriter.notifyMatchFailure(xferOp, "!= 1 stride needs VectorToSCF");

  // If there is broadcasting involved then we first load the unbroadcasted
  // vector, and then broadcast it with `vector.broadcast`.
  ArrayRef<int64_t> vectorShape = xferOp.getVectorType().getShape();
  SmallVector<int64_t> unbroadcastedVectorShape(vectorShape);
  for (unsigned i : broadcastedDims)
    unbroadcastedVectorShape[i] = 1;
  unbroadcastedVectorType = xferOp.getVectorType().cloneWith(
      unbroadcastedVectorShape, xferOp.getVectorType().getElementType());

  // `vector.load` supports vector types as memref's elements only when the
  // resulting vector type is the same as the element type.
  auto memrefElTy = memRefType.getElementType();
  if (isa<VectorType>(memrefElTy) && memrefElTy != unbroadcastedVectorType)
    return rewriter.notifyMatchFailure(xferOp, "incompatible element type");

  // Otherwise, element types of the memref and the vector must match.
  if (!isa<VectorType>(memrefElTy) &&
      memrefElTy != xferOp.getVectorType().getElementType())
    return rewriter.notifyMatchFailure(xferOp, "non-matching element type");

  // Out-of-bounds dims are handled by MaterializeTransferMask.
  if (xferOp.hasOutOfBoundsDim())
    return rewriter.notifyMatchFailure(xferOp, "out-of-bounds needs mask");

  if (xferOp.getVectorType().getRank() != 1)
    // vector.maskedload operates on 1-D vectors.
    return rewriter.notifyMatchFailure(
        xferOp, "vector type is not rank 1, can't create masked load, needs "
                "VectorToSCF");

  return success();
}

struct TransferReadLowering : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {

    SmallVector<unsigned> broadcastedDims;
    VectorType unbroadcastedVectorType;
    if (failed(transferPreconditions(rewriter, readOp, broadcastedDims,
                                     unbroadcastedVectorType))) {
      return failure();
    }

    Value fill = rewriter.create<vector::SplatOp>(
        readOp.getLoc(), unbroadcastedVectorType, readOp.getPadding());
    Value load = rewriter.create<vector::LoadOp>(
        readOp.getLoc(), unbroadcastedVectorType, readOp.getSource(),
        readOp.getIndices());
    Value res = rewriter.create<arith::SelectOp>(
        readOp.getLoc(), readOp.getVectorType(), readOp.getMask(), load, fill);

    // Insert a broadcasting op if required.
    if (!broadcastedDims.empty())
      res = rewriter.create<vector::BroadcastOp>(readOp.getLoc(),
                                                 readOp.getVectorType(), res);

    rewriter.replaceOp(readOp, res);

    return success();
  }
};

void mlir::populateVectorToAMDGPUConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<TransferReadLowering>(patterns.getContext());
}

struct ConvertVectorToAMDGPUPass
    : public impl::ConvertVectorToAMDGPUPassBase<ConvertVectorToAMDGPUPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateVectorToAMDGPUConversionPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
