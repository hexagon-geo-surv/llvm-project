//===- XeGPUUnroll.cpp - patterns to do unrolling ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/XeGPU/Transforms/Passes.h"

#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include <numeric>

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPUUNROLL
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

#define DEBUG_TYPE "xegpu-unroll"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

namespace {

template <typename SourceOp>
struct UnrollPattern : public OpRewritePattern<SourceOp> {
  UnrollPattern(MLIRContext *context, const xegpu::UnrollOptions &options,
                PatternBenefit benefit = 1)
      : OpRewritePattern<SourceOp>(context, benefit), options(options) {}

protected:
  std::optional<SmallVector<int64_t>> getTargetShape(Operation *op) const {
    LDBG("");
    LDBG("Get unroll shape for: " << *op);
    assert(options.nativeShape &&
           "expects the native shape for native shape call back function.");
    auto nativeShape = options.nativeShape(op);
    return nativeShape;
  }

  std::optional<SmallVector<int64_t>>
  computeGrids(llvm::ArrayRef<int64_t> shape,
               llvm::ArrayRef<int64_t> subShape) const {
    // if the shape == subshape, we don't need to unroll.
    if (shape == subShape) {
      LDBG("shape == subshape, no unroll");
      return std::nullopt;
    }
    return computeShapeRatio(shape, subShape);
  }

  // copy the layout attribte and drops the inst_data field.
  xegpu::LayoutAttr getLaneLayoutAttr(Attribute attr) const {
    auto layout = dyn_cast_if_present<xegpu::LayoutAttr>(attr);
    if (!layout || layout.getLaneLayout() == nullptr)
      return xegpu::LayoutAttr();
    return layout.dropInstData();
  };

  SmallVector<Type> convertType(ShapedType type,
                                ArrayRef<int64_t> blockSize) const {
    auto elemTy = type.getElementType();
    Type newTy;
    // TensorDescType needs to drop the inst_data field in the layout attribute
    if (auto tdescTy = dyn_cast<xegpu::TensorDescType>(type)) {
      auto ctx = tdescTy.getContext();
      auto encoding = tdescTy.getEncoding();
      auto layout = tdescTy.getLayout();
      newTy = xegpu::TensorDescType::get(ctx, blockSize, elemTy, encoding,
                                         getLaneLayoutAttr(layout));
    } else {
      newTy = type.clone(blockSize, elemTy);
    }

    auto ratio = computeShapeRatio(type.getShape(), blockSize);
    assert(ratio && "Expecting the ratio to be valid.");
    return llvm::SmallVector<Type>(computeProduct(*ratio), newTy);
  }

  // emulate the the unpack behavior using insert_strided_slice for VectorType
  // values and unrealized_conversion_cast for TileType values.
  Value unpack(ValueRange srcs, Type destTy, llvm::ArrayRef<int64_t> blockSize,
               Location loc, PatternRewriter &rewriter) const {
    if (auto vecTy = dyn_cast<VectorType>(destTy)) {
      assert(vecTy.getRank() == 2 && blockSize.size() == 2 &&
             "Expecting blockSize size to match the rank of destTy.");
      auto shape = vecTy.getShape();
      auto zeroAttr = rewriter.getZeroAttr(vecTy.getElementType());

      Value result = rewriter.create<arith::ConstantOp>(
          loc, vecTy, DenseElementsAttr::get(vecTy, zeroAttr));
      int64_t idx = 0;
      for (int64_t i = 0; i < shape[0]; i += blockSize[0]) {
        for (int64_t j = 0; j < shape[1]; j += blockSize[1]) {
          result = rewriter.create<vector::InsertStridedSliceOp>(
              loc, srcs[idx++], result, llvm::ArrayRef<int64_t>({i, j}),
              llvm::ArrayRef<int64_t>({1, 1}));
        }
      }
      return result;
    }

    if (isa<xegpu::TensorDescType>(destTy)) {
      auto attr = NamedAttribute(rewriter.getStringAttr(unpackAttrName),
                                 rewriter.getUnitAttr());
      auto blkAttr = NamedAttribute(rewriter.getStringAttr(blockAttrName),
                                    rewriter.getDenseI64ArrayAttr(blockSize));
      auto castOp = rewriter.create<UnrealizedConversionCastOp>(
          loc, destTy, srcs, llvm::ArrayRef<NamedAttribute>({attr, blkAttr}));
      return castOp.getResult(0);
    }

    llvm_unreachable("Unexpected destTy.");
    return Value();
  }

  // emulate the the pack behavior using extract_strided_slice for VectorType
  // values and unrealized_conversion_cast for TensorDescType values.
  llvm::SmallVector<Value> pack(Value src, TypeRange destTypes,
                                llvm::ArrayRef<int64_t> blockSize, Location loc,
                                PatternRewriter &rewriter) const {
    if (auto vecTy = dyn_cast<VectorType>(src.getType())) {
      assert(vecTy.getRank() == 2 && blockSize.size() == 2 &&
             "Expecting blockSize size to match the rank of src.");
      auto shape = vecTy.getShape();
      llvm::SmallVector<Value> results;
      for (int64_t i = 0; i < shape[0]; i += blockSize[0]) {
        for (int64_t j = 0; j < shape[1]; j += blockSize[1]) {
          auto slice = rewriter.create<vector::ExtractStridedSliceOp>(
              loc, src, llvm::ArrayRef<int64_t>({i, j}), blockSize,
              llvm::ArrayRef<int64_t>({1, 1}));
          results.push_back(slice);
        }
      }
      return results;
    }

    if (isa<xegpu::TensorDescType>(src.getType())) {
      auto attr = NamedAttribute(rewriter.getStringAttr(packAttrName),
                                 rewriter.getUnitAttr());
      auto blkAttr = NamedAttribute(rewriter.getStringAttr(blockAttrName),
                                    rewriter.getDenseI64ArrayAttr(blockSize));
      auto castOp = rewriter.create<UnrealizedConversionCastOp>(
          loc, destTypes, src, llvm::ArrayRef<NamedAttribute>({attr, blkAttr}));
      return castOp.getResults();
    }

    llvm_unreachable("Unexpected src type.");
    return llvm::SmallVector<Value>();
  }

private:
  const char *const packAttrName = "__xetile_blocking_pack__";
  const char *const unpackAttrName = "__xetile_blocking_unpack__";
  const char *const blockAttrName = "__xetile_blocking_inner_block__";

  xegpu::UnrollOptions options;
};

struct UnrollCreateNdOp : public UnrollPattern<xegpu::CreateNdDescOp> {
  using UnrollPattern<xegpu::CreateNdDescOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::CreateNdDescOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = op.getContext();
    auto tdescTy = op.getType();
    auto shape = tdescTy.getShape();
    auto layout = tdescTy.getLayout();

    auto maybeTargetShape = getTargetShape(op);
    if (!maybeTargetShape)
      return failure();
    auto targetShape = *maybeTargetShape;

    auto maybeGrids = computeGrids(shape, targetShape);
    if (!maybeGrids)
      return failure();
    auto grids = *maybeGrids;

    // TODO: enable 1D block tensor desc
    if (tdescTy.getRank() != 2)
      return failure();

    auto encoding = tdescTy.getEncoding();
    auto newLayout = getLaneLayoutAttr(layout);
    auto newTdescTy = xegpu::TensorDescType::get(
        ctx, targetShape, tdescTy.getElementType(), encoding, newLayout);

    auto addi = [&](OpFoldResult a, int64_t b) -> Value {
      auto maybeInt = getConstantIntValue(a);
      if (maybeInt) {
        return rewriter.create<arith::ConstantIndexOp>(loc, *maybeInt + b);
      } else {
        auto aV = llvm::cast<Value>(a);
        auto bV = rewriter.create<arith::ConstantIndexOp>(loc, b);
        return rewriter.createOrFold<arith::AddIOp>(loc, aV, bV);
      }
    };

    auto mixedOffsets = op.getMixedOffsets();
    // For n-D memrefs where n > 2, we need to handle the last two
    // dimensions, and keep the first n-2 dimensions as is.
    int64_t x = mixedOffsets.size() - 2;
    int64_t y = mixedOffsets.size() - 1;
    OpFoldResult oldX = mixedOffsets[x];
    OpFoldResult oldY = mixedOffsets[y];

    SmallVector<Value> newOps;
    for (int64_t i = 0; i < grids[0]; i++) {
      for (int64_t j = 0; j < grids[1]; j++) {
        auto subOffX = targetShape[0] * i;
        auto subOffY = targetShape[1] * j;
        mixedOffsets[x] = addi(oldX, subOffX);
        mixedOffsets[y] = addi(oldY, subOffY);
        auto newOp = rewriter.create<xegpu::CreateNdDescOp>(
            loc, newTdescTy, op.getSource(), mixedOffsets, op.getMixedSizes(),
            op.getMixedStrides());
        newOps.push_back(newOp);
      }
    }
    auto castOp = unpack(newOps, tdescTy, targetShape, loc, rewriter);
    rewriter.replaceOp(op, castOp);

    return success();
  }
};

struct UnrollUpdateNdOffsetOp : public UnrollPattern<xegpu::UpdateNdOffsetOp> {
  using UnrollPattern<xegpu::UpdateNdOffsetOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::UpdateNdOffsetOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto tdesc = op.getTensorDesc();
    auto tdescTy = tdesc.getType();
    auto shape = tdescTy.getShape();

    auto maybeTargetShape = getTargetShape(op);
    if (!maybeTargetShape)
      return failure();
    auto targetShape = *maybeTargetShape;

    auto maybeGrids = computeGrids(shape, targetShape);
    if (!maybeGrids)
      return failure();
    auto grids = *maybeGrids;

    auto convertedTdescTypes = convertType(tdescTy, targetShape);
    auto convertedTdesc =
        pack(tdesc, convertedTdescTypes, targetShape, loc, rewriter);

    llvm::SmallVector<Value> newOps;
    for (auto t : convertedTdesc) {
      auto newOp = rewriter.create<xegpu::UpdateNdOffsetOp>(
          loc, t.getType(), t, op.getOffsets(), op.getConstOffsets());
      newOps.push_back(newOp);
    }
    auto castOp = unpack(newOps, op.getType(), targetShape, loc, rewriter);
    rewriter.replaceOp(op, castOp);
    return success();
  }
};

struct UnrollPrefetchNdOp : public UnrollPattern<xegpu::PrefetchNdOp> {
  using UnrollPattern<xegpu::PrefetchNdOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::PrefetchNdOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto tdesc = op.getTensorDesc();
    auto tdescTy = tdesc.getType();
    auto shape = tdescTy.getShape();

    auto maybeTargetShape = getTargetShape(op);
    if (!maybeTargetShape)
      return failure();
    auto targetShape = *maybeTargetShape;

    auto maybeGrids = computeGrids(shape, targetShape);
    if (!maybeGrids)
      return failure();
    auto grids = *maybeGrids;

    auto convertedTdescTypes = convertType(tdescTy, targetShape);
    auto convertedTdesc =
        pack(tdesc, convertedTdescTypes, targetShape, loc, rewriter);

    for (auto t : convertedTdesc) {
      rewriter.create<xegpu::PrefetchNdOp>(loc, TypeRange(), t, op->getAttrs());
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct UnrollLoadNdOp : public UnrollPattern<xegpu::LoadNdOp> {
  using UnrollPattern<xegpu::LoadNdOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::LoadNdOp op,
                                PatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto valueTy = op.getType();
    auto tdescTy = op.getTensorDescType();

    // TODO: enable 1D block tensor desc
    if (tdescTy.getRank() != 2)
      return failure();

    auto maybeTargetShape = getTargetShape(op);
    if (!maybeTargetShape)
      return failure();
    auto targetShape = *maybeTargetShape;

    auto maybeGrids = computeGrids(tdescTy.getShape(), targetShape);
    if (!maybeGrids)
      return failure();
    auto grids = *maybeGrids;

    auto elemTy = tdescTy.getElementType();
    auto newValueTy = valueTy.cloneWith(targetShape, elemTy);

    auto convertedTdescTypes = convertType(tdescTy, targetShape);
    auto convertedTdescs = pack(op.getTensorDesc(), convertedTdescTypes,
                                targetShape, loc, rewriter);

    llvm::SmallVector<Value> newOps;
    for (auto t : convertedTdescs) {
      auto newOp =
          rewriter.create<xegpu::LoadNdOp>(loc, newValueTy, t, op->getAttrs());
      newOps.push_back(newOp);
    }

    auto castOp = unpack(newOps, op.getType(), targetShape, loc, rewriter);

    rewriter.replaceOp(op, castOp);
    return success();
  }
};

struct UnrollStoreNdOp : public UnrollPattern<xegpu::StoreNdOp> {
  using UnrollPattern<xegpu::StoreNdOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::StoreNdOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto valueTy = op.getValueType();
    auto tdescTy = op.getTensorDescType();

    // TODO: enable 1D block tensor desc
    if (tdescTy.getRank() != 2)
      return failure();

    auto maybeTargetShape = getTargetShape(op);
    if (!maybeTargetShape)
      return failure();
    auto targetShape = *maybeTargetShape;

    auto maybeGrids = computeGrids(tdescTy.getShape(), targetShape);
    if (!maybeGrids)
      return failure();
    auto grids = *maybeGrids;

    auto convertedValTypes = convertType(valueTy, targetShape);
    auto convertedTdescTypes = convertType(tdescTy, targetShape);

    auto convertedValues =
        pack(op.getValue(), convertedValTypes, targetShape, loc, rewriter);
    auto convertedTdescs = pack(op.getTensorDesc(), convertedTdescTypes,
                                targetShape, loc, rewriter);

    for (auto [v, t] : llvm::zip(convertedValues, convertedTdescs)) {
      rewriter.create<xegpu::StoreNdOp>(loc, v, t, op.getL1HintAttr(),
                                        op.getL2HintAttr(), op.getL3HintAttr());
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct UnrollDpasOp : public UnrollPattern<xegpu::DpasOp> {
  using UnrollPattern<xegpu::DpasOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::DpasOp op,
                                PatternRewriter &rewriter) const override {

    auto loc = op.getLoc();

    // a vector of 3 elements should be returned, representing M, K, N
    // respectively.
    auto maybeTargetShape = getTargetShape(op);
    if (!maybeTargetShape || maybeTargetShape->size() != 3)
      return failure();
    auto M = (*maybeTargetShape)[0];
    auto K = (*maybeTargetShape)[1];
    auto N = (*maybeTargetShape)[2];

    int64_t aBlockSize[2] = {M, K};
    int64_t bBlockSize[2] = {K, N};
    int64_t cBlockSize[2] = {M, N};

    auto packWrapper = [&](TypedValue<VectorType> val,
                           llvm::ArrayRef<int64_t> blockSize) {
      VectorType type = val.getType();
      auto maybeGrids = computeShapeRatio(type.getShape(), blockSize);
      assert(maybeGrids && "Expecting grids to be computed.");
      auto grids = *maybeGrids;
      auto numNewOps = computeProduct(grids);
      if (numNewOps == 1)
        return llvm::SmallVector<Value>({val});
      auto newVecTy = type.cloneWith(blockSize, type.getElementType());
      llvm::SmallVector<Type> convertedTypes(numNewOps, newVecTy);
      auto values = pack(val, convertedTypes, blockSize, loc, rewriter);
      return llvm::to_vector(values);
    };

    auto a = op.getLhs();
    auto b = op.getRhs();
    auto c = op.getAcc();

    auto aShape = a.getType().getShape();
    auto bShape = b.getType().getShape();

    llvm::SmallVector<Value> aVals, bVals, cVals;
    aVals = packWrapper(a, aBlockSize);
    bVals = packWrapper(b, bBlockSize);

    if (c)
      cVals = packWrapper(c, cBlockSize);

    // Vals are empty due to invalid blocking size, or with size 1 due to
    // the original shape is the same with the blocking size. The op will
    // be skipped if every operand got an invalid blocking size or the
    // original shape is the same with the blocking size.
    if (aVals.size() <= 1 && bVals.size() <= 1 && cVals.size() <= 1)
      return failure();

    auto resultTy = op.getResult().getType();
    auto vecTy = VectorType::get(cBlockSize, resultTy.getElementType());

    auto mIters = aShape[0] / M;
    auto kIters = aShape[1] / K;
    auto nIters = bShape[1] / N;

    SmallVector<Value> newOps;
    for (int64_t i = 0; i < mIters; i++) {
      for (int64_t j = 0; j < nIters; j++) {
        Value tmpC;
        if (c)
          tmpC = cVals[i * nIters + j]; // init with acc
        for (int64_t k = 0; k < kIters; k++) {
          auto aVec = aVals[i * kIters + k];
          auto bVec = bVals[k * nIters + j];
          llvm::SmallVector<Value> operands({aVec, bVec});
          if (tmpC)
            operands.push_back(tmpC);
          tmpC = rewriter.create<xegpu::DpasOp>(loc, vecTy, operands,
                                                op->getAttrs());
        }
        newOps.push_back(tmpC);
      }
    }
    auto castOp = unpack(newOps, resultTy, cBlockSize, loc, rewriter);
    rewriter.replaceOp(op, castOp);
    return success();
  }
};

} // namespace

void mlir::xegpu::populateXeGPUUnrollPatterns(
    RewritePatternSet &patterns, const xegpu::UnrollOptions &options) {
  patterns.add<UnrollCreateNdOp, UnrollUpdateNdOffsetOp, UnrollPrefetchNdOp,
               UnrollLoadNdOp, UnrollStoreNdOp, UnrollDpasOp>(
      patterns.getContext(), options);
}
