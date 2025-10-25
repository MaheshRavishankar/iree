// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::TensorExt {

namespace {

/// Pattern to rewrite vector.transfer_read operations that read from a memref
/// defined by a cast_to_ragged_shape op. The transfer_read is rewritten to
/// read from the source of the cast_to_ragged_shape using the resolveRange
/// interface method.
struct RewriteTransferReadFromRaggedShape
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp transferReadOp,
                                PatternRewriter &rewriter) const override {
    // Check if the base is defined by an op implementing SparseOpInterface.
    auto sparseOp = transferReadOp.getBase()
                        .getDefiningOp<IREE::TensorExt::SparseOpInterface>();
    if (!sparseOp) {
      return rewriter.notifyMatchFailure(
          transferReadOp, "base not defined by SparseOpInterface");
    }

    // Only handle identity permutation maps for now.
    if (!transferReadOp.getPermutationMap().isIdentity()) {
      return rewriter.notifyMatchFailure(
          transferReadOp, "non-identity permutation map not supported");
    }

    Location loc = transferReadOp.getLoc();
    VectorType vectorType = transferReadOp.getVectorType();
    ArrayRef<int64_t> vectorShape = vectorType.getShape();

    // Build the Range for each dimension of the result.
    SmallVector<Range> givenRanges;
    ValueRange indices = transferReadOp.getIndices();

    // Build ranges: offset = index, size = vector dimension size, stride = 1
    for (auto [index, dimSize] : llvm::zip_equal(indices, vectorShape)) {
      OpFoldResult offset = index;
      OpFoldResult size = rewriter.getIndexAttr(dimSize);
      OpFoldResult stride = rewriter.getIndexAttr(1);
      givenRanges.push_back(Range{offset, size, stride});
    }

    // Call resolveRange to get the ranges in the source space.
    // If in_bounds is set to all true, we don't need to ensure bounds checking.
    bool ensureInBounds =
        !transferReadOp.getInBoundsAttr() ||
        llvm::any_of(transferReadOp.getInBounds(), [](Attribute attr) {
          return !cast<BoolAttr>(attr).getValue();
        });
    FailureOr<SmallVector<Range>> resolvedRanges =
        sparseOp.resolveRange(rewriter, givenRanges, ensureInBounds);
    if (failed(resolvedRanges)) {
      return rewriter.notifyMatchFailure(
          transferReadOp, "failed to resolve range for sparse op");
    }

    // Verify that all resolved ranges have stride = 1.
    if (llvm::any_of(resolvedRanges.value(), [](const Range &range) {
          std::optional<int64_t> strideVal = getConstantIntValue(range.stride);
          return !strideVal || strideVal.value() != 1;
        })) {
      return rewriter.notifyMatchFailure(transferReadOp,
                                         "resolved range has non-unit stride");
    }

    // Extract the new offsets from the resolved ranges.
    SmallVector<Value> newIndices =
        llvm::map_to_vector(resolvedRanges.value(), [&](const Range &range) {
          return getValueOrCreateConstantIndexOp(rewriter, loc, range.offset);
        });

    // Create a new transfer_read with the source of the sparse op
    // and the resolved offsets.
    auto castOp =
        cast<IREE::TensorExt::CastToRaggedShapeOp>(sparseOp.getOperation());

    // Create new permutation map. Since we're collapsing sparse dimensions,
    // we need to map from the source's rank to the original vector's rank.
    // For dimensions that were collapsed, use constant 0 (since we only
    // support size-1 slices in those dimensions).
    auto sourceType = cast<MemRefType>(castOp.getSource().getType());
    unsigned sourceRank = sourceType.getRank();
    unsigned vectorRank = vectorShape.size();

    // Get the sparse dimensions from the result
    SmallVector<int64_t> sparseDims =
        castOp.getResultSparseEncoding().getSparseDimensions();
    int64_t outerSparseDim = sparseDims[0];

    // Build affine map: dimensions before sparse -> identity,
    // collapsed sparse dims -> constant 0, dimensions after -> identity with
    // offset
    SmallVector<AffineExpr> exprs;
    unsigned sourceDimIdx = 0;
    for (unsigned resultDim = 0; resultDim < vectorRank; ++resultDim) {
      if (resultDim < outerSparseDim) {
        // Non-sparse dimensions before the sparse block
        exprs.push_back(rewriter.getAffineDimExpr(sourceDimIdx++));
      } else if (resultDim == outerSparseDim) {
        // First collapsed sparse dimension - uses first source dim but
        // broadcasts to constant 0
        exprs.push_back(rewriter.getAffineConstantExpr(0));
        sourceDimIdx++; // This consumed one source dimension (the linearized
                        // sparse dim)
      } else if (resultDim == outerSparseDim + 1) {
        // Second collapsed sparse dimension - also broadcasts to constant 0
        exprs.push_back(rewriter.getAffineConstantExpr(0));
      } else {
        // Non-sparse dimensions after the sparse block
        exprs.push_back(rewriter.getAffineDimExpr(sourceDimIdx++));
      }
    }
    auto newPermutationMap =
        AffineMap::get(sourceRank, 0, exprs, rewriter.getContext());

    Value newTransferRead = rewriter.create<vector::TransferReadOp>(
        loc, vectorType, castOp.getSource(), newIndices, newPermutationMap,
        transferReadOp.getPadding(), transferReadOp.getMask(),
        transferReadOp.getInBoundsAttr());

    rewriter.replaceOp(transferReadOp, newTransferRead);
    return success();
  }
};

} // namespace

void populateSparseInterfaceRewritePatterns(RewritePatternSet &patterns) {
  patterns.add<RewriteTransferReadFromRaggedShape>(patterns.getContext());
}

} // namespace mlir::iree_compiler::IREE::TensorExt
