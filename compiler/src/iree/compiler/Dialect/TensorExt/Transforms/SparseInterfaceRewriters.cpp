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

/// Check if the permutation map drops the sparse dimensions.
/// Returns true if none of the sparse dimensions appear in the permutation map.
static bool permutationMapDropsSparseDims(AffineMap permutationMap,
                                          ArrayRef<int64_t> sparseDims) {
  for (int64_t sparseDim : sparseDims) {
    for (AffineExpr expr : permutationMap.getResults()) {
      if (expr.isFunctionOfDim(sparseDim)) {
        return false;
      }
    }
  }
  return true;
}

/// Build a new permutation map for the source memref by adjusting dimension
/// indices. Assumes sparse dimensions are contiguous.
/// Sparse dimensions are collapsed (N contiguous sparse dims become 1
/// linearized dim), so dimension indices need adjustment:
/// - Dims before sparse block: keep same indices
/// - Sparse dims: shouldn't appear (verified earlier)
/// - Dims after sparse block: shift down by (numSparseDims - 1)
static AffineMap buildSourcePermutationMap(OpBuilder &builder,
                                           AffineMap originalMap,
                                           ArrayRef<int64_t> sparseDims,
                                           unsigned sourceRank) {
  assert(!sparseDims.empty() && "Expected at least one sparse dimension");

  // Assert that sparse dimensions are contiguous
  for (size_t i = 1; i < sparseDims.size(); ++i) {
    assert(sparseDims[i] == sparseDims[i - 1] + 1 &&
           "Expected sparse dimensions to be contiguous");
  }

  // Assume sparse dimensions are contiguous starting at sparseDims[0]
  int64_t firstSparseDim = sparseDims[0];
  int64_t numSparseDims = sparseDims.size();

  // Build replacement expressions for each dimension in the original map
  SmallVector<AffineExpr> dimReplacements;
  for (unsigned i = 0; i < originalMap.getNumDims(); ++i) {
    if (i < firstSparseDim) {
      // Dimensions before sparse block stay the same
      dimReplacements.push_back(builder.getAffineDimExpr(i));
    } else if (i < firstSparseDim + numSparseDims) {
      // Sparse dimensions - shouldn't appear in result expressions
      // Map to dimension 0 (arbitrary, since verified not to be used)
      dimReplacements.push_back(builder.getAffineDimExpr(0));
    } else {
      // Dimensions after sparse block shift down by (numSparseDims - 1)
      // because N sparse dims become 1 linearized dim
      dimReplacements.push_back(
          builder.getAffineDimExpr(i - (numSparseDims - 1)));
    }
  }

  // Use replaceDimsAndSymbols to build the new map
  AffineMap newMap = originalMap.replaceDimsAndSymbols(
      dimReplacements, {}, sourceRank, originalMap.getNumSymbols());
  return newMap;
}

/// Build inBounds bit vector from vector.transfer_read's in_bounds attribute.
/// The in_bounds attribute is per-vector-dimension, so we need to map it
/// to per-memref-dimension using the permutation map.
static llvm::BitVector
computeInBoundsVector(AffineMap permutationMap,
                      std::optional<ArrayAttr> inBoundsAttr,
                      unsigned numMemrefDims) {
  llvm::BitVector inBounds(numMemrefDims, true);

  if (!inBoundsAttr) {
    // If no in_bounds attribute, assume all accesses may be out of bounds.
    inBounds.reset();
    return inBounds;
  }

  // Map each memref dimension to its corresponding vector dimension.
  for (unsigned idx = 0; idx < numMemrefDims; ++idx) {
    bool isInBounds = true; // Default for broadcasted dimensions (size = 1)

    // Find if this memref dimension appears in any result expression
    for (auto [resultIdx, expr] :
         llvm::enumerate(permutationMap.getResults())) {
      if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
        if (dimExpr.getPosition() == idx) {
          // This dimension maps to result dimension resultIdx
          isInBounds =
              cast<BoolAttr>(inBoundsAttr->getValue()[resultIdx]).getValue();
          break;
        }
      }
    }

    inBounds[idx] = isInBounds;
  }

  return inBounds;
}

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

    // Get the ragged dimensions from the sparse op.
    auto castOp =
        cast<IREE::TensorExt::CastToRaggedShapeOp>(sparseOp.getOperation());
    SmallVector<int64_t> sparseDims =
        castOp.getResultSparseEncoding().getSparseDimensions();

    // Check that the ragged dimensions are dropped in the permutation map.
    AffineMap permutationMap = transferReadOp.getPermutationMap();
    if (!permutationMapDropsSparseDims(permutationMap, sparseDims)) {
      return rewriter.notifyMatchFailure(
          transferReadOp,
          "ragged dimensions must be dropped in permutation map");
    }

    Location loc = transferReadOp.getLoc();
    VectorType vectorType = transferReadOp.getVectorType();
    ArrayRef<int64_t> vectorShape = vectorType.getShape();

    // Build the Range for each memref dimension based on the permutation map.
    // For each memref dimension:
    // - If it appears in the permutation map results, the size is the
    //   corresponding vector dimension
    // - If it doesn't appear (broadcasted), the size is 1
    SmallVector<Range> givenRanges;
    ValueRange indices = transferReadOp.getIndices();

    for (auto [idx, index] : llvm::enumerate(indices)) {
      OpFoldResult offset = index;
      OpFoldResult stride = rewriter.getIndexAttr(1);

      // Find if this dimension appears in any result expression
      int64_t size = 1; // Default for broadcasted dimensions
      for (auto [resultIdx, expr] :
           llvm::enumerate(permutationMap.getResults())) {
        if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
          if (dimExpr.getPosition() == idx) {
            // This dimension maps to result dimension resultIdx
            size = vectorShape[resultIdx];
            break;
          }
        }
      }

      givenRanges.push_back(Range{offset, rewriter.getIndexAttr(size), stride});
    }

    // Call resolveRange to get the ranges in the source space.
    llvm::BitVector inBounds = computeInBoundsVector(
        permutationMap, transferReadOp.getInBoundsAttr(), indices.size());

    Value paddingValue = transferReadOp.getPadding();
    FailureOr<SmallVector<Range>> resolvedRanges =
        sparseOp.resolveRange(rewriter, givenRanges, inBounds, paddingValue);
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

    // Build the new permutation map for the source memref.
    auto sourceType = cast<MemRefType>(castOp.getSource().getType());
    unsigned sourceRank = sourceType.getRank();
    AffineMap newPermutationMap = buildSourcePermutationMap(
        rewriter, permutationMap, sparseDims, sourceRank);

    Value newTransferRead = vector::TransferReadOp::create(
        rewriter, loc, vectorType, castOp.getSource(), newIndices,
        newPermutationMap, transferReadOp.getPadding(),
        transferReadOp.getMask(), transferReadOp.getInBoundsAttr());

    rewriter.replaceOp(transferReadOp, newTransferRead);
    return success();
  }
};

/// Pattern to rewrite vector.load operations that read from a memref defined
/// by a sparse op. The load is rewritten to read from the source using the
/// resolveRange interface method.
struct RewriteLoadFromRaggedShape : public OpRewritePattern<vector::LoadOp> {
  using OpRewritePattern<vector::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    // Check if the base is defined by an op implementing SparseOpInterface.
    auto sparseOp =
        loadOp.getBase().getDefiningOp<IREE::TensorExt::SparseOpInterface>();
    if (!sparseOp) {
      return rewriter.notifyMatchFailure(
          loadOp, "base not defined by SparseOpInterface");
    }

    // Get the sparse dimensions from the result's encoding using the
    // SparseTensorAttrInterface.
    auto resultType = cast<MemRefType>(loadOp.getBase().getType());
    auto encoding =
        dyn_cast_or_null<IREE::TensorExt::SparseTensorAttrInterface>(
            resultType.getLayout());
    if (!encoding) {
      return rewriter.notifyMatchFailure(
          loadOp, "result type does not have sparse encoding");
    }
    SmallVector<int64_t> sparseDims = encoding.getSparseDimensions();

    Location loc = loadOp.getLoc();
    VectorType vectorType = loadOp.getVectorType();
    ValueRange indices = loadOp.getIndices();
    unsigned memrefRank = resultType.getRank();

    // vector.load loads a vector from the last dimension.
    // Check if this is a sparse dimension - if so, we can't load a vector from
    // it.
    if (llvm::is_contained(sparseDims, static_cast<int64_t>(memrefRank - 1))) {
      return rewriter.notifyMatchFailure(
          loadOp, "cannot load vector from sparse dimension");
    }

    // Build the Range for each memref dimension.
    // For vector.load, we load a vector from the last dimension and index
    // into all other dimensions (size = 1).
    SmallVector<Range> givenRanges =
        llvm::map_to_vector(indices, [&](Value index) {
          return Range{index, rewriter.getIndexAttr(1),
                       rewriter.getIndexAttr(1)};
        });
    // Update the last dimension's size to the vector size.
    givenRanges.back().size =
        rewriter.getIndexAttr(vectorType.getNumElements());

    // Build inBounds bit vector. vector.load has no in_bounds attribute,
    // so we assume all accesses are in-bounds.
    llvm::BitVector inBounds(givenRanges.size(), true);

    // Call resolveRange to get the ranges in the source space.
    FailureOr<SmallVector<Range>> resolvedRanges = sparseOp.resolveRange(
        rewriter, givenRanges, inBounds, /*paddingValue=*/std::nullopt);
    if (failed(resolvedRanges)) {
      return rewriter.notifyMatchFailure(
          loadOp, "failed to resolve range for sparse op");
    }

    // Verify that all resolved ranges have stride = 1.
    if (llvm::any_of(resolvedRanges.value(), [](const Range &range) {
          std::optional<int64_t> strideVal = getConstantIntValue(range.stride);
          return !strideVal || strideVal.value() != 1;
        })) {
      return rewriter.notifyMatchFailure(loadOp,
                                         "resolved range has non-unit stride");
    }

    // Extract the new offsets from the resolved ranges.
    SmallVector<Value> newIndices =
        llvm::map_to_vector(resolvedRanges.value(), [&](const Range &range) {
          return getValueOrCreateConstantIndexOp(rewriter, loc, range.offset);
        });

    // Get the source memref from the sparse op's operands.
    // The first operand is typically the source memref.
    Value sourceMemref = sparseOp.getOperation()->getOperand(0);

    // Create a new vector.load reading from the source memref.
    Value newLoad = vector::LoadOp::create(rewriter, loc, vectorType,
                                           sourceMemref, newIndices);

    rewriter.replaceOp(loadOp, newLoad);
    return success();
  }
};

/// Pattern to rewrite vector.maskedload operations that read from a memref
/// defined by a sparse op. The maskedload is rewritten to read from the source
/// using the resolveRange interface method.
struct RewriteMaskedLoadFromRaggedShape
    : public OpRewritePattern<vector::MaskedLoadOp> {
  using OpRewritePattern<vector::MaskedLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MaskedLoadOp maskedLoadOp,
                                PatternRewriter &rewriter) const override {
    // Check if the base is defined by an op implementing SparseOpInterface.
    auto sparseOp = maskedLoadOp.getBase()
                        .getDefiningOp<IREE::TensorExt::SparseOpInterface>();
    if (!sparseOp) {
      return rewriter.notifyMatchFailure(
          maskedLoadOp, "base not defined by SparseOpInterface");
    }

    // Get the sparse dimensions from the result's encoding using the
    // SparseTensorAttrInterface.
    auto resultType = cast<MemRefType>(maskedLoadOp.getBase().getType());
    auto encoding =
        dyn_cast_or_null<IREE::TensorExt::SparseTensorAttrInterface>(
            resultType.getLayout());
    if (!encoding) {
      return rewriter.notifyMatchFailure(
          maskedLoadOp, "result type does not have sparse encoding");
    }
    SmallVector<int64_t> sparseDims = encoding.getSparseDimensions();

    Location loc = maskedLoadOp.getLoc();
    VectorType vectorType = maskedLoadOp.getVectorType();
    ValueRange indices = maskedLoadOp.getIndices();
    unsigned memrefRank = resultType.getRank();

    // vector.maskedload loads a vector from the last dimension.
    // Check if this is a sparse dimension - if so, we can't load a vector from
    // it.
    if (llvm::is_contained(sparseDims, static_cast<int64_t>(memrefRank - 1))) {
      return rewriter.notifyMatchFailure(
          maskedLoadOp, "cannot load vector from sparse dimension");
    }

    // Build the Range for each memref dimension.
    // For vector.maskedload, we load a vector from the last dimension and index
    // into all other dimensions (size = 1).
    SmallVector<Range> givenRanges =
        llvm::map_to_vector(indices, [&](Value index) {
          return Range{index, rewriter.getIndexAttr(1),
                       rewriter.getIndexAttr(1)};
        });
    // Update the last dimension's size to the vector size.
    givenRanges.back().size =
        rewriter.getIndexAttr(vectorType.getNumElements());

    // Build inBounds bit vector. vector.maskedload has a mask, but we assume
    // all accesses that pass the mask are in-bounds.
    llvm::BitVector inBounds(givenRanges.size(), true);

    // Call resolveRange to get the ranges in the source space.
    FailureOr<SmallVector<Range>> resolvedRanges = sparseOp.resolveRange(
        rewriter, givenRanges, inBounds, /*paddingValue=*/std::nullopt);
    if (failed(resolvedRanges)) {
      return rewriter.notifyMatchFailure(
          maskedLoadOp, "failed to resolve range for sparse op");
    }

    // Verify that all resolved ranges have stride = 1.
    if (llvm::any_of(resolvedRanges.value(), [](const Range &range) {
          std::optional<int64_t> strideVal = getConstantIntValue(range.stride);
          return !strideVal || strideVal.value() != 1;
        })) {
      return rewriter.notifyMatchFailure(maskedLoadOp,
                                         "resolved range has non-unit stride");
    }

    // Extract the new offsets from the resolved ranges.
    SmallVector<Value> newIndices =
        llvm::map_to_vector(resolvedRanges.value(), [&](const Range &range) {
          return getValueOrCreateConstantIndexOp(rewriter, loc, range.offset);
        });

    // Get the source memref from the sparse op's operands.
    // The first operand is typically the source memref.
    Value sourceMemref = sparseOp.getOperation()->getOperand(0);

    // Create a new vector.maskedload reading from the source memref, preserving
    // the mask and passThru.
    auto newMaskedLoad = vector::MaskedLoadOp::create(
        rewriter, loc, vectorType, sourceMemref, newIndices,
        maskedLoadOp.getMask(), maskedLoadOp.getPassThru());

    rewriter.replaceOp(maskedLoadOp, newMaskedLoad);
    return success();
  }
};

/// Pattern to rewrite memref.dim operations on ragged tensors.
/// For dimensions that are ragged, we use the metadata from cast_to_ragged_shape.
/// For non-ragged dimensions, we forward to the source memref.
struct RewriteDimFromRaggedShape : public OpRewritePattern<memref::DimOp> {
  using OpRewritePattern<memref::DimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::DimOp dimOp,
                                PatternRewriter &rewriter) const override {
    // Check if the source is defined by an op implementing SparseOpInterface.
    auto sparseOp =
        dimOp.getSource().getDefiningOp<IREE::TensorExt::SparseOpInterface>();
    if (!sparseOp) {
      return rewriter.notifyMatchFailure(
          dimOp, "source not defined by SparseOpInterface");
    }

    // We only handle cast_to_ragged_shape for now.
    auto castOp =
        dyn_cast<IREE::TensorExt::CastToRaggedShapeOp>(sparseOp.getOperation());
    if (!castOp) {
      return rewriter.notifyMatchFailure(dimOp,
                                         "sparse op is not cast_to_ragged_shape");
    }

    // Get the dimension index.
    std::optional<int64_t> indexOpt = getConstantIntValue(dimOp.getIndex());
    if (!indexOpt) {
      return rewriter.notifyMatchFailure(dimOp,
                                         "dimension index is not constant");
    }
    int64_t index = indexOpt.value();

    Location loc = dimOp.getLoc();
    SmallVector<int64_t> sparseDims =
        castOp.getResultSparseEncoding().getSparseDimensions();

    // Get the ragged dimension index.
    int64_t firstRaggedDim = castOp.getRaggedDim().getSExtValue();

    // Check if this is a ragged dimension.
    if (llvm::is_contained(sparseDims, index)) {
      // For ragged dimensions, return the corresponding metadata value.
      if (index == firstRaggedDim) {
        // First ragged dimension: return num_ragged_rows.
        rewriter.replaceOp(dimOp, castOp.getNumRaggedRows());
        return success();
      } else {
        // Other ragged dimensions: return avg_ragged_column_length.
        rewriter.replaceOp(dimOp, castOp.getAvgRaggedColumnLength());
        return success();
      }
    } else {
      // For non-ragged dimensions, forward to the source memref.
      // Compute the dimension index in the source memref.
      // Since ragged dimensions are collapsed (N dims -> 1 dim), we need to
      // adjust the index.
      int64_t numRaggedDims = sparseDims.size();
      int64_t sourceIndex = index;

      if (index > firstRaggedDim) {
        // Dimensions after the ragged block shift down by (numRaggedDims - 1).
        sourceIndex = index - (numRaggedDims - 1);
      }

      Value sourceDim = memref::DimOp::create(rewriter, loc, castOp.getSource(),
                                               sourceIndex);
      rewriter.replaceOp(dimOp, sourceDim);
      return success();
    }
  }
};

} // namespace

void populateSparseInterfaceRewritePatterns(RewritePatternSet &patterns) {
  patterns.add<RewriteTransferReadFromRaggedShape, RewriteLoadFromRaggedShape,
               RewriteMaskedLoadFromRaggedShape, RewriteDimFromRaggedShape>(
      patterns.getContext());
}

} // namespace mlir::iree_compiler::IREE::TensorExt
