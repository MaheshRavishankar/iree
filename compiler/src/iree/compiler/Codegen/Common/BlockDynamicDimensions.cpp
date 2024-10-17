// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/TensorDynamicDimAnalysis.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-block-dynamic-dimensions"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_BLOCKDYNAMICDIMENSIONSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

using TensorDivisibilityInfo =
    llvm::SmallDenseMap<unsigned, IREE::Util::ConstantIntDivisibility>;

namespace {

struct BlockDynamicDimensionsPass final
    : impl::BlockDynamicDimensionsPassBase<BlockDynamicDimensionsPass> {
  void runOnOperation() override;
};
} // namespace

static TensorDivisibilityInfo
getTensorDivisibilityInfo(const TensorDynamicDimAnalysis &dynamicDimAnalysis,
                          Value v) {
  TensorDivisibilityInfo divisibilityInfo;
  auto tensorType = dyn_cast<RankedTensorType>(v.getType());
  if (!tensorType) {
    return divisibilityInfo;
  }

  for (auto [index, dim] : llvm::enumerate(tensorType.getShape())) {
    if (!tensorType.isDynamicDim(index))
      continue;
    std::optional<IREE::Util::ConstantIntDivisibility> dimDivisibility =
        dynamicDimAnalysis.getDivisibilityInfo(v, index);
    if (!dimDivisibility)
      continue;
    divisibilityInfo[index] = std::move(dimDivisibility.value());
  }

  return divisibilityInfo;
}

static std::optional<Value>
blockDynamicDimensionsOfValue(RewriterBase &rewriter,
                              const TensorDivisibilityInfo &divisibilityInfo,
                              Value v) {
  auto tensorType = dyn_cast<RankedTensorType>(v.getType());
  if (!tensorType) {
    return std::nullopt;
  }

  // Check if we know that the operands have a divisibility information.
  SmallVector<OpFoldResult> outputShape;
  SmallVector<ReassociationIndices> reassociation;
  Location loc = v.getLoc();

  for (auto [index, dim] : llvm::enumerate(tensorType.getShape())) {
    reassociation.emplace_back(ReassociationIndices{});

    // Check if this needs division.
    if (!tensorType.isDynamicDim(index) || !divisibilityInfo.contains(index)) {
      reassociation.back().push_back(outputShape.size());
      outputShape.push_back(rewriter.getIndexAttr(dim));
      continue;
    }

    // Split the dynamic based on the divisibility info.
    IREE::Util::ConstantIntDivisibility currDivisibility =
        divisibilityInfo.lookup(index);
    uint64_t factor = currDivisibility.sdiv();
    AffineExpr s0 = rewriter.getAffineSymbolExpr(0);
    AffineExpr divExpr = s0.floorDiv(factor);
    Value sourceDim = rewriter.create<tensor::DimOp>(loc, v, index).getResult();
    OpFoldResult newDynamicDim = affine::makeComposedFoldedAffineApply(
        rewriter, loc, divExpr, ArrayRef<OpFoldResult>{sourceDim});
    OpFoldResult newStaticDim = rewriter.getIndexAttr(factor);

    reassociation.back().push_back(outputShape.size());
    reassociation.back().push_back(outputShape.size() + 1);

    outputShape.push_back(newDynamicDim);
    outputShape.push_back(newStaticDim);
  }

  auto staticOutputShape =
      llvm::map_to_vector(outputShape, [](OpFoldResult ofr) {
        if (auto staticShapeAttr = dyn_cast<Attribute>(ofr)) {
          return cast<IntegerAttr>(staticShapeAttr).getInt();
        }
        return ShapedType::kDynamic;
      });
  auto outputType = RankedTensorType::get(
      staticOutputShape, tensorType.getElementType(), tensorType.getEncoding());

  Value expandShape = rewriter.create<tensor::ExpandShapeOp>(
      loc, outputType, v, reassociation, outputShape);
  Value barrier =
      rewriter.create<IREE::Util::OptimizationBarrierOp>(loc, expandShape)
          .getResult(0);
  Value collapseShape = rewriter.create<tensor::CollapseShapeOp>(
      loc, tensorType, barrier, reassociation);
  return collapseShape;
}

static LogicalResult
blockDynamicDimensions(RewriterBase &rewriter,
                       const TensorDynamicDimAnalysis &dynamicDimAnalysis,
                       IREE::LinalgExt::AttentionOp attentionOp) {
  OpBuilder::InsertionGuard g(rewriter);

  bool addedReshape = false;
  for (auto operand : attentionOp->getOperands()) {
    if (operand.getDefiningOp<tensor::CollapseShapeOp>())
      continue;
    TensorDivisibilityInfo operandDivisibilityInfo =
        getTensorDivisibilityInfo(dynamicDimAnalysis, operand);
    if (operandDivisibilityInfo.empty())
      continue;
    std::optional<Value> newOperand = blockDynamicDimensionsOfValue(
        rewriter, operandDivisibilityInfo, operand);
    if (newOperand) {
      addedReshape = true;
      rewriter.modifyOpInPlace(attentionOp, [&]() {
        attentionOp.getMaskMutable().assign(newOperand.value());
      });
    }
  }
  return success(addedReshape);
}

void BlockDynamicDimensionsPass::runOnOperation() {
  Operation *operation = getOperation();
  MLIRContext *context = &getContext();
  TensorDynamicDimAnalysis dynamicDimAnalysis(operation);
  if (failed(dynamicDimAnalysis.run())) {
    return signalPassFailure();
  }

  IRRewriter rewriter(context);
  auto walkResult = operation->walk(
      [&](IREE::LinalgExt::AttentionOp attentionOp) -> WalkResult {
        rewriter.setInsertionPoint(attentionOp);
        return blockDynamicDimensions(rewriter, dynamicDimAnalysis,
                                      attentionOp);
      });
  if (walkResult.wasInterrupted()) {
    return signalPassFailure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "After blocking dimensions:\n";
    operation->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n";
  });

  {
    RewritePatternSet bubbleExpandShapePatterns(context);
    linalg::ControlFusionFn controlFn = [](OpOperand *) { return true; };
    linalg::populateFoldReshapeOpsByExpansionPatterns(bubbleExpandShapePatterns,
                                                      controlFn);
    IREE::LinalgExt::populateFoldReshapeOpsByExpansionPatterns(
        bubbleExpandShapePatterns, controlFn);
    populateReshapeToInterfaceTensorPatterns(bubbleExpandShapePatterns);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(
        bubbleExpandShapePatterns);
    tensor::populateFoldTensorEmptyPatterns(bubbleExpandShapePatterns);
    GreedyRewriteConfig config;
    // config.applyFold = false;
    if (failed(applyPatternsAndFoldGreedily(
            operation, std::move(bubbleExpandShapePatterns), config))) {
      operation->emitOpError(
          "failed in application of bubble up expand shape patterns");
      return signalPassFailure();
    }
  }

  // // Apply some cleanup patterns
  // {
  //   RewritePatternSet cleanupPatterns(context);
  //   populateReshapeToInterfaceTensorPatterns(cleanupPatterns);
  //   memref::populateResolveRankedShapedTypeResultDimsPatterns(cleanupPatterns);
  //   tensor::populateFoldTensorEmptyPatterns(cleanupPatterns);
  //   if (failed(applyPatternsAndFoldGreedily(operation,
  //                                           std::move(cleanupPatterns)))) {
  //     operation->emitOpError(
  //         "failed in application of bubble up expand shape patterns");
  //     return signalPassFailure();
  //   }
  // }

  return;
}

} // namespace mlir::iree_compiler
