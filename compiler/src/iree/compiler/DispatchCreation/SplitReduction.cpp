// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--------------- SplitReduction.cpp ----------------------------===//
//
// Split reduction dimension to increase parallelism of a linalg operation.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_SPLITREDUCTIONPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

// TODO(thomasraoux): Move to attributes.
static llvm::cl::opt<int64_t>
    splitMatmulReductionRatio("iree-dispatch-creation-split-matmul-reduction",
                              llvm::cl::desc("split matmul ratio"),
                              llvm::cl::init(1));

static llvm::cl::opt<int64_t> splitArgmaxReductionRatio(
    "iree-dispatch-creation-split-argmax-reduction",
    llvm::cl::desc("Ratio to split argmax. Set to 0 or 1 to disable"),
    llvm::cl::init(128));

static llvm::cl::list<int64_t> topkSplitReductionRatio(
    "iree-dispatch-creation-topk-split-reduction",
    llvm::cl::desc("comma separated list of split ratios"),
    llvm::cl::CommaSeparated);

template <typename LinalgOpTy>
static FailureOr<linalg::SplitReductionResult>
splitReductionImpl(RewriterBase &rewriter, LinalgOpTy op,
                   linalg::ControlSplitReductionFn controlSplitReductionFn) {
  return linalg::splitReduction(rewriter, op, controlSplitReductionFn);
}

template <>
FailureOr<linalg::SplitReductionResult> splitReductionImpl<linalg::GenericOp>(
    RewriterBase &rewriter, linalg::GenericOp genericOp,
    linalg::ControlSplitReductionFn controlSplitReductionFn) {
  assert(IREE::LinalgExt::isArgmaxOp(genericOp) &&
         "expected operation to be an argmax op");

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(genericOp);

  linalg::SplitReductionOptions control = controlSplitReductionFn(genericOp);
  int64_t ratio = control.ratio;
  unsigned insertSplitIndex = control.index;
  unsigned insertSplitDimension = control.index;
  if (ratio <= 1) {
    return rewriter.notifyMatchFailure(
        genericOp, "split ratio needs to be greater than 1");
  }

  SmallVector<unsigned> dims;
  genericOp.getReductionDims(dims);

  if (dims.size() != 1) {
    return rewriter.notifyMatchFailure(genericOp,
                                       "needs a single reduction dimension");
  }
  unsigned reductionDim = dims[0];
  if (control.innerParallel) {
    insertSplitDimension = reductionDim + 1;
  }
  SmallVector<int64_t, 4> loopRanges = genericOp.getStaticLoopRanges();
  int64_t reductionDimSize = loopRanges[reductionDim];
  if (reductionDimSize == ShapedType::kDynamic ||
      reductionDimSize % ratio != 0) {
    return rewriter.notifyMatchFailure(
        genericOp, "Reduction dimension not divisible by split ratio");
  }
  if (insertSplitIndex >
      genericOp.getShape(genericOp.getDpsInitOperand(0)).size()) {
    return rewriter.notifyMatchFailure(genericOp,
                                       "Insert dimension position too large "
                                       "compared to intermediate tensor size");
  }

  assert(0 && "pick the identity value");
  std::pair<Value, Value> identity = {}; // TODO: Fix identity.

  Location loc = genericOp->getLoc();
  SmallVector<Value> newInputs;
  SmallVector<AffineMap> newMaps;
  // Calculate the new shapes and indexing maps of the input operands.
  for (OpOperand *operand : genericOp.getDpsInputOperands()) {
    AffineMap map = genericOp.getMatchingIndexingMap(operand);
    SmallVector<int64_t> newShape;
    SmallVector<AffineExpr> exprs;
    SmallVector<ReassociationIndices> reassociation;
    unsigned index = 0;
    for (unsigned idx : llvm::seq<unsigned>(0, map.getNumResults())) {
      unsigned dim = map.getDimPosition(idx);
      if (reductionDim == dim) {
        if (control.innerParallel) {
          newShape.push_back(genericOp.getShape(operand)[idx] /
                             ratio); // reduce
          newShape.push_back(ratio); // parallel (insert)
          exprs.push_back(rewriter.getAffineDimExpr(
              dim < insertSplitDimension ? dim : dim + 1));
          exprs.push_back(rewriter.getAffineDimExpr(insertSplitDimension));
        } else {
          newShape.push_back(ratio); // parallel (insert)
          newShape.push_back(genericOp.getShape(operand)[idx] /
                             ratio); // reduce
          exprs.push_back(rewriter.getAffineDimExpr(insertSplitDimension));
          exprs.push_back(rewriter.getAffineDimExpr(
              dim < insertSplitDimension ? dim : dim + 1));
          reassociation.push_back({index++, index++});
          continue;
        }
        newShape.push_back(genericOp.getShape(operand)[idx]);
        exprs.push_back(rewriter.getAffineDimExpr(
            dim < insertSplitDimension ? dim : dim + 1));
        reassociation.push_back({index++});
      }
      newMaps.push_back(AffineMap::get(map.getNumDims() + 1, 0, exprs,
                                       genericOp.getContext()));
      // If the shape is unchanged the input doesn't change.
      if (newShape == genericOp.getShape(operand)) {
        newInputs.push_back(operand->get());
        continue;
      }
      Type newType = RankedTensorType::get(
          newShape,
          cast<RankedTensorType>(operand->get().getType()).getElementType());

      Value newInput = rewriter.create<tensor::ExpandShapeOp>(
          loc, newType, operand->get(), reassociation);
      newInputs.push_back(newInput);
    }

    // Calculate the new output map and shape, we insert the new dimension based
    // on the index returned by `controlSplitReductionFn`.
    SmallVector<int64_t> newOutputShape;
    AffineMap oldOutputMap =
        genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0));
    ArrayRef<int64_t> oldShape =
        genericOp.getShape(genericOp.getDpsInitOperand(0));
    SmallVector<AffineExpr> outputExpr;
    for (unsigned idx : llvm::seq<unsigned>(0, oldShape.size() + 1)) {
      if (insertSplitIndex == idx) {
        newOutputShape.push_back(ratio);
        outputExpr.push_back(rewriter.getAffineDimExpr(insertSplitDimension));
      }
      if (idx < oldShape.size()) {
        newOutputShape.push_back(oldShape[idx]);
        unsigned dim = oldOutputMap.getDimPosition(idx);
        outputExpr.push_back(rewriter.getAffineDimExpr(
            dim < insertSplitDimension ? dim : dim + 1));
      }
    }
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, newOutputShape, genericOp.getRegionOutputArgs()[0].getType());

    // Value constantOp = rewriter.create<arith::ConstantOp>(loc, *identity);
    // Value identityTensor =
    //     rewriter.create<linalg::FillOp>(op->getLoc(), constantOp,
    //     emptyOrAllocTensor)
    //         .getResult(0);

    // newMaps.push_back(AffineMap::get(oldOutputMap.getNumDims() + 1, 0,
    // outputExpr,
    //                                  op.getContext()));
  }
  // SmallVector<utils::IteratorType> newIteratorTypes;
  // for (auto [index, iteratorType] :
  //      llvm::enumerate(op.getIteratorTypesArray())) {
  //   if (insertSplitDimension == index)
  //     newIteratorTypes.push_back(utils::IteratorType::parallel);
  //   newIteratorTypes.push_back(iteratorType);
  // }
  // if (insertSplitDimension == op.getIteratorTypesArray().size()) {
  //   newIteratorTypes.push_back(utils::IteratorType::parallel);
  // }
  // // Create the new op matching the original op with an extra parallel
  // // dimension.
  // GenericOp genericOp = b.create<GenericOp>(
  //     loc, TypeRange({emptyOrAllocTensor.getType()}), newInputs,
  //     ValueRange({identityTensor}), newMaps, newIteratorTypes);
  // b.inlineRegionBefore(op->getRegion(0), genericOp.getRegion(),
  //                      genericOp.getRegion().begin());

  // // Then create a new reduction that only reduce the newly added dimension
  // // from the previous op.
  // unsigned intermRank = newOutputShape.size();
  // AffineMap inputMap = b.getMultiDimIdentityMap(intermRank);
  // SmallVector<utils::IteratorType> reductionIteratorTypes;
  // SmallVector<AffineExpr> exprs;
  // for (unsigned i : llvm::seq<unsigned>(0, intermRank)) {
  //   if (insertSplitIndex == i) {
  //     reductionIteratorTypes.push_back(utils::IteratorType::reduction);
  //   } else {
  //     exprs.push_back(b.getAffineDimExpr(i));
  //     reductionIteratorTypes.push_back(utils::IteratorType::parallel);
  //   }
  // }
  // AffineMap outputMap = AffineMap::get(intermRank, 0, exprs,
  // op.getContext()); SmallVector<AffineMap> reductionMaps = {inputMap,
  // outputMap};

  // auto reduction = b.create<GenericOp>(
  //     loc, op->getResultTypes(), ValueRange({genericOp.getResult(0)}),
  //     op.getDpsInits(), reductionMaps, reductionIteratorTypes,
  //     [reductionOp](OpBuilder &b, Location loc, ValueRange inputs) {
  //       Operation *clonedReductionOp = b.clone(*reductionOp);
  //       clonedReductionOp->setOperand(0, inputs[0]);
  //       clonedReductionOp->setOperand(1, inputs[1]);
  //       b.create<linalg::YieldOp>(loc, clonedReductionOp->getResult(0));
  //     });
  // b.replaceOp(op, reduction.getResults());

  // return SplitReductionResult{emptyOrAllocTensor.getDefiningOp(),
  //                             identityTensor.getDefiningOp<FillOp>(),
  //                             cast<LinalgOp>(genericOp.getOperation()),
  //                             reduction};
}

template <typename LinalgOpTy>
static LogicalResult
splitReductionWrapper(RewriterBase &rewriter, LinalgOpTy op,
                      linalg::ControlSplitReductionFn controlSplitReductionFn) {
  // Since user information about compilation are passed through attributes we
  // need to make sure to propagate those.
  SmallVector<NamedAttribute> prunedAttributeList =
      linalg::getPrunedAttributeList(op);

  // Do not transform the matmul ops that have encoded operands.
  auto hasEncoding = [](Type type) -> bool {
    auto rankedTensorType = dyn_cast<RankedTensorType>(type);
    return rankedTensorType && rankedTensorType.getEncoding();
  };
  if (llvm::any_of(op.getOperandTypes(), hasEncoding)) {
    return failure();
  }

  FailureOr<linalg::SplitReductionResult> result =
      splitReductionImpl(rewriter, op, controlSplitReductionFn);
  if (failed(result)) {
    return failure();
  }

  result->splitLinalgOp->setAttrs(prunedAttributeList);
  return result;
}

namespace {
struct SplitReductionPass final
    : public impl::SplitReductionPassBase<SplitReductionPass> {
  void runOnOperation() override {
    if (splitMatmulReductionRatio.getValue() <= 1 &&
        topkSplitReductionRatio.empty() &&
        splitArgmaxReductionRatio.getValue() <= 1) {
      return;
    }

    MLIRContext *context = &getContext();
    auto funcOp = getOperation();

    SmallVector<linalg::MatmulOp> matmulCandidates;
    SmallVector<IREE::LinalgExt::TopkOp> topkCandidates;
    SmallVector<linalg::GenericOp> argmaxCandidates;

    IRRewriter rewriter(context);
    funcOp->walk([&](Operation *op) {
      TypeSwitch<Operation *>(op)
          .Case<linalg::MatmulOp>([&](auto matmulOp) {
            if (splitMatmulReductionRatio > 1) {
              matmulCandidates.push_back(matmulOp);
            }
          })
          .Case<IREE::LinalgExt::TopkOp>([&](auto topkOp) {
            if (!topkSplitReductionRatio.empty()) {
              topkCandidates.push_back(topkOp);
            }
          })
          .Case<linalg::GenericOp>([&](auto genericOp) {
            if (splitArgmaxReductionRatio > 1 &&
                IREE::LinalgExt::isArgmaxOp(genericOp)) {
              argmaxCandidates.push_back(genericOp);
            }
          });
    });

    // Split matmul ops.
    auto matmulSplitReductionControlFn =
        [&](linalg::LinalgOp op) -> linalg::SplitReductionOptions {
      // For matmul make the new parallel dimension first so that it looks
      // like a batch_matmul and can follow the same codegen.
      return {int64_t(splitMatmulReductionRatio), 0, /*innerParallel=*/false};
    };
    for (auto op : matmulCandidates) {
      (void)splitReductionWrapper(rewriter, op, matmulSplitReductionControlFn);
    }

    // Split argmax ops.
    auto argmaxSplitReductionControlFn =
        [&](linalg::LinalgOp op) -> linalg::SplitReductionOptions {
      return {splitArgmaxReductionRatio, op.getNumLoops() - 1,
              /*innerParallel=*/false};
    };
    for (auto op : argmaxCandidates) {
      if (failed(splitReductionWrapper(rewriter, op,
                                       argmaxSplitReductionControlFn))) {
        op.emitOpError("failed to split argmax operation");
        return signalPassFailure();
      }
    }

    // Split topk ops.
    IREE::LinalgExt::TopkSplitReductionControlFn topkSplitReductionControlFn =
        [&](int64_t splitReductionDepth) -> int64_t {
      SmallVector<int64_t> reductionRatios(topkSplitReductionRatio.begin(),
                                           topkSplitReductionRatio.end());
      if (splitReductionDepth >= reductionRatios.size()) {
        return -1;
      } else {
        return reductionRatios[splitReductionDepth];
      }
    };
    for (auto op : topkCandidates) {
      (void)splitReduction(rewriter, op, topkSplitReductionControlFn);
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::DispatchCreation
