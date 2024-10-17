// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"

namespace mlir::iree_compiler {

// Analysis to compute information about dynamic dimensions of tensors.
class TensorDynamicDimAnalysis {
public:
  explicit TensorDynamicDimAnalysis(Operation *rootOperation);

  LogicalResult run();

  using TensorDimDivisibilityInfo =
      DenseMap<std::tuple<Value, unsigned>,
               IREE::Util::ConstantIntDivisibility>;
  using TensorDimRangeInfo =
      DenseMap<std::tuple<Value, unsigned>, ConstantIntRanges>;

private:
  DataFlowSolver solver;

  // Operation scope within which the analysis is run.
  Operation *rootOperation;

  // Map of tensor value to integer divisibility information for each dimension.
  TensorDimDivisibilityInfo divisibilityInfo;
  TensorDimRangeInfo rangeInfo;
};

} // namespace mlir::iree_compiler