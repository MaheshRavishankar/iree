// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/TensorDynamicDimAnalysis.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_BLOCKDYNAMICDIMENSIONSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

struct BlockDynamicDimensionsPass final
    : impl::BlockDynamicDimensionsPassBase<BlockDynamicDimensionsPass> {
  void runOnOperation() override;
};
} // namespace

void BlockDynamicDimensionsPass::runOnOperation() {
  Operation *operation = getOperation();
  TensorDynamicDimAnalysis dynamicDimAnalysis(operation);
  if (failed(dynamicDimAnalysis.run())) {
    return signalPassFailure();
  }
  return;
}

} // namespace mlir::iree_compiler
