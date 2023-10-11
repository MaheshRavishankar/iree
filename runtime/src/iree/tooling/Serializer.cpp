#include "Serializer.h"
#include "llvm/Support/Debug.h"

namespace mlir {

Serializer::Serializer(mlir::ModuleOp module)
    : module(module), mlirBuilder(module.getContext()) {}

LogicalResult Serializer::serialize() {
  LLVM_DEBUG(llvm::dbgs() << "+++ starting serialization +++\n");

  if (failed(module.verifyInvariants()))
    return failure();

  // Iterate over the module body to serialize it. Assumptions are that there is
  // only one basic block in the moduleOp
  for (auto &op : *module.getBody()) {
    if (failed(processOperation(&op))) {
      return failure();
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "+++ completed serialization +++\n");
  return success();
}

LogicalResult Serializer::processOperation(Operation *opInst) {
  LLVM_DEBUG(llvm::dbgs() << "[op] '" << opInst->getName() << "'\n");

  return success();
}

} // namespace mlir
