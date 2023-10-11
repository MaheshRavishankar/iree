#include "Serializer.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "iree-translate"

namespace mlir {

Serializer::Serializer(mlir::ModuleOp module)
    : module(module), mlirBuilder(module.getContext()) {}

LogicalResult Serializer::serialize() {
  LLVM_DEBUG(llvm::dbgs() << "+++ starting serialization +++\n");

  if (failed(module.verifyInvariants())) return failure();

  // Iterate over the module body to serialize it. Assumptions are that there is
  // only one basic block in the moduleOp
  for (auto &op : *module.getBody()) {
    if (failed(processOperation(&op, globalScope))) {
      return failure();
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "+++ completed serialization +++\n");
  return success();
}

void Serializer::collect(SmallVector<char> &binary) {
  std::swap(binary, globalScope.buffer);
  globalScope.symbolTable.clear();
}

LogicalResult Serializer::processOperation(Operation *opInst,
                                           ScopeInfo &scope) {
  LLVM_DEBUG(llvm::dbgs() << "[op] '" << opInst->getName() << "'\n");

  auto status = TypeSwitch<Operation *, LogicalResult>(opInst)
                    .Case<func::FuncOp>(
                        [&](auto funcOp) { return processFunc(funcOp, scope); })
                    .Default([&](Operation *op) {
                      return opInst->emitOpError(
                          "unhandled operation during serialization");
                    });

  return status;
}

LogicalResult Serializer::processFunc(func::FuncOp funcOp, ScopeInfo &scope) {
  auto fnStr = Twine(funcOp.getName()) + Twine('\n');
  fnStr.toVector(scope.buffer);
  return success();
}

}  // namespace mlir
