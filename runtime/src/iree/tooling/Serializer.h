#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
class Serializer {
public:
  /// Creates a serializer for the given `module`.
  explicit Serializer(mlir::ModuleOp module);

  /// Serializes the remembered module.
  LogicalResult serialize();

  /// Collects the final `binary`.
  void collect(SmallVectorImpl<uint32_t> &binary);

  /// Main dispatch method for serializing an operation.
  LogicalResult processOperation(Operation *op);

private:
  /// The accel module to be serialized.
  mlir::ModuleOp module;

  /// An MLIR builder for getting MLIR constructs.
  mlir::Builder mlirBuilder;
};
} // namespace mlir
