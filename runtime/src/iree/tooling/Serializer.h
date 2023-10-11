#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class Serializer {
 public:
  /// Creates a serializer for the given `module`.
  explicit Serializer(mlir::ModuleOp module);

  // Data structure to carry the current scope information.
  struct ScopeInfo {
    // Current buffer to write into.
    SmallVector<char> buffer;

    // Current symbol table.
    DenseMap<Value, std::string> symbolTable;
  };

  /// Serializes the remembered module.
  LogicalResult serialize();

  /// Collects the final `binary`.
  void collect(SmallVector<char> &binary);

  /// Main dispatch method for serializing an operation.
  LogicalResult processOperation(Operation *op, ScopeInfo &scope);

 private:
  /// The accel module to be serialized.
  mlir::ModuleOp module;

  /// An MLIR builder for getting MLIR constructs.
  mlir::Builder mlirBuilder;

  /// Methods to process individual functions.
  LogicalResult processFunc(func::FuncOp funcOp, ScopeInfo &scope);

  /// GLobal scope.
  ScopeInfo globalScope;
};
}  // namespace mlir
