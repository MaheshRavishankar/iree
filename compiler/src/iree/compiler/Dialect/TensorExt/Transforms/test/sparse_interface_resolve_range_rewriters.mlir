// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-tensor-ext-test-sparse-op-interface-methods{test-resolve-range=true}))' %s --split-input-file --mlir-print-local-scope | FileCheck %s

// Test vector.transfer_read rewriting for cast_to_ragged_shape with ragged_dim(0)
func.func @test(%arg0 : index, %arg1: index, %arg2 : index, %num_ragged_rows : index,
    %source : memref<?x512xf16>, %column_lengths : memref<?xi32>, %d0 : index)
    -> vector<1x1x4xf16> {
  %cst = arith.constant 0.0 : f16
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(0)
      column_lengths(%column_lengths) num_ragged_rows(%num_ragged_rows)
      : (memref<?x512xf16>{%d0}, memref<?xi32>) -> memref<?x?x512xf16, #iree_tensor_ext.ragged_tensor<0>>
  %1 = vector.transfer_read %0[%arg0, %arg1, %arg2], %cst
      {in_bounds = [true, true, true], permutation_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}
      : memref<?x?x512xf16, #iree_tensor_ext.ragged_tensor<0>>, vector<1x1x4xf16>
  return %1 : vector<1x1x4xf16>
}
// CHECK-LABEL: func @test
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[NUM_ROWS:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[SOURCE:[a-zA-Z0-9]+]]: memref<?x512xf16>
//  CHECK-SAME:     %[[COLUMN_LENGTHS:[a-zA-Z0-9]+]]: memref<?xi32>
//       CHECK:   %[[CST:.+]] = arith.constant 0.000000e+00 : f16
//       CHECK:   %[[COLUMN_LENGTH_I32:.+]] = memref.load %[[COLUMN_LENGTHS]][%[[ARG0]]]
//       CHECK:   %[[COLUMN_LENGTH:.+]] = arith.index_cast %[[COLUMN_LENGTH_I32]]
//       CHECK:   %[[RESULT:.+]] = vector.transfer_read %[[SOURCE]][%[[COLUMN_LENGTH]], %[[ARG2]]], %[[CST]] {in_bounds = [true, true, true], permutation_map = affine_map<(d0, d1) -> (0, 0, d1)>} : memref<?x512xf16>, vector<1x1x4xf16>
//       CHECK:   return %[[RESULT]]
