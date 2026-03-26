// RUN: iree-compile --iree-hal-target-backends=llvm-cpu %s -o %t.vmfb
// RUN: iree-run-module --device=local-task --module=%t.vmfb \
// RUN:     --function=block_sparse_matmul \
// RUN:     --input="6x4xf32=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[1,0,1,0],[0,1,0,1],[2,2,2,2]]" \
// RUN:     --input="4x8xf32=[[1,0,0,1,2,0,1,0],[0,1,0,1,0,2,0,1],[0,0,1,1,1,1,0,0],[1,1,1,1,0,0,1,1]]" \
// RUN:     --input=3 \
// RUN:     --input="4xi32=[0 2 3 6]" \
// RUN:     --input=3 | \
// RUN: FileCheck %s

// Block sparse matmul e2e test.
//
// 3 ragged rows with column lengths [2, 1, 3].
// column_lengths (CSR offsets): [0, 2, 3, 6].
// lhs_source: 6x4 flat storage, rhs: 4x8, output: 3x3x8.
//
// Expected output:
//   Row 0 (2 cols): [5,6,7,10,5,7,5,6] [13,14,15,26,17,19,13,14] [0,...]
//   Row 1 (1 col):  [21,22,23,42,29,31,21,22] [0,...] [0,...]
//   Row 2 (3 cols): [1,0,1,2,3,1,1,0] [1,2,1,2,0,2,1,2] [4,4,4,8,6,6,4,4]

// CHECK-LABEL: EXEC @block_sparse_matmul
// CHECK: 3x3x8xf32=
// CHECK-SAME: {{\[}}[5 6 7 10 5 7 5 6]
// CHECK-SAME: [13 14 15 26 17 19 13 14]
// CHECK-SAME: [0 0 0 0 0 0 0 0]
// CHECK-SAME: {{\]}}
// CHECK-SAME: {{\[}}[21 22 23 42 29 31 21 22]
// CHECK-SAME: [0 0 0 0 0 0 0 0]
// CHECK-SAME: [0 0 0 0 0 0 0 0]
// CHECK-SAME: {{\]}}
// CHECK-SAME: {{\[}}[1 0 1 2 3 1 1 0]
// CHECK-SAME: [1 2 1 2 0 2 1 2]
// CHECK-SAME: [4 4 4 8 6 6 4 4]

func.func @block_sparse_matmul(
    %lhs_source : tensor<?x4xf32>, %rhs : tensor<4x8xf32>,
    %num_rows : index, %column_lengths : tensor<?xi32>,
    %max_column_length : index) -> tensor<?x?x8xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %lhs_source, %c0 : tensor<?x4xf32>

  %lhs = iree_tensor_ext.cast_to_ragged_shape %lhs_source ragged_dim(0)
      column_lengths(%column_lengths) num_ragged_rows(%num_rows)
      avg_ragged_column_length(%max_column_length)
      : (tensor<?x4xf32>{%d0}, tensor<?xi32>)
      -> tensor<?x?x4xf32, #iree_tensor_ext.ragged_shape<0>>

  %empty = tensor.empty(%num_rows, %max_column_length) : tensor<?x?x8xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<?x?x8xf32>)
      -> tensor<?x?x8xf32>

  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d3, d2)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%lhs, %rhs
          : tensor<?x?x4xf32, #iree_tensor_ext.ragged_shape<0>>,
            tensor<4x8xf32>)
      outs(%fill : tensor<?x?x8xf32>) {
    ^bb0(%a : f32, %b : f32, %c : f32):
      %mul = arith.mulf %a, %b : f32
      %add = arith.addf %c, %mul : f32
      linalg.yield %add : f32
  } -> tensor<?x?x8xf32>
  return %result : tensor<?x?x8xf32>
}
