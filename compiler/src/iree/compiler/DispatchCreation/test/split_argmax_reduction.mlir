util.func public @argmax(%arg0: tensor<?x131072xbf16>, %arg1: index) -> tensor<?xi64> {
  %cst = arith.constant 0xFF80 : bf16
  %c0_i64 = arith.constant 0 : i64
  %0 = tensor.empty(%arg1) : tensor<?xbf16>
  %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<?xbf16>) -> tensor<?xbf16>
  %2 = tensor.empty(%arg1) : tensor<?xi64>
  %3 = linalg.fill ins(%c0_i64 : i64) outs(%2 : tensor<?xi64>) -> tensor<?xi64>
  %4:2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<?x131072xbf16>) outs(%1, %3 : tensor<?xbf16>, tensor<?xi64>) {
  ^bb0(%in: bf16, %out: bf16, %out_0: i64):
    %5 = linalg.index 1 : index
    %6 = arith.index_cast %5 : index to i64
    %7 = arith.maximumf %in, %out : bf16
    %8 = arith.cmpf ogt, %in, %out : bf16
    %9 = arith.select %8, %6, %out_0 : i64
    linalg.yield %7, %9 : bf16, i64
  } -> (tensor<?xbf16>, tensor<?xi64>)
  util.return %4#1 : tensor<?xi64>
}