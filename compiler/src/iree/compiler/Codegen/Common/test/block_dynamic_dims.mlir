func.func @prefill_bs4$async_dispatch_5_transpose_4x32xDx128_f16() {
  %c0 = arith.constant 0 : index
  %c32_i64 = arith.constant 32 : i64
  %cst = arith.constant 8.837890e-02 : f16
  %0 = hal.interface.constant.load layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(1) : i32
  %2 = hal.interface.constant.load layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(2) : i32
  %3 = hal.interface.constant.load layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(3) : i32
  %4 = arith.extui %0 : i32 to i64
  %5 = arith.extui %1 : i32 to i64
  %6 = arith.shli %5, %c32_i64 : i64
  %7 = arith.ori %4, %6 : i64
  %8 = arith.index_castui %7 : i64 to index
  %9 = arith.extui %2 : i32 to i64
  %10 = arith.extui %3 : i32 to i64
  %11 = arith.shli %10, %c32_i64 : i64
  %12 = arith.ori %9, %11 : i64
  %13 = arith.index_castui %12 : i64 to index
  %14:5 = util.assume.int 
      %c0<umin = 0, umax = 0>, 
      %c0<umin = 0, umax = 0>, 
      %c0<umin = 0, umax = 0>, 
      %c0<umin = 0, umax = 0>, 
      %13<umin = 16, umax = 4080, udiv = 16>
    : index, index, index, index, index
  %15 = flow.dispatch.workload.ordinal %14#4, 0 : index
  %16 = hal.interface.binding.subspan layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%14#0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<4x?x32x128xf16>>{%15}
  %17 = hal.interface.binding.subspan layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%14#1) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<4x?x32x128xf16>>{%15}
  %18 = hal.interface.binding.subspan layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%14#2) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<4x?x32x128xf16>>{%15}
  %19 = hal.interface.binding.subspan layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(3) alignment(64) offset(%8) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<4x32x?x?xf16>>{%15, %15}
  %20 = hal.interface.binding.subspan layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(4) alignment(64) offset(%14#3) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<4x?x32x128xf16>>{%15}
  %21 = flow.dispatch.tensor.load %16, offsets = [0, 0, 0, 0], sizes = [4, %15, 32, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4x?x32x128xf16>>{%15} -> tensor<4x?x32x128xf16>
  %22 = flow.dispatch.tensor.load %17, offsets = [0, 0, 0, 0], sizes = [4, %15, 32, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4x?x32x128xf16>>{%15} -> tensor<4x?x32x128xf16>
  %23 = flow.dispatch.tensor.load %18, offsets = [0, 0, 0, 0], sizes = [4, %15, 32, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4x?x32x128xf16>>{%15} -> tensor<4x?x32x128xf16>
  %24 = flow.dispatch.tensor.load %19, offsets = [0, 0, 0, 0], sizes = [4, 32, %15, %15], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4x32x?x?xf16>>{%15, %15} -> tensor<4x32x?x?xf16>
  %25 = tensor.empty(%15) : tensor<4x?x32x128xf16>
  %26 = tensor.empty(%15) : tensor<4x32x?x128xf16>
  %27 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d1, d4)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d5, d1, d4)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d5, d1, d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> ()>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>]} ins(%21, %22, %23, %cst, %24 : tensor<4x?x32x128xf16>, tensor<4x?x32x128xf16>, tensor<4x?x32x128xf16>, f16, tensor<4x32x?x?xf16>) outs(%26 : tensor<4x32x?x128xf16>) -> tensor<4x32x?x128xf16>
  %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%27 : tensor<4x32x?x128xf16>) outs(%25 : tensor<4x?x32x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<4x?x32x128xf16>
  flow.dispatch.tensor.store %28, %20, offsets = [0, 0, 0, 0], sizes = [4, %15, 32, 128], strides = [1, 1, 1, 1] : tensor<4x?x32x128xf16> -> !flow.dispatch.tensor<writeonly:tensor<4x?x32x128xf16>>{%15}
  return
}
