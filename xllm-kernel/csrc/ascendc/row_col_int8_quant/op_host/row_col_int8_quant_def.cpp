// Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.

#include "register/op_def_registry.h"
#include "row_col_int8_quant_tiling.h"

namespace {
bool Valid(const gert::Shape& shape) {
  return shape.GetDimNum() == 2 && shape.GetDim(0) >= 1 &&
         shape.GetDim(0) <= 4096 && shape.GetDim(1) >= 2 &&
         shape.GetDim(1) <= 4096 && shape.GetDim(1) % 2 == 0;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  const auto& shape = context->GetInputShape(0)->GetStorageShape();
  if (!Valid(shape)) {
    return ge::GRAPH_FAILED;
  }
  auto* tiling = context->GetTilingData<RowColInt8QuantTilingData>();
  tiling->rows = shape.GetDim(0);
  tiling->cols = shape.GetDim(1);
  // Keep both INT8 rows and the FP32 statistics on distinct DMA blocks.
  uint32_t rows_per_group = 8;
  while (tiling->cols * rows_per_group % 32 != 0) {
    rows_per_group *= 2;
  }
  tiling->rows_per_group = rows_per_group;
  const uint32_t row_groups =
      (tiling->rows + rows_per_group - 1) / rows_per_group;
  const uint32_t col_groups = (tiling->cols + 63) / 64;
  const uint32_t groups = row_groups > col_groups ? row_groups : col_groups;
  context->SetBlockDim(groups < 48 ? groups : 48);
  context->GetWorkspaceSizes(1)[0] = 0;
  return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context) {
  const auto& shape = *context->GetInputShape(0);
  if (!Valid(shape)) {
    return GRAPH_FAILED;
  }
  *context->GetOutputShape(0) = shape;
  *context->GetOutputShape(1) = shape;
  *context->GetOutputShape(2) = gert::Shape({shape.GetDim(0)});
  *context->GetOutputShape(3) = gert::Shape({shape.GetDim(1)});
  return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context) {
  context->SetOutputDataType(0, DT_INT8);
  context->SetOutputDataType(1, DT_INT8);
  context->SetOutputDataType(2, DT_FLOAT);
  context->SetOutputDataType(3, DT_FLOAT);
  return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class XllmKernelRowColInt8Quant : public OpDef {
 public:
  explicit XllmKernelRowColInt8Quant(const char* name) : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("qRow")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("qCol")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("rowAbsmax")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("colAbsmax")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});

    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

    this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910_93");
  }
};

OP_ADD(XllmKernelRowColInt8Quant);
}  // namespace ops
