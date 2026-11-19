// Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.

#include "dual_affine_layer_norm_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  const auto& x = context->GetInputShape(0)->GetStorageShape();
  if (x.GetDimNum() != 2 || x.GetDim(0) < 1 || x.GetDim(0) > 4096 ||
      x.GetDim(1) < 1 || x.GetDim(1) > 8192) {
    return ge::GRAPH_FAILED;
  }
  for (uint32_t i = 1; i < 5; ++i) {
    const auto& parameter = context->GetInputShape(i)->GetStorageShape();
    if (parameter.GetDimNum() != 1 || parameter.GetDim(0) != x.GetDim(1)) {
      return ge::GRAPH_FAILED;
    }
  }
  const float epsilon = *context->GetAttrs()->GetAttrPointer<float>(0);
  if (!(epsilon >= 1e-7f && epsilon <= 1e-2f)) {
    return ge::GRAPH_FAILED;
  }
  auto* tiling = context->GetTilingData<DualAffineLayerNormTilingData>();
  tiling->rows = x.GetDim(0);
  tiling->width = x.GetDim(1);
  tiling->epsilon = epsilon;
  tiling->reciprocal_width = 1.0f / static_cast<float>(tiling->width);
  // A worker boundary must also be a 32-byte output boundary.
  uint32_t rows_per_group = 1;
  while ((tiling->width * rows_per_group) % 16 != 0) {
    rows_per_group *= 2;
  }
  tiling->rows_per_group = rows_per_group;
  const uint32_t groups = (tiling->rows + rows_per_group - 1) / rows_per_group;
  context->SetBlockDim(groups < 48 ? groups : 48);
  context->GetWorkspaceSizes(1)[0] = 0;
  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context) {
  *context->GetOutputShape(0) = *context->GetInputShape(0);
  *context->GetOutputShape(1) = *context->GetInputShape(0);
  return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext* context) {
  const auto input_type = context->GetInputDataType(0);
  context->SetOutputDataType(0, input_type);
  context->SetOutputDataType(1, input_type);
  return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class XllmKernelDualAffineLayerNorm : public OpDef {
 public:
  explicit XllmKernelDualAffineLayerNorm(const char* name) : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("gamma1")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("beta1")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("gamma2")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("beta2")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("y1")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("y2")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-5f);

    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

    this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910_93");
  }
};

OP_ADD(XllmKernelDualAffineLayerNorm);
}  // namespace ops
