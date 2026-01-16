#include "tensorflow/lite/delegates/gpu/common/mediapipe/landmarks_to_transform_matrix.h"
#include "tensorflow/lite/delegates/gpu/common/mediapipe/transform_landmarks.h"
#include "tensorflow/lite/delegates/gpu/common/mediapipe/transform_tensor_bilinear.h"

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "flatbuffers/flexbuffers.h"

namespace tflite {
namespace gpu {

absl::Status ParseLandmarksToTransformMatrixV1Attributes(
    const void* data, uint32_t data_size,
    LandmarksToTransformMatrixV1Attributes* attr, BHWC* output_shape) {
  const flexbuffers::Map m =
      flexbuffers::GetRoot(reinterpret_cast<const uint8_t*>(data), data_size)
          .AsMap();

  const auto input_hw = m["input_hw"].AsTypedVector();
  attr->input_hw = HW(input_hw[0].AsInt32(), input_hw[1].AsInt32());

  const auto output_hw = m["output_hw"].AsTypedVector();
  attr->output_hw = HW(output_hw[0].AsInt32(), output_hw[1].AsInt32());

  attr->dimensions = m["dimensions"].AsInt32();
  attr->landmarks_range = m["landmarks_range"].AsInt32();
  attr->bbox_size_multiplier = m["bbox_size_multiplier"].AsFloat();
  attr->left_rotation_idx = m["left_rotation_idx"].AsInt32();
  attr->right_rotation_idx = m["right_rotation_idx"].AsInt32();

  const auto subset = m["subset"].AsTypedVector();
  for (int i = 0; i < subset.size() / 2; i++) {
    attr->subset.emplace_back(subset[i * 2].AsInt32(),
                              subset[i * 2 + 1].AsInt32());
  }
  if (subset.size() % 2 != 0) {
    attr->subset.emplace_back(subset[subset.size() - 1].AsInt32(),
                              subset[subset.size() - 1].AsInt32());
  }

  *output_shape = BHWC(1, 1, 4, 4);
  return absl::OkStatus();
}

absl::Status ParseLandmarksToTransformMatrixV2Attributes(
    const void* data, uint32_t data_size,
    LandmarksToTransformMatrixV2Attributes* attr, BHWC* output_shape) {
  const flexbuffers::Map m =
      flexbuffers::GetRoot(reinterpret_cast<const uint8_t*>(data), data_size)
          .AsMap();
  const auto subset_idxs = m["subset_idxs"].AsTypedVector();
  int amount = subset_idxs.size();
  for (int i = 0; i < amount / 2; i++) {
    attr->subset_idxs.emplace_back(subset_idxs[i * 2].AsInt32(),
                                   subset_idxs[i * 2 + 1].AsInt32());
  }
  if (amount % 2 != 0) {
    int previous = amount - 1;
    attr->subset_idxs.emplace_back(subset_idxs[previous].AsInt32(),
                                   subset_idxs[previous].AsInt32());
  }
  attr->left_rotation_idx = m["left_rotation_idx"].AsInt32();
  attr->right_rotation_idx = m["right_rotation_idx"].AsInt32();
  attr->target_rotation_radians = m["target_rotation_radians"].AsFloat();
  attr->output_height = m["output_height"].AsInt32();
  attr->output_width = m["output_width"].AsInt32();
  attr->scale_x = m["scale_x"].AsFloat();
  attr->scale_y = m["scale_y"].AsFloat();

  *output_shape = BHWC(1, 1, 4, 4);
  return absl::OkStatus();
}

absl::Status ParseTransformLandmarksV1Attributes(
    const void* data, uint32_t data_size, TransformLandmarksAttributes* attr,
    BHWC* output_shape) {
  attr->version = 1;

  const flexbuffers::Map m =
      flexbuffers::GetRoot(reinterpret_cast<const uint8_t*>(data), data_size)
          .AsMap();
  const flexbuffers::TypedVector keys = m.Keys();

  for (int k = 0; k < keys.size(); ++k) {
    const std::string key = keys[k].ToString();
    const auto value = m[key];
    if (key == "dimensions") {
      attr->dimensions = value.AsInt32();
    }
    if (key == "scale") {
      attr->scale = value.AsFloat();
    }
  }
  return absl::OkStatus();
}

absl::Status ParseTransformLandmarksV2Attributes(
    const void* data, uint32_t data_size, TransformLandmarksAttributes* attr,
    BHWC* output_shape) {
  attr->version = 2;
  attr->dimensions = output_shape->c;
  attr->scale = 1.0;

  return absl::OkStatus();
}

absl::Status ParseTransformTensorBilinearV1Attributes(
    const void* data, uint32_t data_size,
    TransformTensorBilinearAttributes* attr, BHWC* output_shape) {
  attr->version = 1;

  const flexbuffers::Map m =
      flexbuffers::GetRoot(reinterpret_cast<const uint8_t*>(data), data_size)
          .AsMap();
  const flexbuffers::TypedVector keys = m.Keys();

  for (int k = 0; k < keys.size(); ++k) {
    const std::string key = keys[k].ToString();
    const auto value = m[key];
    if (key == "mode") {
      if (value.AsString().str() != "bilinear") {
        return absl::UnimplementedError(
            "TransformTensor operation supports only bilinear interpolation.");
      }
    }

    if (key == "output_size") {
      attr->output_size = HW(value.AsTypedVector()[0].AsInt32(),
                             value.AsTypedVector()[1].AsInt32());
    }
  }
  attr->align_corners = false;
  *output_shape = BHWC(1, attr->output_size.h, attr->output_size.w, 1);
  return absl::OkStatus();
}

absl::Status ParseTransformTensorBilinearV2Attributes(
    const void* data, uint32_t data_size,
    TransformTensorBilinearAttributes* attr, BHWC* output_shape) {
  attr->version = 2;

  const flexbuffers::Map m =
      flexbuffers::GetRoot(reinterpret_cast<const uint8_t*>(data), data_size)
          .AsMap();
  const flexbuffers::TypedVector keys = m.Keys();
  HW output_size;
  for (int k = 0; k < keys.size(); ++k) {
    const std::string key = keys[k].ToString();
    const auto value = m[key];
    if (key == "output_height") {
      output_size.h = value.AsInt32();
    }
    if (key == "output_width") {
      output_size.w = value.AsInt32();
    }
  }
  attr->output_size = std::move(output_size);
  attr->align_corners = true;
  *output_shape = BHWC(1, attr->output_size.h, attr->output_size.w, 1);
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
