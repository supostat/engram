import os

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def export_model_to_onnx(
    model,
    feature_count: int,
    models_path: str,
    filename: str,
) -> str:
    os.makedirs(models_path, exist_ok=True)

    initial_type = [
        ("float_input", FloatTensorType([None, feature_count])),
    ]

    onnx_model = convert_sklearn(model, initial_types=initial_type)
    onnx_path = os.path.join(models_path, filename)

    with open(onnx_path, "wb") as onnx_file:
        onnx_file.write(onnx_model.SerializeToString())

    return onnx_path
