# Copyright (C) 2023 KNS Group LLC (YADRO)
# SPDX-License-Identifier: Apache-2.0

MODELS_RELATIVES_PATHS = {
    "GPT-J_onnx": "onnx_model/gpt-j.onnx"
}


def get_model_path(model_name: str, launcher_type: str):
    full_path = model_name + "_" + launcher_type
    return MODELS_RELATIVES_PATHS[full_path]
