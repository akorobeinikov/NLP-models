# Copyright (C) 2023 KNS Group LLC (YADRO)
# SPDX-License-Identifier: Apache-2.0


import logging as log
import onnxruntime as ort
from openvino.runtime import Core, get_version, PartialShape, Dimension
import numpy as np
from typing import Any
import torch
import transformers

from provider import ClassProvider
from utils import get_model_path
from types import SimpleNamespace

MODEL_TO_URL = {
    "GPT-2": "gpt2",
    "GPT-J": "EleutherAI/gpt-j-6B",
    "GPT-NeoX": "EleutherAI/gpt-neox-20b",
    "BLOOM": "bigscience/bloom",
    "OPT": "facebook/opt-66b"
}


class BaseLauncher(ClassProvider):
    __provider_type__ = "launcher"
    def __init__(self, model_name: str, config=None) -> None:
        """
        Load model using a model_name

        :param model_name
        """
        pass

    def process(self, input_ids: np.array) -> Any:
        """
        Run launcher with user's input

        :param input_ids
        """
        pass


class PyTorchLauncher(BaseLauncher):
    __provider__ = "pytorch"
    def __init__(self, model_name: str, config=None) -> None:
        log.info('PyTorch Runtime')
        self.model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_TO_URL[model_name], torch_dtype=torch.float32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def process(self, input_ids: np.array) -> Any:
        generated_ids = self.model(torch.from_numpy(input_ids))
        return generated_ids.logits.detach().numpy()


class ONNXLauncher(BaseLauncher):
    __provider__ = "onnx"
    def __init__(self, model_name: str, config=None) -> None:
        log.info('ONNX Runtime')
        self.session = ort.InferenceSession(get_model_path(model_name, "onnx"))

    def process(self, input_ids: np.array) -> Any:
        outputs = self.session.run(["output"], {"input": input_ids})
        return outputs[0]


class OpenVINOLaucnher(BaseLauncher):
    __provider__ = "openvino"
    def __init__(self, model_name: str, config: SimpleNamespace) -> None:
        log.info('OpenVINO Runtime')
        super().__init__(model_name)
        core = Core()
        self.model = core.read_model(get_model_path(model_name, "openvino"))
        self.input_tensor = self.model.inputs[0].any_name
        if not config.dynamic_shape and (self.model.inputs[0].partial_shape.is_dynamic or self.model.inputs[0].shape[1] != config.max_seq_len):
            self.model.reshape({self.input_tensor: PartialShape([Dimension(1), Dimension(config.max_seq_len)])})

        if config.dynamic_shape:
            self.model.reshape({self.input_tensor: PartialShape([Dimension(1), Dimension(0, config.max_seq_len)])})

        # load model to the device
        self.compiled_model = core.compile_model(self.model, "CPU")#{'CPU_THREADS_NUM': '48', 'CPU_BIND_THREAD': 'NO', 'CPU_THROUGHPUT_STREAMS': 'CPU_THROUGHPUT_AUTO'})
        self.output_tensor = self.compiled_model.outputs[0]
        self.infer_request = self.compiled_model.create_infer_request()

    def process(self, input_ids: np.array) -> Any:
        inputs = {
            self.input_tensor: input_ids
        }
        # infer by OpenVINO runtime
        outputs = self.infer_request.infer(inputs)[self.output_tensor]
        return outputs


def create_launcher(laucnher_name: str, model_name: str, config: SimpleNamespace):
    return BaseLauncher.provide(laucnher_name, model_name, config)
