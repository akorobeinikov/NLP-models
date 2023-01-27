# Copyright (C) 2023 KNS Group LLC (YADRO)
# SPDX-License-Identifier: Apache-2.0


import json
import numpy as np


def load_vocab_file(vocab_file_name):
    with open(vocab_file_name, "r", encoding="utf-8") as content:
        return json.load(content)


def get_top_k_logits(scores, top_k):
    filter_value = -float("Inf")
    top_k = min(max(top_k, 1), scores.shape[-1])
    top_k_scores = -np.sort(-scores)[:, :top_k]
    indices_to_remove = scores < np.min(top_k_scores)
    filtred_scores = np.ma.array(scores, mask=indices_to_remove, fill_value=filter_value).filled()
    return filtred_scores


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    sum = e_x.sum(axis=-1, keepdims=True)
    return e_x / sum


def get_top_p_logits(scores, top_p):
    filter_value = -float("Inf")
    sorted_indices = np.argsort(-scores)
    sorted_logits = -np.sort(-scores)
    cumulative_probs = np.cumsum(softmax(sorted_logits), axis=-1)
    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1]
    sorted_indices_to_remove[:, 0] = 0
    np.put_along_axis(sorted_indices_to_remove, sorted_indices, sorted_indices_to_remove, axis=1)
    filtred_scores = np.ma.array(scores, mask=sorted_indices_to_remove, fill_value=filter_value).filled()
    return filtred_scores


def process_logits(input_ids, scores, eos_token_id, min_length=0):
    cur_len = input_ids.shape[-1]
    if cur_len < min_length:
        scores[:, eos_token_id] = -float("inf")
    return scores


def stop_criteria(input_ids, max_length, eos_token_id):
    if input_ids[0][-1] == eos_token_id:
        return True

    return input_ids.shape[-1] >= max_length


MODELS_RELATIVES_PATHS = {
    "GPT-J_onnx": "models/GPT-J/model_files/onnx/gpt-j.onnx",
    "GPT-J_openvino": "models/GPT-J/model_files/IR/gpt-j.xml"
}


def get_model_path(model_name: str, launcher_type: str):
    full_path = model_name + "_" + launcher_type
    return MODELS_RELATIVES_PATHS[full_path]
