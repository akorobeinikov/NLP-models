# Copyright (C) 2023 KNS Group LLC (YADRO)
# SPDX-License-Identifier: Apache-2.0

import logging as log
import transformers
from argparse import ArgumentParser
import sys
import torch
from torchsummary import summary
from time import perf_counter
import onnx
from launchers import create_launcher, MODEL_TO_URL

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)

MODELS = [
    "GPT-J",
    "GPT-NeoX",
    "BLOOM",
    "OPT"
]

LAUNCHERS = [
    "PyTorch",
    "onnx",
    "openvino"
]


def build_argparser():
    parser = ArgumentParser()

    options = parser.add_argument_group('Options')
    options.add_argument('-i', '--input', default="", type=str,
                         help='Optional. An text input to process. If is not specified, use interactive mode')
    options.add_argument('-m', '--model', required=False, choices=MODELS, default="GPT-J",
                         help=f"Optional. Name of using model. Available names = {MODELS}. Default is 'GPT-J'")
    options.add_argument('-l', '--launcher', required=False, choices=LAUNCHERS, default="PyTorch",
                         help="Optional. Name of using backend for runtime. Available backends = {LAUNCHERS}. Default is 'PyTorch'")
    options.add_argument('-o', '--output',
                         help='Optional. Name of the output file(s) to save.')
    return parser


@torch.no_grad()
def convert_to_onnx(model, input_shapes, output_file):
    """Convert PyTorch model to ONNX and check the resulting onnx model"""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy_inputs = tuple(
        torch.zeros(input_shape, dtype=torch.float32)
        for input_shape in input_shapes)
    model(*dummy_inputs)
    torch.onnx.export(model, dummy_inputs, str(output_file), verbose=False, input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size", 1: "sequence_len"},"output": {0: "batch_size", 1: "sequence_len"}})

    model = onnx.load(str(output_file))


def create_tokenier_launcher(model_name: str, laucnher_name: str):
    start_time = perf_counter()
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_TO_URL[model_name])
    launcher = create_launcher(laucnher_name, model_name)
    load_time = perf_counter() - start_time

    log.info("Load time = {:.3f} seconds".format(load_time))

    return tokenizer, launcher

def generate_text(input, tokenizer, launcher, device="cpu"):
    start_time = perf_counter()
    # Get input_ids using tokenizer
    input_ids = tokenizer(input, return_tensors="np").input_ids

    generated_ids = launcher.process(input_ids)
    print(generated_ids.shape)
    # generated_text = tokenizer.decode(generated_ids[0])
    infer_time = perf_counter() - start_time

    return (generated_ids.shape, infer_time)


def interactive_mode(tokenizer, model, device="cpu"):
    user_input = input("Please enter new input: ")
    all_time = 0
    count_inputs = 0
    while(user_input!="exit"):
        model_answer, time = generate_text(user_input, tokenizer, model, device)
        all_time += time
        count_inputs += 1
        print(f"Answer:  {model_answer}")
        user_input = input("Please enter new input: ")

    infer_time = all_time / count_inputs
    log.info("Average infer time = {:.3f} seconds".format(infer_time))
    return


# def gpt2_code():
#      # loop on user's or prepared prompts
#     for prompt in prompts():
#         if not prompt.strip():
#             break

#         # encode input
#         tokens = tokenizer.encode_batch([prompt])[0].ids
#         input_ids = np.array([tokens], dtype=np.int32)

#         # maximum number of tokens that can be processed by network at once
#         max_length = args.max_seq_len

#         eos_token_id = len(vocab) - 1

#         cur_input_len = input_ids.shape[-1]

#         # maximum number of tokens that will be generated
#         max_sample_token_num = args.max_sample_token_num + cur_input_len

#         t0 = time.perf_counter()
#         t_count = 0

#         while True:
#             model_input = input_ids
#             if not args.dynamic_shape:
#                 # pad the rest of the request
#                 pad_len = max_length - cur_input_len
#                 model_input = np.concatenate((input_ids, [[eos_token_id] * pad_len]), axis=-1)

#             # create numpy inputs for OpenVINO runtime
#             inputs = {
#                 input_tensor: model_input,
#             }

#             # infer by OpenVINO runtime
#             t_start = time.perf_counter()
#             outputs = infer_request.infer(inputs)[output_tensor]
#             t_end = time.perf_counter()
#             t_count += 1
#             log.info("Sequence of length {} is processed with {:0.2f} requests/sec ({:0.2} sec per request)".format(
#                 model_input.shape[1], 1 / (t_end - t_start), t_end - t_start))

#             next_token_logits = outputs[:, cur_input_len-1, :]

#             # pre-process distribution
#             next_token_scores = process_logits(input_ids, next_token_logits, eos_token_id)
#             if args.top_k > 0:
#                 next_token_scores = get_top_k_logits(next_token_scores, args.top_k)

#             if args.top_p < 1.0:
#                 next_token_scores = get_top_p_logits(next_token_scores, args.top_p)

#             # get next token id
#             probs = softmax(next_token_scores)
#             next_tokens = np.random.choice(probs.shape[-1], 1, p=probs[0], replace=True)

#             # update info for the next step
#             input_ids = np.concatenate((input_ids, [next_tokens]), axis=-1)

#             cur_input_len = input_ids.shape[-1]

#             if stop_criteria(input_ids, min(max_length, max_sample_token_num), eos_token_id):
#                 break

#         t1 = time.perf_counter()

#         text = tokenizer.decode_batch(input_ids)[0]

#         log.info("{} requests were processed in {:0.2f}sec ({:0.2}sec per request)".format(
#             t_count, t1 - t0, (t1 - t0) / t_count))

#         # print result
#         log.info("GENERATED SEQUENCE: {}".format(text))

def main():
    args = build_argparser().parse_args()

    tokenizer, launcher = create_tokenier_launcher(args.model, args.launcher)

    if args.input:
        model_answer, time = generate_text(args.input, tokenizer, launcher)
        print(f"Answer:  {model_answer}")
        log.info("Average infer time = {:.3f} seconds".format(time))
    else:
        interactive_mode(tokenizer, launcher)


if __name__ == '__main__':
    sys.exit(main() or 0)
