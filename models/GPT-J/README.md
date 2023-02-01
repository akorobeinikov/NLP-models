# GPT-J

## Use Case and High-Level Description

GPT-J 6B is a transformer model trained using Ben Wang's Mesh Transformer JAX. "GPT-J" refers to the class of model, while "6B" represents the number of trainable parameters.

More details provided in the [paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), [repository](https://github.com/huggingface/transformers) and [model card](https://huggingface.co/EleutherAI/gpt-j-6B).

## Specification

| Metric            | Value            |
|-------------------|------------------|
| Type              | Text Generation  |
| GFlops            | -                |
| number of layers  | 28               |
| Params            | 6 053 381 344    |
| Embedding size    | 4096             |
| Vocab size        | 50400            |
| Source framework  | PyTorch\*        |

## Input

### Original model

Token ids, name: `input`, dynamic shape in the format `B, L`, where:

- `B` - batch size
- `L` - sequence length

### Converted model

Token ids, name: `input`, dynamic shape in the format `B, L`, where:

- `B` - batch size
- `L` - sequence length

## Output

### Original model

Prediction scores of language modeling head, name: `output`, dynamic shape `B, L, 50257` in the format `B, L, S`, where:

- `B` - batch size
- `L` - sequence length
- `S` - vocab size

### Converted model

Prediction scores of language modeling head, name: `output`, dynamic shape `B, L, 50257` in the format `B, L, S`, where:

- `B` - batch size
- `L` - sequence length
- `S` - vocab size


## Demo usage

To run demo with GPT-J, use the next command line:

```
  python3 demo.py -m GPT-J -l <launcher_name>
```

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/huggingface/transformers/master/LICENSE).
