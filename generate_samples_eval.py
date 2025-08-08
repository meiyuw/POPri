from vllm import LLM, SamplingParams
import hydra
from omegaconf import DictConfig
import os
import json
from vllm.lora.request import LoRARequest
import re
from transformers import AutoTokenizer
import torch
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)
import ray

import gc
import contextlib
import sys
from utils import find_latest_checkpoint_path

def extract_and_truncate_synthetic_samples(s, tokenizer, token_limit):
    extracted_samples = []
    for text in s:
        tokens = tokenizer.tokenize(text)[:token_limit]
        truncated_text = tokenizer.convert_tokens_to_string(tokens)
        extracted_samples.append(truncated_text)
    return extracted_samples


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(cfg: DictConfig):
    round_directory = os.path.join(
        cfg.logging.root_dir,
        "trial{0}/round{1}".format(cfg.logging.trial, cfg.round_number),
    )
    checkpoints_directory = os.path.join(
        cfg.logging.root_dir, "trial{0}".format(cfg.logging.trial)
    )
    with open(os.path.join(round_directory, f"prompt_set.json"), encoding="utf8") as f:
        prompt_data = json.load(f)

    seed = 2 ** (cfg.logging.trial + 1) * 3 ** (cfg.round_number + 1)

    latest_ckpt_dir = find_latest_checkpoint_path(checkpoints_directory)
    merged_model_directory = os.path.join(latest_ckpt_dir, 'merged_model')
    print('Merged model directory', merged_model_directory)
    llm = LLM(
        merged_model_directory,
        max_model_len=cfg.dataset.max_model_len,
        seed=seed,
        tensor_parallel_size=torch.cuda.device_count(),
    )

    sampling_params = SamplingParams(
        temperature=cfg.vllm_sampling.temperature,
        top_p=cfg.vllm_sampling.top_p,
        max_tokens=cfg.vllm_sampling.max_tokens,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.trunc_model.model_path, cache_dir=cfg.trunc_model.save_dir
    )
    prompt_list = []
    for idx in range(len(prompt_data)):
        curr_prompt = prompt_data[idx]
        prompt_list.append(curr_prompt)

    outputs = llm.generate(
        prompt_list,
        sampling_params,
    )
    output_list = []
    for output in outputs:
        generated_text = output.outputs[0].text
        output_list.append(generated_text)
    all_data = []
    for text in output_list:
        split_samples = re.split("Orig", text)
        raw_sample = split_samples[0]
        raw_sample = raw_sample.strip()
        raw_sample = raw_sample.strip("\n")
        all_data.append(raw_sample.replace("\n\n", " ").replace("\n", " "))

    trunc_samples = extract_and_truncate_synthetic_samples(
        all_data, tokenizer, cfg.dataset.trunc_len
    )
    with open(
        os.path.join(round_directory, f"fid_checkpoint.json"),
        "w+",
        encoding="utf8",
    ) as f:
        json.dump(trunc_samples, f, ensure_ascii=False)

    destroy_model_parallel()
    destroy_distributed_environment()
    del llm
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()
    print("Successfully delete the llm pipeline and free the GPU memory.")


if __name__ == "__main__":
    sys.exit(main())
