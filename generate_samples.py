import hydra
from omegaconf import OmegaConf, DictConfig
import sys
import os
from vllm import LLM, SamplingParams
import torch
import random
import json
import re
from transformers import AutoTokenizer
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)
import ray

import gc
import contextlib
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
    print(OmegaConf.to_yaml(cfg))
    round_directory = os.path.join(
        cfg.logging.root_dir, "trial{0}/round{1}".format(cfg.logging.trial, cfg.round_number)
    )
    checkpoints_directory = os.path.join(
        cfg.logging.root_dir, "trial{0}".format(cfg.logging.trial)
    )
    os.makedirs(round_directory, exist_ok=True)

    seed = 2 ** (cfg.logging.trial + 1) * 3 ** (cfg.round_number + 1)

    if cfg.round_number == 0:
        llm = LLM(
            model=cfg.model.model_path,
            download_dir=cfg.model.save_dir,
            max_model_len=512,
            seed=seed,
            tensor_parallel_size=torch.cuda.device_count(),
        )
    else:
        latest_ckpt_dir = find_latest_checkpoint_path(checkpoints_directory)
        merged_model_directory = os.path.join(
            latest_ckpt_dir, "merged_model",
        )
        print(merged_model_directory)
        llm = LLM(
            merged_model_directory,
            max_model_len=512,
            seed=seed,
            tensor_parallel_size=torch.cuda.device_count(),
        )

    with open(cfg.dataset.prompt_template, "r") as f:
        prompt_template = f.read()
    with open(cfg.dataset.initial_set, "r", encoding="utf8") as f:
        base_data = json.load(f)

    prompt_list = []
    prompt_set = []
    for _ in range(cfg.dataset.num_pref_pairs):
        examples = random.sample(base_data, 3)
        ex1 = examples[0]
        ex2 = examples[1]
        ex3 = examples[2]
        curr_prompt = prompt_template.format(
            ex1.replace("\n", " ").replace("\t", " "),
            ex2.replace("\n", " ").replace("\t", " "),
            ex3.replace("\n", " ").replace("\t", " "),
        )
        prompt_set.append(curr_prompt)
        for _ in range(cfg.dataset.rank_samples):
            prompt_list.append(curr_prompt)

    sampling_params = SamplingParams(
        temperature=cfg.vllm_sampling.temperature,
        top_p=cfg.vllm_sampling.top_p,
        max_tokens=cfg.vllm_sampling.max_tokens,
    )
    outputs = llm.generate(prompt_list, sampling_params)


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
    generated_samples = all_data

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_path, cache_dir=cfg.model.save_dir
    )

    trunc_samples = extract_and_truncate_synthetic_samples(
        generated_samples, tokenizer, cfg.dataset.trunc_len
    )
    print("round_directory:", round_directory)
    with open(
        os.path.join(round_directory, "generated_samples.json"), "w+", encoding="utf8"
    ) as f:
        json.dump(trunc_samples, f, ensure_ascii=False)

    print(trunc_samples[0])
    with open(
        os.path.join(round_directory, "prompt_set.json"), "w+", encoding="utf8"
    ) as f:
        json.dump(prompt_set, f, ensure_ascii=False)
    
    print(
        f"#####################\n# Data generation done {cfg.round_number}\n#####################",
        file=sys.stderr,
    )

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
    try:
        main()
    finally:
        sys.exit(0)
