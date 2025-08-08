from omegaconf import DictConfig
import hydra
from transformers import (
    AutoModelForCausalLM,
    LlamaConfig,
    AutoTokenizer,
)
from peft import PeftModel
from utils import find_latest_checkpoint_path
import sys
import os
import shutil

@hydra.main(config_path=None, config_name=None, version_base=None)
def main(cfg: DictConfig):
    checkpoints_directory = os.path.join(
        cfg.logging.root_dir, "trial{0}".format(cfg.logging.trial)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_path, cache_dir=cfg.model.save_dir
    )
    curr_config = LlamaConfig.from_pretrained(
        cfg.model.model_path, cache_dir=cfg.model.save_dir
    )
    tokenizer.pad_token_id = 0
    prev_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_path, cache_dir=cfg.model.save_dir
    )
    curr_checkpoint_directory = find_latest_checkpoint_path(checkpoints_directory)
    merge_model_peft = PeftModel.from_pretrained(
        prev_model,
        curr_checkpoint_directory,
    )
    merged_model = merge_model_peft.merge_and_unload()
    merged_model.save_pretrained(
        os.path.join(curr_checkpoint_directory, "merged_model")
    )
    curr_config.save_pretrained(
        os.path.join(curr_checkpoint_directory, "merged_model")
    )
    tokenizer.save_pretrained(
        os.path.join(curr_checkpoint_directory, "merged_model")
    )

    # if cfg.round_number > 0:
    #     previous_checkpoint_num = checkpoint_interval * cfg.training.in_round_epochs * cfg.round_number
    #     last_merged_model_directory = os.path.join(
    #         checkpoints_directory,
    #         f"training_logs/checkpoint-{previous_checkpoint_num}/merged_model",
    #     )
    #     shutil.rmtree(last_merged_model_directory)




if __name__ == "__main__":
   sys.exit(main())
