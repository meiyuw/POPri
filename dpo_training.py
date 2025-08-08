import hydra
from torch.nn import Module, Dropout
from datasets import Dataset
from omegaconf import OmegaConf, DictConfig
import sys
import os
import json
import glob
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    AutoTokenizer,
)
import contextlib
import torch

from trl import DPOTrainer, DPOConfig
from peft import LoraConfig
from utils import find_latest_checkpoint_path


def set_dropout_(model: Module, prob: float):
    for module in model.modules():
        if isinstance(module, Dropout):
            module.p = prob


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print(
        f"#####################\n# DPO training round {cfg.round_number}\n#####################",
        file=sys.stderr,
    )
    round_directory = os.path.join(
        cfg.logging.root_dir, "trial{0}/round{1}".format(cfg.logging.trial, cfg.round_number)
    )
    checkpoints_directory = os.path.join(
        cfg.logging.root_dir, "trial{0}".format(cfg.logging.trial)
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_path,
        cache_dir=cfg.model.save_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_path,
        cache_dir=cfg.model.save_dir,
    )
    set_dropout_(model, 0.1)
    tokenizer.pad_token_id = 0
    with open(
        os.path.join(round_directory, "preference_dataset.json"), "r", encoding="utf8"
    ) as f:
        pref_dataset = json.load(f)
    train_dataset_hf = Dataset.from_list(pref_dataset).shuffle()

    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=[
            "q_proj",
            "o_proj",
            "v_proj",
            "k_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    if cfg.round_number == 0:
            init_epoch = 0
    else:
        latest_ckpt_dir = find_latest_checkpoint_path(checkpoints_directory)
        state_path = os.path.join(latest_ckpt_dir, "trainer_state.json")
        with open(state_path, 'r', encoding='utf8') as f:
            trainer_state = json.load(f)
            init_epoch = trainer_state["epoch"]
    dpo_config = DPOConfig(
        beta=cfg.training.beta,
        max_length=512,
        max_prompt_length=256,
        output_dir=os.path.join(checkpoints_directory, "training_logs"),
        learning_rate=cfg.training.start_lr,
        gradient_accumulation_steps=cfg.training.gradient_accumulation,
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.eval_batch_size,
        save_strategy="epoch",
        logging_steps=1,
        lr_scheduler_type="cosine_with_min_lr",
        warmup_steps=10,
        logging_strategy="steps",
        lr_scheduler_kwargs={"min_lr": cfg.training.end_lr},
        remove_unused_columns=False,
        num_train_epochs=float(init_epoch + cfg.training.in_round_epochs),
        log_on_each_node=True,
        run_name=cfg.run_name,
        report_to="none",
        disable_tqdm=True,
        label_smoothing=cfg.training.label_smoothing,
    )
    trainer = DPOTrainer(
        model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_dataset_hf,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.model.print_trainable_parameters()
    if len(glob.glob(os.path.join(checkpoints_directory, 'training_logs')+'/checkpoint-*')) > 0:
        print('restart from previous checkpoint...')
        trainer.train(resume_from_checkpoint = True)
    else:
        trainer.train()
    trainer.save_model()



if __name__ == "__main__":
    sys.exit(main())
