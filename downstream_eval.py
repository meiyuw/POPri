import torch
import os
import json
import hydra
import torch.nn as nn
import sys
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from accelerate import Accelerator
from datasets import Dataset
import json
from omegaconf import OmegaConf, DictConfig


def evaluate(model, eval_loader, accelerator, xent_loss):
    model.eval()
    total_loss = 0.0
    top_k_accuracies = {1: 0, 3: 0, 5: 0}
    total_evaluated_tokens = 0

    with torch.no_grad():
        for batch in eval_loader:
            # Move batch to the appropriate device            
            outputs = model(**batch)        
            logits = outputs.logits
            labels = batch['labels']
            
            # Shift logits and labels to align them properly
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the logits and labels to calculate the loss
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)

            # Create a mask to ignore the padding tokens (-100) in loss calculation
            valid_mask = flat_labels != -100
            
            # Apply the mask to filter out invalid entries
            filtered_logits = flat_logits[valid_mask]
            filtered_labels = flat_labels[valid_mask]

            # Calculate the loss for valid entries
            loss = xent_loss(filtered_logits, filtered_labels)
            total_loss += loss

            # Calculate top-k accuracies
            top_k_values, top_k_indices = torch.topk(filtered_logits, k=max(top_k_accuracies.keys()), dim=-1)
            expanded_labels = filtered_labels.unsqueeze(1)
            
            correct_predictions = top_k_indices == expanded_labels
            for k in top_k_accuracies:
                top_k_accuracies[k] += correct_predictions[:, :k].sum()

            
            # Update the total count of evaluated tokens
            total_evaluated_tokens += valid_mask.sum()
    
    total_evaluated_tokens = torch.sum(accelerator.gather(total_evaluated_tokens).detach().cpu()).item()
    total_loss = torch.sum(accelerator.gather(total_loss).detach().cpu()).item()
    # Normalize the top-k accuracies by the total number of evaluated tokens
    for k in top_k_accuracies:
        correct_tokens = torch.sum(accelerator.gather(top_k_accuracies[k]).detach().cpu()).item()
        top_k_accuracies[k] = correct_tokens / total_evaluated_tokens if total_evaluated_tokens > 0 else 0

    # Calculate the average loss
    avg_loss = total_loss / total_evaluated_tokens if len(eval_loader) > 0 else 0.0

    return avg_loss, top_k_accuracies


def save_checkpoint(model, optimizer, accelerator, epoch, filename="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': accelerator.unwrap_model(model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accelerator_rng_state': accelerator.state
    }
    # Save the checkpoint
    accelerator.save(checkpoint, filename)
def add_module_prefix(state_dict):
    """Add 'module.' prefix to state dict keys if not present."""
    return {('module.' + k if not k.startswith('module.') else k): v for k, v in state_dict.items()}

def load_checkpoint(model, optimizer, accelerator, filename="checkpoint.pth"):
    checkpoint = torch.load(filename, map_location=accelerator.device)
    adjusted_model_state_dict = add_module_prefix(checkpoint['model_state_dict'])
    model.load_state_dict(adjusted_model_state_dict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch

def find_latest_checkpoint(checkpoint_dir):
    # List all files in the directory
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint") and f.endswith(".pth")]
    # Sort files by epoch number in descending order
    checkpoint_files.sort(key=lambda x: int(x.replace('checkpoint', '').replace('.pth', '')), reverse=True)
    # Return the latest checkpoint file
    return checkpoint_files[0] if checkpoint_files else None

@hydra.main(config_path=None, config_name=None, version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    tokenizer = GPT2Tokenizer.from_pretrained(cfg.downstream_model.model, cache_dir=cfg.downstream_model.save_dir)
    model = AutoModelForCausalLM.from_pretrained(cfg.downstream_model.model, cache_dir=cfg.downstream_model.save_dir)
    vocab_size = tokenizer.vocab_size
    cached_embedding = model.transformer.wte.weight[:vocab_size]
    dim = model.transformer.wte.weight.shape[1]
    pad_idx = vocab_size
    extended_embedding = nn.Embedding(vocab_size + 1, dim, padding_idx=pad_idx)
    extended_weight = torch.cat([cached_embedding, torch.zeros(1, dim)])
    del cached_embedding
    extended_embedding.load_state_dict({"weight": extended_weight})
    model.transformer.wte = extended_embedding
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    cutoff_len = cfg.dataset.trunc_len
    grad_accum_steps = cfg.downstream_eval_settings.grad_accum_steps
    total_epochs = cfg.downstream_eval_settings.total_epochs
    batch_size = cfg.downstream_eval_settings.batch_size
    cur_lr = cfg.downstream_eval_settings.learning_rate
    downstream_eval_folder = os.path.join(cfg.logging.root_dir, 'downstream_eval')

    log_dir = os.path.join(downstream_eval_folder, f'downstream_accuracy')
    accelerator = Accelerator(mixed_precision="bf16")
    pretrained_ckpt = cfg.downstream_eval_settings.pretrained_eval_model_checkpoint
    if accelerator.is_main_process:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    accelerator.wait_for_everyone()

    def tokenize(example):
        # Tokenizing the sentence and adding BOS and EOS tokens.
        sent = example['text']
        sent = tokenizer.tokenize(sent)
        sent = [tokenizer.bos_token] + sent + [tokenizer.eos_token]
        
        # Encoding the tokens to get 'input_ids' and 'attention_mask'
        encoded_dict = tokenizer.encode_plus(
            sent,
            max_length=cutoff_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            is_split_into_words=True
        )
        
        # Flatten the 'input_ids' and convert to long for consistency
        input_ids = encoded_dict['input_ids'].flatten().long()
        
        # Constructing 'labels' based on 'input_ids': ignoring padding tokens by setting them to -100
        labels = [-100 if token == tokenizer.pad_token_id else token for token in input_ids.tolist()]
        
        # Building the final result dictionary
        result = {
            'input_ids': input_ids.tolist(),
            'attention_mask': encoded_dict['attention_mask'].flatten().long().tolist(),
            'labels': labels
        }
        return result
    with open(os.path.join(downstream_eval_folder, 'generated_samples.json'), encoding='utf8') as f:
        train_texts = json.load(f)
    print("Number of training samples", len(train_texts))
    with open(cfg.dataset.eval_data, 'r', encoding="utf8") as f:
        eval_texts = json.load(f)
    train_dict = [{'text': x} for x in train_texts]
    train_dataset_hf = Dataset.from_list(train_dict)
    train_data_tokenized = train_dataset_hf.shuffle().map(tokenize, num_proc=10)
    train_data_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    test_data_dict = [{'text': x} for x in eval_texts]
    test_dataset_hf = Dataset.from_list(test_data_dict)
    test_data_tokenized = test_dataset_hf.map(tokenize, num_proc=10)
    test_data_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    train_dataloader = DataLoader(train_data_tokenized, batch_size=batch_size, num_workers=32)
    test_dataloader = DataLoader(test_data_tokenized, batch_size=8*2, drop_last=False, shuffle=False)

    # accelerator
    checkpoint = torch.load(pretrained_ckpt, map_location=accelerator.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model, optimizer, train_loader, eval_loader = accelerator.prepare(
        model, AdamW(model.parameters(), lr=cur_lr), train_dataloader, test_dataloader
    )

    cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')

    avg_loss, top_k_accuracies = evaluate(model, eval_loader, accelerator, cross_entropy_loss)
    if accelerator.is_main_process:
        print(f'No finetuning, evaluation Loss: {avg_loss:.4f}', file=sys.stderr)
        for k, accuracy in top_k_accuracies.items():
            print(f'No finetuning, Top-{k} Accuracy: {accuracy:.4f}', file=sys.stderr)

    start_epoch = 0
    checkpoint_num = find_latest_checkpoint(log_dir)
    if checkpoint_num is not None:
         latest_checkpoint_file = os.path.join(log_dir, checkpoint_num)
        
         start_epoch = load_checkpoint(model, optimizer, accelerator, latest_checkpoint_file) + 1
         if accelerator.is_main_process:
            print('Found checkpoint, loading', latest_checkpoint_file, file=sys.stderr)
            print('Starting from epoch', start_epoch, file=sys.stderr)

    best_accuracy = 0.
    best_dict = None
    # Run training and evaluation
    for epoch in range(start_epoch, total_epochs):
        model.train()
        total_loss = 0
        num_accumulated_steps = 0  # Track the number of accumulated steps
        curr_step_loss = 0
        num_actual_steps = 1
        for step, batch in enumerate(train_loader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / grad_accum_steps  # Normalize loss to account for accumulation
            accelerator.backward(loss)
            total_loss += loss.item()
            curr_step_loss += loss.item()
            num_accumulated_steps += 1

            # Perform the optimization step at the specified accumulation interval or at the last batch
            if (step + 1) % grad_accum_steps == 0 or step == len(train_loader) - 1:
                optimizer.step()
                optimizer.zero_grad()
                num_accumulated_steps = 0  # Reset the accumulation counter after an optimizer step
                if accelerator.is_main_process:
                    print(f"Epoch {epoch}, Step {num_actual_steps} loss: {curr_step_loss}", file=sys.stderr)
                curr_step_loss = 0
                num_actual_steps += 1

        # Calculate average loss over actual number of updates to adjust for any smaller final accumulation
        actual_updates = len(train_loader) // grad_accum_steps + (1 if len(train_loader) % grad_accum_steps != 0 else 0)
        avg_loss = total_loss / actual_updates
        if accelerator.is_main_process:
            print(f"Epoch {epoch} Avg training loss: {avg_loss}", file=sys.stderr)

        avg_loss, top_k_accuracies = evaluate(model, eval_loader, accelerator, cross_entropy_loss)
        if accelerator.is_main_process:
            print(f'Epoch {epoch} evaluation Loss: {avg_loss:.4f}', file=sys.stderr)
            for k, accuracy in top_k_accuracies.items():
                print(f'Epoch {epoch} Top-{k} Accuracy: {accuracy:.4f}', file=sys.stderr)
            top_k_accuracies['cross_entropy_loss'] = avg_loss
            stats_path = os.path.join(log_dir, f'epoch{epoch}_stats.json')
            print('Saving stats in ', stats_path, file=sys.stderr)
            with open(stats_path, 'w+') as f:
                json.dump(top_k_accuracies, f)
            
            if best_accuracy < top_k_accuracies[1]:
                best_accuracy = top_k_accuracies[1]
                best_dict = top_k_accuracies
                best_dict['epoch'] = epoch
                with open(os.path.join(log_dir, f'best_accruacy.json'), 'w') as f:
                    json.dump(best_dict, f)
            if epoch % 10 ==0:
                checkpoint_path = os.path.join(log_dir, f'checkpoint{epoch}.pth')
                save_checkpoint(model, optimizer, accelerator, epoch, filename=checkpoint_path)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            stats_path = os.path.join(log_dir, f'best_stats.json')
            print('Saving stats in ', stats_path, file=sys.stderr)
            with open(stats_path, 'w+') as f:
                json.dump(best_dict, f)


if __name__ == "__main__":
    main()