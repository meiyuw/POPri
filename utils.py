import os
import re
import json

def find_latest_checkpoint_path(checkpoints_directory):
    logs_dir = os.path.join(checkpoints_directory, "training_logs")
    if not os.path.isdir(logs_dir):
        raise FileNotFoundError(f"{logs_dir} not found")

    pattern = re.compile(r"^checkpoint-(\d+)$")
    max_ckpt_num = None

    for entry in os.listdir(logs_dir):
        m = pattern.match(entry)
        if not m:
            continue
        ckpt_num = int(m.group(1))
        if max_ckpt_num is None or ckpt_num > max_ckpt_num:
            max_ckpt_num = ckpt_num

    if max_ckpt_num is None:
        # no checkpoints present
        return None

    return os.path.join(logs_dir, f"checkpoint-{max_ckpt_num}")