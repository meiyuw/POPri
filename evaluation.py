import hydra
from omegaconf import OmegaConf, DictConfig
from scipy.linalg import sqrtm
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
import gc
import torch
from peft import PeftModel


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    """Calculate the Frechet Distance between two multivariate Gaussians."""
    diff = mu1 - mu2
    # Product might be almost singular
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    # Check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # Calculate FID
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


def calculate_fid(set1, set2):
    # Calculate mean and covariance for both sets
    mu1, sigma1 = set1.mean(axis=0), np.cov(set1, rowvar=False)
    mu2, sigma2 = set2.mean(axis=0), np.cov(set2, rowvar=False)

    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid


def calculate_mean_fid(set1, set2):
    # Calculate mean and covariance for both sets
    mu1 = set1.mean(axis=0)
    mu2 = set2.mean(axis=0)
    diff = mu1 - mu2
    fid = diff.dot(diff)
    return fid


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(cfg: DictConfig):
    round_directory = os.path.join(
        cfg.logging.root_dir,
        "trial{0}/round{1}".format(cfg.logging.trial, cfg.round_number),
    )
    embedding_model = SentenceTransformer(
        cfg.embedding_model.model,
        cache_folder=cfg.embedding_model.save_dir,
    )

    with open(cfg.dataset.eval_data, "r", encoding="utf8") as f:
        eval_samples = json.load(f)
    eval_encodings = embedding_model.encode(eval_samples)
    eval_encodings = np.vstack(eval_encodings)

    with open(
        os.path.join(round_directory, f"fid_checkpoint.json"),
        "r",
        encoding="utf8",
    ) as f:
        generated_samples = json.load(f)
    gen_encodings = embedding_model.encode(generated_samples)
    gen_encodings = np.vstack(gen_encodings)
    gen_fid = calculate_fid(gen_encodings, eval_encodings)
    mean_fid = calculate_mean_fid(gen_encodings, eval_encodings)
    result = {"FID": float(gen_fid), "MSE": float(mean_fid)}
    print("Round {0}".format(cfg.round_number), result)
    with open(
        os.path.join(round_directory, f"result.json"), "w+", encoding="utf8"
    ) as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
