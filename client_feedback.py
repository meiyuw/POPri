import hydra
from omegaconf import OmegaConf, DictConfig
import os
import sys
import numpy as np
import json
import random
from sentence_transformers import SentenceTransformer, util


def neural_scores_for_policy(
    policy_outputs_list, client_text_embeddings, embedding_model, noise, clip=True
):
    """Computes the score of each text by the policy model via neural embedding model

    Args:
        policy_outputs_list: A list of strings to be scored.
        client_text_embeddings: A dictionary of client_id -> np array of size [num_texts, embed_size]
        clients: A dictionary of client_id -> list of strings representing the strings at each client.
        embedding_model: A sentence_transformers model that embeds strings
    Returns:
        A np array of size len(policy_outputs_list) which scores each policy output on how well it
        represents the client data (higher is better)
    """
    policy_outputs_embeddings = embedding_model.encode(policy_outputs_list)
    policy_outputs_embeddings = np.vstack(policy_outputs_embeddings)
    sum_scores = np.zeros(len(policy_outputs_list))
    total_data = 0.0
    for _, text_embeddings in client_text_embeddings.items():
        cos_sims = util.cos_sim(policy_outputs_embeddings, text_embeddings).numpy()
        cos_sims_agg = np.sum(cos_sims, axis=1)
        if clip:
            scaled_scores = (cos_sims_agg - np.mean(cos_sims_agg)) / np.std(
                cos_sims_agg
            )
            scores_norm = np.linalg.norm(scaled_scores)
            if scores_norm > 1.0:
                sum_scores += scaled_scores / scores_norm
            else:
                sum_scores += scaled_scores
        else:
            sum_scores += cos_sims_agg
        total_data += 1.0

    sum_scores += np.random.normal(size=len(policy_outputs_list)) * noise
    return sum_scores / total_data


def chunk_sort(A, B, rank_samples):
    chunks = [A[i : i + rank_samples] for i in range(0, len(A), rank_samples)]
    # Step 2: Sort each chunk based on scores from B
    sorted_chunks = []
    for chunk in chunks:
        # Sort the chunk using the scores in B
        sorted_chunk = sorted(chunk, key=lambda x: B[A.index(x)], reverse=True)
        sorted_chunks.append(sorted_chunk)
    return sorted_chunks


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(cfg: DictConfig):
    print(
        f"#####################\n# Client feedback round {cfg.round_number}\n#####################",
        file=sys.stderr,
    )
    round_directory = os.path.join(
        cfg.logging.root_dir, "trial{0}/round{1}".format(cfg.logging.trial, cfg.round_number)
    )

    # load the prompt set and the corresponding synthetic data
    with open(
        os.path.join(round_directory, "generated_samples.json"), "r", encoding="utf8"
    ) as f:
        policy_outputs = json.load(f)
    with open(
        os.path.join(round_directory, "prompt_set.json"), "r", encoding="utf8"
    ) as f:
        prompt_set = json.load(f)

    # load the client data
    with open(cfg.dataset.client_data, "r", encoding="utf8") as f:
        client_data_full = json.load(f)

    # handling client sampling
    if cfg.federated.per_round_clients < cfg.federated.total_clients:
        sub_key = random.sample(
            list(client_data_full.keys()), cfg.federated.per_round_clients
        )
        client_data = {key: client_data_full[key] for key in sub_key}
    else:
        sub_key = list(client_data_full.keys())
        client_data = client_data_full

    with open(
        os.path.join(round_directory, "selected_train_data_key.json"),
        "w+",
        encoding="utf8",
    ) as f:
        json.dump(sub_key, f, ensure_ascii=False)

    print('Number of clients', len(client_data))
    # score the synthetic samples
    embedding_model = SentenceTransformer(
        cfg.embedding_model.model,
        cache_folder=cfg.embedding_model.save_dir,
        device="cuda:0",
    )
    # Flatten texts and keep track of boundaries
    all_texts = []
    boundaries = {}
    index = 0
 
    for client_id, texts in client_data.items():
        all_texts.extend(texts)
        boundaries[client_id] = (index, index + len(texts))
        index += len(texts)

    # Encode all texts at once
    all_embeddings = embedding_model.encode(all_texts, show_progress_bar=True)

    # Reconstruct client-wise embeddings
    client_text_embeddings = {}
    for client_id, (start, end) in boundaries.items():
        client_text_embeddings[client_id] = np.vstack(all_embeddings[start:end])


    print('Finished encoding sentences')
    noise_scores = neural_scores_for_policy(
        policy_outputs,
        client_text_embeddings,
        embedding_model,
        cfg.privacy.noise,
        clip=True,
    )
    real_scores = neural_scores_for_policy(
        policy_outputs, client_text_embeddings, embedding_model, 0.0, clip=False
    )

    pair_real = zip(real_scores, policy_outputs)
    string_to_real_score = {string: score for score, string in pair_real}
    policy_output_ranked_by_prompt = chunk_sort(policy_outputs, noise_scores, cfg.dataset.rank_samples)

    assert len(policy_output_ranked_by_prompt) == cfg.dataset.num_pref_pairs
    assert len(prompt_set) == cfg.dataset.num_pref_pairs
    correct_pairs = 0
    preference_dataset = []
    preference_rank_dataset = [[] for i in range(cfg.dataset.rank_samples)]
    for i in range(cfg.dataset.num_pref_pairs):
        curr_prompt_responses_ranked = policy_output_ranked_by_prompt[i]
        curr_top_sample = curr_prompt_responses_ranked[0]
        curr_bottom_sample = curr_prompt_responses_ranked[cfg.dataset.reject_rank] 
        preference_dataset.append({'prompt': prompt_set[i], 'chosen': curr_top_sample, 'rejected': curr_bottom_sample})
        for r in range(cfg.dataset.rank_samples):
            preference_rank_dataset[r].append({'prompt': prompt_set[i], 'chosen': curr_top_sample, 'rejected': curr_prompt_responses_ranked[r]})
        if string_to_real_score[curr_top_sample] > string_to_real_score[curr_bottom_sample]:
            correct_pairs += 1
    
    correct_percentage = correct_pairs / cfg.dataset.num_pref_pairs
    print('Recovery percentage', correct_percentage, file=sys.stderr)

    with open(os.path.join(round_directory, 'preference_dataset.json'), 'w+', encoding='utf8') as f:
        json.dump(preference_dataset, f, ensure_ascii=False)
    with open(os.path.join(round_directory, 'recovery_percentage.json'), 'w+', encoding='utf8') as f:
        json.dump({'Recovery percentage': correct_percentage}, f, ensure_ascii=False)

if __name__ == "__main__":
    main()