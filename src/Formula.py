import pandas as pd
import numpy as np
import json
from scipy.stats import pearsonr, spearmanr, kendalltau


# Load and process a single file
def load_and_process_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)

    records = []
    for entry in data:
        metrics = entry['best_model_scores']
        record = {
            'ExactMatch': metrics.get('ExactMatch', 0) > 0.5,
            'BLEUScore': metrics.get('BLEUScore', 0) > 0.5,
            'ROUGE-2-precision': metrics.get('ROUGEScore', {}).get('rouge-2', {}).get('precision', 0) > 0.5,
            'ROUGE-2-recall': metrics.get('ROUGEScore', {}).get('rouge-2', {}).get('recall', 0) > 0.5,
            'ROUGE-2-f1': metrics.get('ROUGEScore', {}).get('rouge-2', {}).get('f1', 0) > 0.5,
            'ROUGE-L-precision': metrics.get('ROUGEScore', {}).get('rouge-l', {}).get('precision', 0) > 0.5,
            'ROUGE-L-recall': metrics.get('ROUGEScore', {}).get('rouge-l', {}).get('recall', 0) > 0.5,
            'ROUGE-L-f1': metrics.get('ROUGEScore', {}).get('rouge-l', {}).get('f1', 0) > 0.5,
            'METEOR-meteor': metrics.get('METEORScore', {}).get('meteor', 0) > 0.5,
            'METEOR-precision': metrics.get('METEORScore', {}).get('precision', 0) > 0.5,
            'METEOR-recall': metrics.get('METEORScore', {}).get('recall', 0) > 0.5,
            'METEOR-f1': metrics.get('METEORScore', {}).get('f1', 0) > 0.5,
            'BERT-precision': metrics.get('BERTScore', {}).get('precision', 0) > 0.5,
            'BERT-recall': metrics.get('BERTScore', {}).get('recall', 0) > 0.5,
            'BERT-f1': metrics.get('BERTScore', {}).get('f1', 0) > 0.5,
            'Precision': metrics.get('Precision', 0) > 0.5,
            'Recall': metrics.get('Recall', 0) > 0.5,
            'SentenceTransformerSimilarity': metrics.get('SentenceTransformerSimilarity', 0) > 0.5,
            'BertEmbeddingMetric': metrics.get('BertEmbeddingMetric', 0) > 0.5,
            'WordMatchingMetric': metrics.get('WordMatchingMetric', 0) > 0.5,
            'judge_gpt4_o_with_gold': metrics.get('judge_gpt4_o_with_gold', 0) > 0.5,
            'judge_gpt4_o_without_gold': metrics.get('judge_gpt4_o_without_gold', 0) > 0.5,
            'judge_gpt35_turbo_with_gold': metrics.get('judge_gpt35_turbo_with_gold', 0) > 0.5,
            'judge_gpt35_turbo': metrics.get('judge_gpt35_turbo', 0) > 0.5,
            'judge_meta-llama_Meta-Llama-3_1-8B-Instruct-Turbo_with_gold': metrics.get(
                'judge_meta-llama_Meta-Llama-3_1-8B-Instruct-Turbo_with_gold', 0) > 0.5,
            'judge_meta-llama_Meta-Llama-3_1-8B-Instruct-Turbo_without_gold': metrics.get(
                'judge_meta-llama_Meta-Llama-3_1-8B-Instruct-Turbo_without_gold', 0) > 0.5,
            'judge_meta-llama_Meta-Llama-3_1-70B-Instruct-Turbo_with_gold': metrics.get(
                'judge_meta-llama_Meta-Llama-3_1-70B-Instruct-Turbo_with_gold', 0) > 0.5,
            'judge_meta-llama_Meta-Llama-3_1-70B-Instruct-Turbo_without_gold': metrics.get(
                'judge_meta-llama_Meta-Llama-3_1-70B-Instruct-Turbo_without_gold', 0) > 0.5,
            'Judge': entry.get('judge', np.nan)
        }
        records.append(record)

    df = pd.DataFrame(records)
    df['Judge'] = df['Judge'].fillna(-1).astype(int)
    df = df[df['Judge'] != -1]
    return df

