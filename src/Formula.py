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


# Calculation functions remain the same
def calculate_accuracy(df):
    all_metrics = df.columns[:-1]  # Exclude 'Judge' column
    metrics = {metric: {'overall': [0, 0]} for metric in all_metrics}

    for i in range(len(df)):
        row = df.iloc[i]
        human_judge = row['Judge']

        for metric in all_metrics:
            value = row[metric]
            metric_decision = value >= 0.5

            if metric_decision == human_judge:
                metrics[metric]['overall'][0] += 1  # Correct
            else:
                metrics[metric]['overall'][1] += 1  # Incorrect

    accuracies = {}
    for metric, counts in metrics.items():
        correct, incorrect = counts['overall']
        total = correct + incorrect
        accuracies[metric] = correct / total if total > 0 else 0

    return accuracies


def compute_correlations(df):
    metrics = df.columns[:-1]  # Exclude 'Judge' column
    correlation_results = pd.DataFrame(index=metrics, columns=metrics)

    # Convert boolean columns to integers
    df = df.astype(int)

    for metric1 in metrics:
        for metric2 in metrics:
            if metric1 != metric2:
                # Compute Pearson correlation
                pearson_corr, _ = pearsonr(df[metric1], df[metric2])
                # Compute Spearman correlation
                spearman_corr, _ = spearmanr(df[metric1], df[metric2])
                # Compute Kendall Tau correlation
                kendall_corr, _ = kendalltau(df[metric1], df[metric2])
                # Average correlation
                avg_corr = np.mean([pearson_corr, spearman_corr, kendall_corr])
                correlation_results.loc[metric1, metric2] = avg_corr

    return correlation_results


def compute_judge_correlations(df):
    metrics = df.columns[:-1]  # Exclude 'Judge' column
    judge_correlations = {}

    # Convert boolean columns to integers
    df = df.astype(int)

    for metric in metrics:
        # Compute Pearson correlation with 'Judge'
        pearson_corr, _ = pearsonr(df[metric], df['Judge'])
        # Compute Spearman correlation with 'Judge'
        spearman_corr, _ = spearmanr(df[metric], df['Judge'])
        # Compute Kendall Tau correlation with 'Judge'
        kendall_corr, _ = kendalltau(df[metric], df['Judge'])
        # Average correlation
        avg_corr = np.mean([pearson_corr, spearman_corr, kendall_corr])
        judge_correlations[metric] = avg_corr

    return judge_correlations

