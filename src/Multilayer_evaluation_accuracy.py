import json
import numpy as np

def load_json_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

def extract_metric_values(metric_name, scores):
    if metric_name in scores:
        value = scores[metric_name]
        if isinstance(value, dict):
            precision = value.get('precision', 0)
            recall = value.get('recall', 0)
            f1 = value.get('f1', 0)
            return {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        return value
    return 0

def extract_rouge_values(scores):
    rouge_scores = {}
    if 'ROUGEScore' in scores:
        rouge_2 = scores['ROUGEScore'].get('rouge-2', {})
        rouge_l = scores['ROUGEScore'].get('rouge-l', {})

        rouge_scores['rouge-2'] = {
            'precision': rouge_2.get('precision', 0),
            'recall': rouge_2.get('recall', 0),
            'f1': rouge_2.get('f1', 0)
        }

        rouge_scores['rouge-l'] = {
            'precision': rouge_l.get('precision', 0),
            'recall': rouge_l.get('recall', 0),
            'f1': rouge_l.get('f1', 0)
        }
    return rouge_scores

def calculate_accuracy(data, relevant_metrics):
    metrics = initialize_metrics(relevant_metrics)

    for entry in data:
        model = entry["model"]
        human_judge = entry.get("judge")
        if human_judge is None:
            continue
        human_judge = bool(human_judge)
        scores = entry["best_model_scores"]

        for metric in relevant_metrics:
            if metric == "ROUGEScore":
                rouge_scores = extract_rouge_values(scores)
                for rouge_type, rouge_values in rouge_scores.items():
                    for sub_metric in ['precision', 'recall', 'f1']:
                        metric_name = f"{rouge_type}_{sub_metric}"
                        value = rouge_values[sub_metric]
                        metric_decision = value >= 0.5
                        update_metric_counts(metrics, metric_name, metric_decision, human_judge, value)

            else:
                value = extract_metric_values(metric, scores)
                if isinstance(value, dict):
                    for sub_metric in ['precision', 'recall', 'f1']:
                        metric_name = f"{metric}_{sub_metric}"
                        sub_value = value.get(sub_metric, 0)
                        metric_decision = sub_value >= 0.5
                        update_metric_counts(metrics, metric_name, metric_decision, human_judge, sub_value)
                else:
                    metric_decision = value >= 0.5
                    update_metric_counts(metrics, metric, metric_decision, human_judge, value)

    return metrics

def initialize_metrics(all_metrics):
    metrics = {}
    for metric in all_metrics:
        if metric == "ROUGEScore":
            for rouge_type in ['rouge-2', 'rouge-l']:
                for sub_metric in ['precision', 'recall', 'f1']:
                    metric_name = f"{rouge_type}_{sub_metric}"
                    metrics[metric_name] = {'overall': [0, 0], 'yes': [0, 0], 'no': [0, 0],
                                            'above_or_equal_0.5': [0, 0], 'below_0.5': [0, 0]}
        elif metric in ["METEORScore", "BERTScore"]:
            for sub_metric in ['precision', 'recall', 'f1']:
                metric_name = f"{metric}_{sub_metric}"
                metrics[metric_name] = {'overall': [0, 0], 'yes': [0, 0], 'no': [0, 0],
                                        'above_or_equal_0.5': [0, 0], 'below_0.5': [0, 0]}
        else:
            metrics[metric] = {'overall': [0, 0], 'yes': [0, 0], 'no': [0, 0],
                               'above_or_equal_0.5': [0, 0], 'below_0.5': [0, 0]}
    return metrics

def update_metric_counts(metrics, metric_name, metric_decision, human_judge, value):
    if human_judge:
        if metric_decision == human_judge:
            metrics[metric_name]['yes'][0] += 1
        else:
            metrics[metric_name]['yes'][1] += 1
    else:
        if metric_decision == human_judge:
            metrics[metric_name]['no'][0] += 1
        else:
            metrics[metric_name]['no'][1] += 1

    if value >= 0.5:
        if metric_decision == human_judge:
            metrics[metric_name]['above_or_equal_0.5'][0] += 1
        else:
            metrics[metric_name]['above_or_equal_0.5'][1] += 1
    else:
        if metric_decision == human_judge:
            metrics[metric_name]['below_0.5'][0] += 1
        else:
            metrics[metric_name]['below_0.5'][1] += 1

def calculate_accuracies(metrics):
    accuracies = {metric: {} for metric in metrics}
    for metric, counts in metrics.items():
        correct, incorrect = counts['overall']
        total = correct + incorrect
        accuracies[metric]['overall'] = correct / total if total > 0 else 0

        correct, incorrect = counts['yes']
        total = correct + incorrect
        accuracies[metric]['yes'] = correct / total if total > 0 else 0

        correct, incorrect = counts['no']
        total = correct + incorrect
        accuracies[metric]['no'] = correct / total if total > 0 else 0

        correct, incorrect = counts['above_or_equal_0.5']
        total = correct + incorrect
        accuracies[metric]['above_or_equal_0.5'] = correct / total if total > 0 else 0

        correct, incorrect = counts['below_0.5']
        total = correct + incorrect
        accuracies[metric]['below_0.5'] = correct / total if total > 0 else 0

    return accuracies

def print_metrics_above_threshold_and_accuracy(data, relevant_metrics, threshold=0.5, accuracy_threshold=0.9):
    metrics = calculate_accuracy(data, relevant_metrics)
    accuracies = calculate_accuracies(metrics)

    filtered_metrics = {
        metric: acc['above_or_equal_0.5']
        for metric, acc in accuracies.items()
        if acc['above_or_equal_0.5'] > accuracy_threshold
    }

    for metric in filtered_metrics:
        print(f"Metric: {metric}, Accuracy above or equal to {threshold}: {filtered_metrics[metric]:.2f}")

    return filtered_metrics

def judge_output_based_on_filtered_metrics(data, filtered_metrics):
    judgments = {'correct': 0, 'incorrect': 0}
    remaining_judgments = 0
    calculated_judgments = 0

    remaining_data = []

    for entry in data:
        model = entry["model"]
        human_judge = entry.get("judge")
        if human_judge is None:
            continue
        human_judge = bool(human_judge)
        scores = entry["best_model_scores"]
        processed = False

        for metric in filtered_metrics:
            value = extract_metric_values(metric, scores)
            if isinstance(value, dict):
                for sub_metric in ['precision', 'recall', 'f1']:
                    sub_value = value.get(sub_metric, 0)
                    if sub_value >= 0.5:
                        metric_decision = 1
                        if metric_decision == human_judge:
                            judgments['correct'] += 1
                        else:
                            judgments['incorrect'] += 1
                        processed = True
                        calculated_judgments += 1
                        break
                if processed:
                    break
            else:
                if value >= 0.5:
                    metric_decision = 1
                    if metric_decision == human_judge:
                        judgments['correct'] += 1
                    else:
                        judgments['incorrect'] += 1
                    processed = True
                    calculated_judgments += 1
                    break

        if not processed:
            remaining_data.append(entry)
            remaining_judgments += 1

    total = judgments['correct'] + judgments['incorrect']
    accuracy = judgments['correct'] / total if total > 0 else 0
    print(f"Judgment Accuracy based on filtered metrics (Layer1): {accuracy:.2f}")
    print(f"Remaining human judgments not calculated: {remaining_judgments}")
    print(f"Calculated human judgments: {calculated_judgments}")
    return accuracy, remaining_judgments, remaining_data, calculated_judgments

def apply_layer_2_voting_and_evaluate(data, voting_metrics):
    judgments = {'correct': 0, 'incorrect': 0}
    remaining_judgments = 0
    processed_entries = 0

    for entry in data:
        human_judge = entry.get("judge")
        if human_judge is None:
            continue

        human_judge = bool(human_judge)
        scores = entry.get("best_model_scores", {})

        votes = 0
        total_metrics = len(voting_metrics)

        for metric in voting_metrics:
            value = extract_metric_values(metric, scores)
            if isinstance(value, dict):
                for sub_metric in ['precision', 'recall', 'f1']:
                    sub_value = value.get(sub_metric, 0)
                    if sub_value >= 0.5:
                        votes += 1
                        break
            else:
                if value >= 0.5:
                    votes += 1

        majority_vote = (votes / total_metrics) > 0.5
        metric_decision = int(majority_vote)

        if metric_decision == human_judge:
            judgments['correct'] += 1
        else:
            judgments['incorrect'] += 1

        processed_entries += 1

    total = judgments['correct'] + judgments['incorrect']
    accuracy = judgments['correct'] / total if total > 0 else 0

    print(f"Voting Accuracy (Layer 2): {accuracy:.2f}")
    print(f"Processed human judgments: {processed_entries}")

    return accuracy, remaining_judgments, processed_entries


def process_json_file(filename, relevant_metrics, layer_2_metrics, accuracy_threshold=0.96):
    data = load_json_file(filename)

    print(f"\nProcessing file: {filename}")

    # Print filtered metrics based on threshold
    filtered_metrics = print_metrics_above_threshold_and_accuracy(data, relevant_metrics,
                                                                  accuracy_threshold=accuracy_threshold)

    # Judge output based on filtered metrics
    accuracy, remaining_judgments, remaining_data, calculated_judgments = judge_output_based_on_filtered_metrics(data,
                                                                                                                 filtered_metrics)

    # Apply Layer 2 voting and evaluate
    layer_2_accuracy, layer_2_remaining_judgments, processed_entries = apply_layer_2_voting_and_evaluate(remaining_data,
                                                                                                         layer_2_metrics)

    # Print overall accuracy
    overall_accuracy = (processed_entries * layer_2_accuracy + calculated_judgments * accuracy) / (
                processed_entries + calculated_judgments)
    print(f"Overall accuracy for {filename}: {overall_accuracy:.2f}\n")

print("---------------LexicalMatchingOnly----------")
files = ['NQP2_LLM_metrics_output.json', 'TQP2_LLM_metrics_output.json']

# Define metrics to be used for accuracy calculation
relevant_metrics = ["ExactMatch", "BLEUScore"]
layer_2_metrics = ['WordMatchingMetric', 'Precision', 'Recall']

# Process each file
for file in files:
    process_json_file(file, relevant_metrics, layer_2_metrics)

print("---------------LexicalMatching+SemanticBased----------")
files = ['NQP2_LLM_metrics_output.json', 'TQP2_LLM_metrics_output.json']

# Define metrics to be used for accuracy calculation
relevant_metrics = ["ExactMatch", "BLEUScore"]
layer_2_metrics = ['SentenceTransformerSimilarity', 'BertEmbeddingMetric', 'Recall']

# Process each file
for file in files:
    process_json_file(file, relevant_metrics, layer_2_metrics)

print("---------------LexicalMatching+SemanticBased+Llama----------")
files = ['NQP2_LLM_metrics_output.json', 'TQP2_LLM_metrics_output.json', ]

# Define metrics to be used for accuracy calculation
relevant_metrics = ["ExactMatch", "BLEUScore"]
layer_2_metrics = ['SentenceTransformerSimilarity', 'judge_meta-llama_Meta-Llama-3_1-70B-Instruct-Turbo_with_gold', 'judge_meta-llama_Meta-Llama-3_1-8B-Instruct-Turbo_with_gold']

# Process each file
for file in files:
    process_json_file(file, relevant_metrics, layer_2_metrics)

print("---------------LexicalMatching+SemanticBased+Llama8B----------")
files = ['NQP2_LLM_metrics_output.json', 'TQP2_LLM_metrics_output.json']

# Define metrics to be used for accuracy calculation
relevant_metrics = ["ExactMatch", "BLEUScore"]
layer_2_metrics = ['SentenceTransformerSimilarity', 'BertEmbeddingMetric', 'judge_meta-llama_Meta-Llama-3_1-8B-Instruct-Turbo_with_gold']

# Process each file
for file in files:
    process_json_file(file, relevant_metrics, layer_2_metrics)

print("---------------Only LLMs----------")
files = ['NQP2_LLM_metrics_output.json', 'TQP2_LLM_metrics_output.json']

# Define metrics to be used for accuracy calculation
relevant_metrics = []
relevant_metrics = []
layer_2_metrics = ['judge_meta-llama_Meta-Llama-3_1-70B-Instruct-Turbo_with_gold', 'judge_gpt4_o_with_gold', 'judge_gpt35_turbo_with_gold']

# Process each file
for file in files:
    process_json_file(file, relevant_metrics, layer_2_metrics)
print("---------------Our Method Only One LLM----------")
files = ['NQP2_LLM_metrics_output.json']

# Define metrics to be used for accuracy calculation
relevant_metrics = ["ExactMatch", "BLEUScore"]
layer_2_metrics = ['judge_gpt4_o_with_gold', 'SentenceTransformerSimilarity', 'METEOR-recall']

# Process each file
for file in files:
    process_json_file(file, relevant_metrics, layer_2_metrics)
files = ['TQP2_LLM_metrics_output.json']

# Define metrics to be used for accuracy calculation
relevant_metrics = ["ExactMatch", "BLEUScore"]
layer_2_metrics = ['Recall', 'judge_gpt4_o_with_gold', 'SentenceTransformerSimilarity']



# Process each file
for file in files:
    process_json_file(file, relevant_metrics, layer_2_metrics)
print("---------------Our Method----------")
files = ['NQP2_LLM_metrics_output.json']

# Define metrics to be used for accuracy calculation
relevant_metrics = ["ExactMatch", "BLEUScore"]
layer_2_metrics = ['judge_gpt4_o_with_gold', 'SentenceTransformerSimilarity', 'METEOR-recall']

# Process each file
for file in files:
    process_json_file(file, relevant_metrics, layer_2_metrics)
files = ['TQP2_LLM_metrics_output.json']

# Define metrics to be used for accuracy calculation
relevant_metrics = ["ExactMatch", "BLEUScore"]
layer_2_metrics = ['judge_meta-llama_Meta-Llama-3_1-70B-Instruct-Turbo_with_gold', 'judge_gpt4_o_with_gold', 'SentenceTransformerSimilarity']


# Process each file
for file in files:
    process_json_file(file, relevant_metrics, layer_2_metrics)

print("---------------Our Method Default----------")
files = ['NQP2_LLM_metrics_output.json', 'TQP2_LLM_metrics_output.json', 'PQP_LLM_metrics_output.json']

# Define metrics to be used for accuracy calculation
relevant_metrics = ["ExactMatch", "BLEUScore"]
layer_2_metrics = ['judge_gpt4_o_with_gold', 'judge_meta-llama_Meta-Llama-3_1-70B-Instruct-Turbo_with_gold', 'Recall']

# Process each file
for file in files:
    process_json_file(file, relevant_metrics, layer_2_metrics)