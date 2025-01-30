import json
# import matplotlib.pyplot as plt
import pandas as pd

short_names = {
    'NQ2_LLM_metrics_output.json': 'NQ2',
    'TQ_LLM_metrics_output.json': 'TQ',
    'NQP2_LLM_metrics_output.json': 'NQP2',
    'TQP2_LLM_metrics_output.json': 'TQP2',
    'PQ_LLM_metrics_output.json': 'PQ',
    'PQP_LLM_metrics_output.json': 'PQP',

    # Metrics short names
    'ExactMatch': 'EM',
    'BLEUScore': 'BLEU',
    'METEORScore': 'METEOR',
    'BERTScore': 'BERT',
    'ROUGEScore': 'ROUGE',
    'SentenceTransformerSimilarity': 'STS',
    'BertEmbeddingMetric': 'BERT-EM',
    'WordMatchingMetric': 'WMM',
    'Precision': 'Prec',
    'Recall': 'Rec',

    # Judges short names
    'judge_gpt35_turbo_with_gold': 'GPT3.5-Gold',
    'judge_gpt35_turbo': 'GPT3.5-NoGold',
    'judge_meta-llama_Meta-Llama-3_1-70B-Instruct-Turbo_with_gold': 'Meta-Llama-70B-Gold',
    'judge_meta-llama_Meta-Llama-3_1-70B-Instruct-Turbo_without_gold': 'Meta-Llama-70B-NoGold',
    'judge_gpt4_o_with_gold': 'GPT4-Gold',
    'judge_gpt4_o_without_gold': 'GPT4-NoGold',
    'judge_meta-llama_Meta-Llama-3_1-8B-Instruct-Turbo_with_gold': 'Meta-Llama-8B-Gold',
    'judge_meta-llama_Meta-Llama-3_1-8B-Instruct-Turbo_without_gold': 'Meta-Llama-8B-NoGold'
}


# Load JSON data from file
def load_json_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)


# Function to load and process multiple JSON files
def load_and_process_files(filenames):
    all_accuracies = pd.DataFrame()

    for filename in filenames:
        data = load_json_file(filename)
        accuracies = calculate_accuracy(data)

        # Store or aggregate the accuracies
        df = pd.DataFrame(accuracies).T
        # Apply short names to dataset names
        short_filename = short_names.get(filename, filename)
        df.columns = [f"{short_filename}_{col}" for col in df.columns]

        if not all_accuracies.empty:
            all_accuracies = pd.concat([all_accuracies, df], axis=1)
        else:
            all_accuracies = df

    return all_accuracies


# Extract the metric values separately for precision, recall, and f1 if they exist
def extract_metric_values(metric_name, scores):
    if metric_name in scores:
        value = scores[metric_name]
        if isinstance(value, dict):
            # If precision, recall, and f1 exist, extract them separately
            # precision = value.get('precision', None)
            # recall = value.get('recall', None)
            f1 = value.get('f1', None)
            return {
                # 'precision': precision if precision is not None else 0,
                # 'recall': recall if recall is not None else 0,
                'f1': f1 if f1 is not None else 0
            }
        return value
    return 0


# Specialized extraction for ROUGEScore since it contains rouge-2 and rouge-l
def extract_rouge_values(scores):
    rouge_scores = {}
    if 'ROUGEScore' in scores:
        rouge_2 = scores['ROUGEScore'].get('rouge-2', {})
        rouge_l = scores['ROUGEScore'].get('rouge-l', {})

        rouge_scores['rouge-2'] = {
            # 'precision': rouge_2.get('precision', 0),
            # 'recall': rouge_2.get('recall', 0),
            'f1': rouge_2.get('f1', 0)
        }

        rouge_scores['rouge-l'] = {
            # 'precision': rouge_l.get('precision', 0),
            # 'recall': rouge_l.get('recall', 0),
            'f1': rouge_l.get('f1', 0)
        }
    return rouge_scores


# Calculate accuracy for each metric based on human judge comparison
def calculate_accuracy(data):
    all_metrics = ["ExactMatch", "BLEUScore", "METEORScore", "BERTScore", "ROUGEScore",
                   "SentenceTransformerSimilarity", "BertEmbeddingMetric", "WordMatchingMetric",
                   "Precision", "Recall"]

    judge_metrics = ["judge_gpt35_turbo_with_gold", "judge_gpt35_turbo",
                     "judge_meta-llama_Meta-Llama-3_1-70B-Instruct-Turbo_with_gold",
                     "judge_meta-llama_Meta-Llama-3_1-70B-Instruct-Turbo_without_gold",
                     "judge_gpt4_o_with_gold", "judge_gpt4_o_without_gold",
                     "judge_meta-llama_Meta-Llama-3_1-8B-Instruct-Turbo_with_gold",
                     "judge_meta-llama_Meta-Llama-3_1-8B-Instruct-Turbo_without_gold"]

    metrics = initialize_metrics(all_metrics + judge_metrics)

    for entry in data:
        human_judge = entry.get("judge")  # Get the judge value
        if human_judge is None:  # Skip entries with null judgments
            continue
        human_judge = bool(human_judge)  # Ensure human_judge is a boolean
        scores = entry.get("best_model_scores", {})

        for metric in all_metrics:
            if metric == "ROUGEScore":
                rouge_scores = extract_rouge_values(scores)
                # Handle rouge-2 and rouge-l separately
                for rouge_type, rouge_values in rouge_scores.items():
                    for sub_metric in ['f1']:
                        metric_name = f"{rouge_type}"
                        value = rouge_values[sub_metric]
                        metric_decision = value >= 0.5

                        # Update counts for accuracy calculations
                        if metric_decision == human_judge:
                            metrics[metric_name]['overall'][0] += 1  # Correct
                        else:
                            metrics[metric_name]['overall'][1] += 1  # Incorrect

                        # Further updates for "yes" and "no" judgments, and scores above/below 0.5
                        update_metric_subscores(metrics, metric_name, metric_decision, human_judge, value)

            else:
                # Handle other metrics
                value = extract_metric_values(metric, scores)
                if isinstance(value, dict):
                    for sub_metric in ['f1']:
                        metric_name = f"{metric}"
                        sub_value = value.get(sub_metric, 0)
                        metric_decision = sub_value >= 0.5

                        if metric_decision == human_judge:
                            metrics[metric_name]['overall'][0] += 1  # Correct
                        else:
                            metrics[metric_name]['overall'][1] += 1  # Incorrect

                        # Further updates for "yes" and "no" judgments, and scores above/below 0.5
                        update_metric_subscores(metrics, metric_name, metric_decision, human_judge, sub_value)
                else:
                    # For metrics like BLEUScore and ExactMatch which are simple values
                    metric_decision = value >= 0.5

                    if metric_decision == human_judge:
                        metrics[metric]['overall'][0] += 1  # Correct
                    else:
                        metrics[metric]['overall'][1] += 1  # Incorrect

                    # Further updates for "yes" and "no" judgments, and scores above/below 0.5
                    update_metric_subscores(metrics, metric, metric_decision, human_judge, value)

        # Handle judge metrics
        for judge_metric in judge_metrics:
            judge_value = entry.get("best_model_scores", {}).get(judge_metric, 0)
            judge_decision = judge_value >= 0.5

            if judge_decision == human_judge:
                metrics[judge_metric]['overall'][0] += 1  # Correct
            else:
                metrics[judge_metric]['overall'][1] += 1  # Incorrect

            # Further updates for "yes" and "no" judgments, and scores above/below 0.5
            update_metric_subscores(metrics, judge_metric, judge_decision, human_judge, judge_value)

    # Calculate and return the accuracies
    return calculate_accuracies(metrics)


def initialize_metrics(all_metrics):
    metrics = {}
    for metric in all_metrics:
        if metric == "ROUGEScore":
            for rouge_type in ['rouge-2', 'rouge-l']:
                for sub_metric in ['f1']:
                    metric_name = f"{rouge_type}"
                    metrics[metric_name] = {'overall': [0, 0], 'yes': [0, 0], 'no': [0, 0],
                                            'above_or_equal_0.5': [0, 0], 'below_0.5': [0, 0]}
        elif metric in ["METEORScore", "BERTScore"]:
            for sub_metric in ['f1']:
                metric_name = f"{metric}"
                metrics[metric_name] = {'overall': [0, 0], 'yes': [0, 0], 'no': [0, 0],
                                        'above_or_equal_0.5': [0, 0], 'below_0.5': [0, 0]}
        else:
            metrics[metric] = {'overall': [0, 0], 'yes': [0, 0], 'no': [0, 0],
                               'above_or_equal_0.5': [0, 0], 'below_0.5': [0, 0]}
    return metrics


def update_metric_subscores(metrics, metric_name, metric_decision, human_judge, value):
    # Accuracy for "Yes" judgments
    """
    if human_judge:
        if metric_decision == human_judge:
            metrics[metric_name]['yes'][0] += 1  # Correct
        else:
            metrics[metric_name]['yes'][1] += 1  # Incorrect

    # Accuracy for "No" judgments
    else:
        if metric_decision == human_judge:
            metrics[metric_name]['no'][0] += 1  # Correct
        else:
            metrics[metric_name]['no'][1] += 1  # Incorrect
    """
    # Accuracy for values above 0.5
    if value >= 0.5:
        if metric_decision == human_judge:
            metrics[metric_name]['above_or_equal_0.5'][0] += 1  # Correct
        else:
            metrics[metric_name]['above_or_equal_0.5'][1] += 1  # Incorrect

    # Accuracy for values below or equal to 0.5
    else:
        if metric_decision == human_judge:
            metrics[metric_name]['below_0.5'][0] += 1  # Correct
        else:
            metrics[metric_name]['below_0.5'][1] += 1  # Incorrect



def calculate_accuracies(metrics):
    accuracies = {metric: {} for metric in metrics}
    for metric, counts in metrics.items():
        """
        # Overall accuracy
        correct, incorrect = counts['overall']
        total = correct + incorrect
        accuracies[metric]['overall'] = correct / total if total > 0 else 0
      
        # Accuracy for "Yes" judgments
        yes_correct, yes_incorrect = counts['yes']
        total_yes = yes_correct + yes_incorrect
        accuracies[metric]['yes'] = yes_correct / total_yes if total_yes > 0 else 0

        # Accuracy for "No" judgments
        no_correct, no_incorrect = counts['no']
        total_no = no_correct + no_incorrect
        accuracies[metric]['no'] = no_correct / total_no if total_no > 0 else 0
"""
        # Accuracy for values above or equal to 0.5
        above_or_equal_0_5_correct, above_or_equal_0_5_incorrect = counts['above_or_equal_0.5']
        total_above_or_equal_0_5 = above_or_equal_0_5_correct + above_or_equal_0_5_incorrect
        accuracies[metric][
            'above_or_equal_0.5'] = above_or_equal_0_5_correct / total_above_or_equal_0_5 if total_above_or_equal_0_5 > 0 else 0

        # Accuracy for values below or equal to 0.5
        below_0_5_correct, below_0_5_incorrect = counts['below_0.5']
        total_below_0_5 = below_0_5_correct + below_0_5_incorrect
        accuracies[metric]['below_0.5'] = below_0_5_correct / total_below_0_5 if total_below_0_5 > 0 else 0

    return accuracies


def escape_latex(text):
    # Replace special LaTeX characters with their escaped versions
    replacements = {
        '_': r'\_',
        '%': r'\%',
        '$': r'\$',
        '&': r'\&',
        '#': r'\#',
        '^': r'\textasciicircum{}',
        '~': r'\textasciitilde{}',
        '\\': r'\textbackslash{}'
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


def escape_latex(s):
    """Escape special characters for LaTeX."""
    return s.replace('_', r'\_').replace('&', r'\&').replace('%', r'\%').replace('#', r'\#')


def generate_latex_table(accuracies):
    # Create a DataFrame from the accuracies dictionary
    df = pd.DataFrame(accuracies).T
    df = df.round(2)  # Round to two decimal places

    # Transpose the DataFrame
    df = df.T

    # Filter out columns that contain 'yes' and 'no'
    columns_to_exclude = [col for col in df.columns if 'yes' in col or 'no' in col]
    df = df.drop(columns=columns_to_exclude, errors='ignore')

    # Start the LaTeX table
    latex_table = r"\begin{table}[h!]" + "\n"
    latex_table += r"\centering" + "\n"
    latex_table += r"\small" + "\n"  # Apply smaller font size
    latex_table += r"\resizebox{\textwidth}{!}{" + "\n"  # Resize table to text width
    latex_table += r"\begin{tabular}{|l|" + "c|" * len(df.columns) + "}" + "\n"
    latex_table += r"\hline" + "\n"

    # Table header
    headers = ['Metric'] + list(df.columns)
    latex_table += " & ".join(headers) + r" \\" + "\n"
    latex_table += r"\hline" + "\n"

    # Table rows
    for index, row in df.iterrows():
        row_str = escape_latex(index) + " & " + " & ".join(f"{val:.2f}" for val in row) + r" \\" + "\n"
        latex_table += row_str
        latex_table += r"\hline" + "\n"

    # End the LaTeX table
    latex_table += r"\end{tabular}}" + "\n"  # Close the resizebox environment
    latex_table += r"\caption{Accuracy metrics for different evaluation methods.}" + "\n"
    latex_table += r"\label{tab:accuracy_metrics}" + "\n"
    latex_table += r"\end{table}" + "\n"

    return latex_table


# Print accuracy tables using Pandas DataFrame
def print_accuracy_tables(accuracies):
    # Set pandas display options to ensure the table is shown completely
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', None)  # Adjust display width to terminal
    pd.set_option('display.max_colwidth', None)  # Set no limit on column width

    df = pd.DataFrame(accuracies).T
    df = df.round(2)  # Round to two decimal places

    print("Accuracies for Metrics (Overall, Yes, No Judgments):")
    print(df)

    # Reset display options to default (optional)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.max_colwidth')


# Main function
def main():
    # List of dataset filenames
    filenames = [
        'NQ2_LLM_metrics_output.json', 'NQP2_LLM_metrics_output.json',
        'TQ_LLM_metrics_output.json', 'TQP2_LLM_metrics_output.json',
        'PQ_LLM_metrics_output.json', 'PQP_LLM_metrics_output.json',
    ]

    # Load and process data from multiple files
    aggregated_accuracies = load_and_process_files(filenames)

    # Print the table
    print_accuracy_tables(aggregated_accuracies)

    # Generate and print LaTeX table
    latex_table = generate_latex_table(aggregated_accuracies)
    print(latex_table)


if __name__ == "__main__":
    main()
