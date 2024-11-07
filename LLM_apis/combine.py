import json

# List of JSON file paths Do this for NQ,TQ and NQP and TQP files also
"""
file_paths = [
    "PQP_updated_data_without_ref_gpt4_o.json",
    "PQP_updated_data_without_ref_gpt35_turbo2.json",
    "PQP_updated_data_without_ref_llamma8b_meta-llama_Meta-Llama-3_1-8B-Instruct-Turbo.json",
    "PQP_updated_data_without_ref_llamma70b_meta-llama_Meta-Llama-3_1-70B-Instruct-Turbo.json"
]
"""

data_list = []
for file_path in file_paths:
    with open(file_path, 'r') as f:
        data_list.append(json.load(f))

# Assuming both files have the same structure and the same number of entries
for i in range(len(data_list[0])):
    entry = data_list[0][i]
    best_model_scores = entry.get("best_model_scores", {})

    # Combine judge_ keys from both files into one
    for data in data_list:
        judge_keys = [key for key in data[i].keys() if key.startswith("judge_")]
        for key in judge_keys:
            best_model_scores[key] = data[i].pop(key)

    # Update the entry
    entry["best_model_scores"] = best_model_scores

# Output the combined result to a new JSON file
output_path = "PQP_LLM_metrics_output.json"
with open(output_path, 'w') as f:
    json.dump(data_list[0], f, indent=4)

output_path
