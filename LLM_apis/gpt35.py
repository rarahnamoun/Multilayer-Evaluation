import os
import json
import time
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(
    #api_key="", #Your api key
    #base_url=""  # Replace with your custom URL
)

# Define prompts for scoring with all gold answers
def generate_judge_prompt_with_gold(question, answer, gold_answers):
    # Join all gold answers into a single string
    gold_answers_text = " | ".join(gold_answers)
    return f"""
    Given the question: '{question}', the predicted answer: '{answer}', and the correct gold answers: '{gold_answers_text}',
    score the predicted answer from 0 to 1 based on its correctness and similarity to the gold answers.
    Just return the score in the format: score:<value>
    """

def generate_judge_prompt_without_gold(question, answer):
    return f"""
    Given the question: '{question}' and the predicted answer: '{answer}',
    score the predicted answer from 0 to 1 based on its correctness.
    Just return the score in the format: score:<value>
    """

# Function to get completion from OpenAI API
def get_openai_completion(prompt, model="gpt-3.5-turbo"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": prompt}],
            max_tokens=50,
            temperature=0.7,
            top_p=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error: {e}")
        return ""

# Extract score from response
def extract_score(response_content):
    try:
        if "score:" in response_content:
            score_str = response_content.split("score:")[1].strip()
            return float(score_str)
        else:
            return 0.0
    except Exception:
        return 0.0

# JSON file paths
input_json_file_path = ""#TQ2P, TQ2,NQ2,NQ2P files
output_file_with_gold = 'PQP_updated_data_with_ref_gpt35_turbo2.json'
output_file_without_gold = 'PQP_updated_data_without_ref_gpt35_turbo2.json'
tracking_file_path = 'progress_tracking1.json'

# Load the JSON file
with open(input_json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Load progress if tracking file exists
if os.path.exists(tracking_file_path):
    with open(tracking_file_path, 'r', encoding='utf-8') as f:
        processed_indices = set(json.load(f))
else:
    processed_indices = set()

REQUEST_DELAY = 1 

# Create two copies of the data: one for with gold, one for without
data_with_gold = []
data_without_gold = []

# Iterate through each example and generate scores for both scenarios
for i, example in enumerate(data):
    if i in processed_indices:
        continue  # Skip already processed examples

    question = example['question']
    answer = example.get('predicted_answer') or example.get('raw_answer')  # Check for either predicted_answer or raw_answer
    gold_answers = example['gold_answer']  # Use all gold answers from the list
    if not answer:
        print(f"No predicted or raw answer found for question: {question}")
        continue

    # Prompt for scoring with all gold answers
    prompt_with_gold = generate_judge_prompt_with_gold(question, answer, gold_answers)
    model_response_with_gold = get_openai_completion(prompt_with_gold)
    score_with_gold = extract_score(model_response_with_gold)
    example['judge_gpt35_turbo_with_gold'] = score_with_gold
    data_with_gold.append(example.copy())
    print(f"With Gold - Model Response: {model_response_with_gold}, Extracted Score: {score_with_gold}")


    time.sleep(REQUEST_DELAY)

    # Prompt for scoring without the gold answer
    prompt_without_gold = generate_judge_prompt_without_gold(question, answer)
    model_response_without_gold = get_openai_completion(prompt_without_gold)
    score_without_gold = extract_score(model_response_without_gold)
    example['judge_gpt35_turbo'] = score_without_gold
    data_without_gold.append(example.copy())
    print(f"Without Gold - Model Response: {model_response_without_gold}, Extracted Score: {score_without_gold}")

    # Mark this example as processed
    processed_indices.add(i)

    # Save progress to the tracking file
    with open(tracking_file_path, 'w', encoding='utf-8') as f:
        json.dump(list(processed_indices), f, indent=4)

    # Save the updated JSON data incrementally
    with open(output_file_with_gold, 'w', encoding='utf-8') as f:
        json.dump(data_with_gold, f, indent=4)

    with open(output_file_without_gold, 'w', encoding='utf-8') as f:
        json.dump(data_without_gold, f, indent=4)

print(f"Updated JSON files saved as {output_file_with_gold} and {output_file_without_gold}")
