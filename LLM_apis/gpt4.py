import os
import json
import time
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(
    #api_key="",  #your api key
    #base_url=""  # Replace with your  URL
)

# Define prompts for scoring
def generate_judge_prompt_with_gold(question, answer, gold_answers):
    # Join all gold answers with a separator, like "or" or commas
    gold_answers_str = " or ".join(gold_answers) if len(gold_answers) > 1 else gold_answers[0]
    return f"""
    Given the question: '{question}', the predicted answer: '{answer}', and the correct gold answers: '{gold_answers_str}',
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
def get_openai_completion(prompt, model="gpt-4o"):
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
input_json_file_path = ''#TQ2P, TQ2,NQ2,NQ2P files
output_file_with_gold = 'PQ_updated_data_with_ref_gpt4_o.json'
output_file_without_gold = 'PQ_updated_data_without_ref_gpt4_o.json'
tracking_file_path = 'progress_tracking2.json'

# Load the JSON file
with open(input_json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Load progress if tracking file exists
if os.path.exists(tracking_file_path):
    with open(tracking_file_path, 'r', encoding='utf-8') as f:
        processed_indices = set(json.load(f))
else:
    processed_indices = set()


REQUEST_DELAY = 1  # Adjust as needed

# Create two copies of the data: one for with gold, one for without
data_with_gold = []
data_without_gold = []

# Iterate through each example and generate scores for both scenarios
for i, example in enumerate(data):
    if i in processed_indices:
        continue  # Skip already processed examples

    question = example['question']
    answer = example.get('predicted_answer') or example.get('raw_answer')  # Check for either predicted_answer or raw_answer
    gold_answers = example.get('gold_answer', [])  # Ensure it's a list

    if not answer:
        print(f"No predicted or raw answer found for question: {question}")
        continue

    if not gold_answers:
        print(f"No gold answers found for question: {question}")
        continue

    # Prompt for scoring with the gold answers
    prompt_with_gold = generate_judge_prompt_with_gold(question, answer, gold_answers)
    model_response_with_gold = get_openai_completion(prompt_with_gold)
    score_with_gold = extract_score(model_response_with_gold)
    example['judge_gpt4_o_with_gold'] = score_with_gold
    data_with_gold.append(example.copy())
    print(f"With Gold - Model Response: {model_response_with_gold}, Extracted Score: {score_with_gold}")

    # Wait before making the next request to avoid hitting the rate limit
    time.sleep(REQUEST_DELAY)

    # Prompt for scoring without the gold answer
    prompt_without_gold = generate_judge_prompt_without_gold(question, answer)
    model_response_without_gold = get_openai_completion(prompt_without_gold)
    score_without_gold = extract_score(model_response_without_gold)
    example['judge_gpt4_o_without_gold'] = score_without_gold
    data_without_gold.append(example.copy())
    print(f"Without Gold - Model Response: {model_response_without_gold}, Extracted Score: {score_without_gold}")


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
