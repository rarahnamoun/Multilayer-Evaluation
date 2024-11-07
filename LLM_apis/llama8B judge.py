import os
import json
import time
from together import Together
from together.error import RateLimitError

# Initialize the Together client
client = Together(api_key="")#your api key

# JSON file path (update with actual file path)
json_file_path = ""
#TQ2P, TQ2,NQ2,NQ2P files

# Specify the model you want to use (70B model here)
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

# Convert the model name to a valid JSON key format (replace slashes and dots)
metric_col_name = model_name.replace("/", "_").replace(".", "_")

# Define a prompt for the Llama model to act as a judge with the gold answer
def generate_judge_prompt_with_gold(question, answer, gold_answers):
    # Combine all gold answers into a single string, separated by commas or new lines
    combined_gold_answers = ', '.join(gold_answers)
    prompt = f"""
    Given the question: '{question}', the predicted answer: '{answer}', and the correct gold answers: '{combined_gold_answers}',
    score the predicted answer from 0 to 1 based on its correctness and similarity to the gold answers.
    Only return the score in the format: score:<value>.
    """
    return prompt


# Define a prompt for the Llama model to act as a judge without the gold answer
def generate_judge_prompt_without_gold(question, answer):
    prompt = f"""
    Given the question: '{question}' and the predicted answer: '{answer}',
    score the predicted answer from 0 to 1 based on its correctness.
    Only return the score in the format: score:<value>.
    """
    return prompt


# Function to extract and validate score from the model's response
def extract_score(response_content):
    try:
        # Extract score value from "score:<value>"
        if "score:" in response_content:
            score_str = response_content.split("score:")[1].strip()
            score_value = float(score_str)
            return score_value
        else:
            # If response does not contain "score:", return 0
            return 0.0
    except Exception:
        # If any error occurs during extraction, return 0
        return 0.0


# Function to make API requests with retry logic
def make_request(prompt):
    while True:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": prompt}],
                max_tokens=50,
                temperature=0.7,
                top_p=0.7,
                top_k=50,
                repetition_penalty=1,
                stop=["<|eot_id|>", "<|eom_id|>"],
                stream=False
            )
            return response
        except RateLimitError as e:
            print(f"Rate limit hit: {e}. Waiting before retrying...")
            time.sleep(60)  # Wait for a minute before retrying


# Load the JSON file
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Initialize tracking file
tracking_file_path = 'progress_tracking30bllamaPQP.json'

# Load progress if tracking file exists
if os.path.exists(tracking_file_path):
    with open(tracking_file_path, 'r', encoding='utf-8') as f:
        processed_indices = set(json.load(f))
else:
    processed_indices = set()

# Define the delay (in seconds) between requests
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
    gold_answers = example['gold_answer']  # Assuming there are multiple gold answers in the list

    if answer is None:
        print(f"No predicted or raw answer found for question: {question}")
        continue

    # Prompt for scoring with the gold answers
    prompt_with_gold = generate_judge_prompt_with_gold(question, answer, gold_answers)
    response_with_gold = make_request(prompt_with_gold)

    # Extract and store the score with gold
    model_response_with_gold = response_with_gold.choices[0].message.content.strip()
    score_with_gold = extract_score(model_response_with_gold)  # Extract and validate score
    example[f'judge_{metric_col_name}_with_gold'] = score_with_gold
    data_with_gold.append(example.copy())
    print(f"With Gold - Model Response: {model_response_with_gold}, Extracted Score: {score_with_gold}")

    # Wait before making the next request to avoid hitting the rate limit
    time.sleep(REQUEST_DELAY)

    # Prompt for scoring without the gold answer
    prompt_without_gold = generate_judge_prompt_without_gold(question, answer)
    response_without_gold = make_request(prompt_without_gold)

    # Extract and store the score without gold
    model_response_without_gold = response_without_gold.choices[0].message.content.strip()
    score_without_gold = extract_score(model_response_without_gold)  # Extract and validate score
    example[f'judge_{metric_col_name}_without_gold'] = score_without_gold
    data_without_gold.append(example.copy())
    print(f"Without Gold - Model Response: {model_response_without_gold}, Extracted Score: {score_without_gold}")

    # Mark this example as processed
    processed_indices.add(i)

    # Save progress to the tracking file
    with open(tracking_file_path, 'w', encoding='utf-8') as f:
        json.dump(list(processed_indices), f, indent=4)

    # Save the updated JSON data incrementally
    output_file_with_gold = f'PQP_updated_data_with_ref_llamma8b_{metric_col_name}.json'
    output_file_without_gold = f'PQP_updated_data_without_ref_llamma8b_{metric_col_name}.json'

    with open(output_file_with_gold, 'w', encoding='utf-8') as f:
        json.dump(data_with_gold, f, indent=4)

    with open(output_file_without_gold, 'w', encoding='utf-8') as f:
        json.dump(data_without_gold, f, indent=4)

print(f"Updated JSON files saved as {output_file_with_gold} and {output_file_without_gold}")
