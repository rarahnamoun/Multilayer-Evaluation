import nltk
nltk.download('wordnet')
import numpy as np
from collections import Counter
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from sklearn.metrics import f1_score as sklearn_f1_score
import torch
import bert_score
import nltk
import torch.nn.functional as F
from word2number import w2n
import string
import random


# Download the necessary NLTK data
nltk.download('punkt')

# Define the new metric class for Sentence Transformer similarity
class SentenceTransformerSimilarity:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def compute(self, predictions: list, references: list) -> dict:
        results = []
        for pred, ref in zip(predictions, references):
            sentences = [pred, ref]

            # Tokenize sentences
            encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

            # Compute token embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)

            # Perform pooling
            sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

            # Normalize embeddings
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

            # Calculate cosine similarities
            cosine_similarities = torch.mm(sentence_embeddings, sentence_embeddings.T)

            # Get the similarity value
            similarity = cosine_similarities[0, 1].item()
            results.append(similarity)

        # Return the average similarity if there are multiple predictions
        return np.mean(results) if results else  0.0

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

# Metric Classes (Existing metrics)

class ExactMatch:
    def compute(self, prediction: str, reference: str) -> float:
        return float(str(prediction).strip().lower() == str(reference).strip().lower())

class F1Score:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def compute(self, predictions: list, references: list) -> float:
        pred_tokens = [self._tokenize(p) for p in predictions]
        ref_tokens = [self._tokenize(r) for r in references]

        # Flatten token lists
        pred_tokens = [item for sublist in pred_tokens for item in sublist]
        ref_tokens = [item for sublist in ref_tokens for item in sublist]

        common_tokens = Counter(pred_tokens) & Counter(ref_tokens)
        num_common = sum(common_tokens.values())

        if num_common == 0:
            return 0.0

        precision = num_common / len(pred_tokens)
        recall = num_common / len(ref_tokens)
        return 2 * (precision * recall) / (precision + recall)

    def _tokenize(self, text: str):
        if self.tokenizer:
            return self.tokenizer.tokenize(text)
        return text.split()

class BLEUScore:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.smoothie = SmoothingFunction().method4

    def compute(self, predictions: list, references: list) -> float:
        pred_tokens = [self._tokenize(p) for p in predictions]
        ref_tokens = [self._tokenize(r) for r in references]

        scores = [sentence_bleu([ref], pred, smoothing_function=self.smoothie)
                  for pred, ref in zip(pred_tokens, ref_tokens)]

        return np.mean(scores) if scores else 0.0

    def _tokenize(self, text: str):
        if self.tokenizer:
            return self.tokenizer.tokenize(text)
        return text.split()

class ROUGEScore:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def compute(self, predictions: list, references: list) -> dict:
        pred_tokens = [self._tokenize(p) for p in predictions]
        ref_tokens = [self._tokenize(r) for r in references]

        # Initialize accumulators for ROUGE metrics
        rouge_2_results = []
        rouge_l_results = []

        # Calculate ROUGE-2 and ROUGE-L for each prediction-reference pair
        for p_tokens, r_tokens in zip(pred_tokens, ref_tokens):
            rouge_2 = self._compute_rouge_n(p_tokens, r_tokens, n=2)
            rouge_l = self._compute_rouge_l(p_tokens, r_tokens)

            rouge_2_results.append(rouge_2)
            rouge_l_results.append(rouge_l)

        # Calculate average ROUGE-2 and ROUGE-L scores
        avg_rouge_2 = self._average_metrics(rouge_2_results)
        avg_rouge_l = self._average_metrics(rouge_l_results)

        return {
            'rouge-2': {

                'precision': avg_rouge_2['precision'],
                'recall': avg_rouge_2['recall'],
                'f1': avg_rouge_2['f']
            },
            'rouge-l': {

                'precision': avg_rouge_l['precision'],
                'recall': avg_rouge_l['recall'],
                'f1': avg_rouge_l['f']
            }
        }

    def _tokenize(self, text: str):
        if self.tokenizer:
            return self.tokenizer.tokenize(text)
        return text.split()

    def _n_grams(self, sequence, n):
        return [tuple(sequence[i:i + n]) for i in range(len(sequence) - n + 1)]

    def _lcs_length(self, x, y):
        m, n = len(x), len(y)
        table = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    table[i][j] = table[i - 1][j - 1] + 1
                else:
                    table[i][j] = max(table[i - 1][j], table[i][j - 1])
        return table[m][n]

    def _compute_rouge_n(self, pred_tokens, ref_tokens, n):
        pred_ngrams = Counter(self._n_grams(pred_tokens, n))
        ref_ngrams = Counter(self._n_grams(ref_tokens, n))

        overlap_ngrams = pred_ngrams & ref_ngrams
        overlap_count = sum(overlap_ngrams.values())
        total_pred_count = sum(pred_ngrams.values())
        total_ref_count = sum(ref_ngrams.values())

        if total_pred_count == 0 or total_ref_count == 0:
            return {"precision": 0.0, "recall": 0.0, "f": 0.0}

        precision = overlap_count / total_pred_count if total_pred_count > 0 else 0.0
        recall = overlap_count / total_ref_count if total_ref_count > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

        return {"precision": precision, "recall": recall, "f": f1_score}

    def _compute_rouge_l(self, pred_tokens, ref_tokens):
        lcs = self._lcs_length(pred_tokens, ref_tokens)

        total_pred_count = len(pred_tokens)
        total_ref_count = len(ref_tokens)

        if total_pred_count == 0 or total_ref_count == 0:
            return {"precision": 0.0, "recall": 0.0, "f": 0.0}

        precision = lcs / total_pred_count if total_pred_count > 0 else 0.0
        recall = lcs / total_ref_count if total_ref_count > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

        return {"precision": precision, "recall": recall, "f": f1_score}

    def _average_metrics(self, results):
        if not results:
            return {"precision": 0.0, "recall": 0.0, "f": 0.0}

        precision = np.mean([r["precision"] for r in results])
        recall = np.mean([r["recall"] for r in results])
        f1 = np.mean([r["f"] for r in results])

        return {"precision": precision, "recall": recall, "f": f1}


class METEORScore:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def compute(self, predictions: list, references: list) -> dict:
        pred_tokens = [self._tokenize(p) for p in predictions]
        ref_tokens = [self._tokenize(r) for r in references]

        meteor_scores = [meteor_score([ref], pred) for pred, ref in zip(pred_tokens, ref_tokens)]

        if not meteor_scores:
            return {'meteor': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        meteor_mean = np.mean(meteor_scores)

        # Calculate precision, recall, and F1 score for METEOR
        precision = np.mean([self._precision(p, r) for p, r in zip(pred_tokens, ref_tokens)])
        recall = np.mean([self._recall(p, r) for p, r in zip(pred_tokens, ref_tokens)])
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {'meteor': meteor_mean, 'precision': precision, 'recall': recall, 'f1': f1}

    def _tokenize(self, text: str):
        if self.tokenizer:
            return self.tokenizer.tokenize(text)
        return text.split()

    def _precision(self, pred_tokens, ref_tokens):
        pred_counter = Counter(pred_tokens)
        ref_counter = Counter(ref_tokens)

        common_tokens = pred_counter & ref_counter
        num_common = sum(common_tokens.values())

        return num_common / len(pred_tokens) if pred_tokens else 0.0

    def _recall(self, pred_tokens, ref_tokens):
        pred_counter = Counter(pred_tokens)
        ref_counter = Counter(ref_tokens)

        common_tokens = pred_counter & ref_counter
        num_common = sum(common_tokens.values())

        return num_common / len(ref_tokens) if ref_tokens else 0.0

class BERTScore:
    def __init__(self, model_name="bert-base-uncased", device='cpu'):
        self.device = device
        self.model_name = model_name

    def compute(self, predictions: list, references: list) -> dict:
        P, R, F1 = bert_score.score(predictions, references, model_type=self.model_name, device=self.device)
        return {
            'precision': P.mean().item(),
            'recall': R.mean().item(),
            'f1': F1.mean().item()
        }

class Precision:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def compute(self, predictions: list, references: list) -> float:
        pred_tokens = [self._tokenize(p) for p in predictions]
        ref_tokens = [self._tokenize(r) for r in references]

        pred_counter = Counter([item for sublist in pred_tokens for item in sublist])
        ref_counter = Counter([item for sublist in ref_tokens for item in sublist])

        common_tokens = pred_counter & ref_counter
        num_common = sum(common_tokens.values())

        precision = num_common / len(pred_counter) if pred_counter else 0.0
        return precision

    def _tokenize(self, text: str):
        if self.tokenizer:
            return self.tokenizer.tokenize(text)
        return text.split()
# Define the BertEmbeddingMetric class
class BertEmbeddingMetric:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def clean_and_split(self, text):
        """Utility function to clean text and split into words"""
        if isinstance(text, str):
            # Convert to lowercase, remove punctuation, and split by space
            words = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
        else:
            words = []
        return words

    def compute(self, predictions: list, references: list) -> float:
        scores = []
        for pred, ref in zip(predictions, references):
            if not pred or not ref:
                scores.append(0.0)
                continue

            # Clean and split the answers
            ref_words = self.clean_and_split(ref)
            pred_words = self.clean_and_split(pred)

            # Convert pred_words to a set for fast lookup
            pred_set = set(pred_words)

            # Calculate exact match word overlap
            match_count = sum(word in pred_set for word in ref_words)
            match_percentage = match_count / len(ref_words) if ref_words else 0

            # Remove exact matching words from reference
            ref_words_remaining = [word for word in ref_words if word not in pred_set]

            # Compute BERT embedding similarity for remaining words
            if ref_words_remaining and pred_words:
                ref_text = ' '.join(ref_words_remaining)
                pred_text = ' '.join(pred_words)

                # Tokenize and encode with BERT
                inputs_ref = self.tokenizer(ref_text, return_tensors='pt')
                inputs_pred = self.tokenizer(pred_text, return_tensors='pt')

                with torch.no_grad():
                    outputs_ref = self.model(**inputs_ref)
                    outputs_pred = self.model(**inputs_pred)

                # Extract the embeddings (mean pooling across tokens)
                ref_embedding = outputs_ref.last_hidden_state.mean(dim=1)
                pred_embedding = outputs_pred.last_hidden_state.mean(dim=1)

                # Compute cosine similarity
                similarity = F.cosine_similarity(ref_embedding, pred_embedding).item()
            else:
                similarity = 0.0

            # Combine exact match percentage and BERT similarity
            combined_score = (match_percentage + similarity) / 2
            scores.append(combined_score)

        # Return the average score
        return np.mean(scores) if scores else 0.0


class Recall:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def compute(self, predictions: list, references: list) -> float:
        pred_tokens = [self._tokenize(p) for p in predictions]
        ref_tokens = [self._tokenize(r) for r in references]

        pred_counter = Counter([item for sublist in pred_tokens for item in sublist])
        ref_counter = Counter([item for sublist in ref_tokens for item in sublist])

        common_tokens = pred_counter & ref_counter
        num_common = sum(common_tokens.values())

        recall = num_common / len(ref_counter) if ref_counter else 0.0
        return recall

    def _tokenize(self, text: str):
        if self.tokenizer:
            return self.tokenizer.tokenize(text)
        return text.split()

class MetricEvaluator:
    def __init__(self, metrics):
        self.metrics = metrics

    def evaluate(self, predictions, references):
        results = {}
        for metric in self.metrics:
            if isinstance(predictions, list) and isinstance(references, list):
                # Ensure predictions and references are of the same length
                if len(predictions) != len(references):
                    print(predictions)
                    print(references)
                    raise ValueError("Predictions and references must have the same length")

                results[metric.__class__.__name__] = metric.compute(predictions, references)
            else:
                print(predictions)
                print(references)
                raise ValueError("Predictions and references should be lists.")
        return results

class WordMatchingMetric:
    def __init__(self):
        pass

    def convert_number_words(self, text):
        """Convert number words in the text to numeric values."""
        words = text.split()
        for i in range(len(words)):
            try:
                words[i] = str(w2n.word_to_num(words[i]))
            except ValueError:
                pass
        return ' '.join(words)

    def compute(self, predictions: list, references: list) -> float:
        scores = []
        for pred, ref in zip(predictions, references):
            # Convert number words to numeric values
            ref_text = self.convert_number_words(str(ref)).lower().split()
            pred_text = self.convert_number_words(str(pred)).lower().split()

            # Count the number of words in ref_text that are also in pred_text
            match_count = sum(word in pred_text for word in ref_text)

            # Calculate match percentage
            match_percentage = match_count / len(ref_text) if ref_text else 0
            scores.append(match_percentage)

        # Return the average score
        return np.mean(scores) if scores else 0.0

# Initialize Tokenizer and Model for Perplexity
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForCausalLM.from_pretrained("bert-base-uncased")

# Metrics
metrics = [
    ExactMatch(),
    BLEUScore(tokenizer),
    ROUGEScore(tokenizer),
    METEORScore(tokenizer),
    BERTScore(),
    Precision(tokenizer),
    Recall(tokenizer),
    SentenceTransformerSimilarity(),
    BertEmbeddingMetric(),
    WordMatchingMetric()
]
from transformers import pipeline
import json

# Define the model
model_name = "deepset/roberta-base-squad2"
device = 0 if torch.cuda.is_available() else -1
# Initialize the pipeline
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name, device=device)

# Define file paths
#input_file_path = 'NQ.json'
#input_file_name ='TQ.json'
metrics_with_predictions_file_path = 'metrics_with_predictions.json'
metrics_without_predictions_file_path = 'metrics_without_predictions.json'




evaluator = MetricEvaluator(metrics)

# Load the evaluation data
try:
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    print("JSON loaded successfully.")
except json.JSONDecodeError as e:
    print(f"Error loading JSON: {e}")
    data = []
except Exception as e:
    print(f"An error occurred: {e}")
    data = []

# Initialize results storage
results_with_predictions = []
results_without_predictions = []
#data = random.sample(data, 250)
# Loop over each entry in the JSON
for entry in data:
    question = entry["question"]
    golden_answers = entry["golden_answer"].split('/')  # Support multiple golden answers
    predictions = {
        "fid": entry.get("answer_fid"),
        "gpt35": entry.get("answer_gpt35"),
        "chatgpt": entry.get("answer_chatgpt"),
        "gpt4": entry.get("answer_gpt4"),
        "newbing": entry.get("answer_newbing")
    }

    judges = {
        "fid": entry.get("judge_fid"),
        "gpt35": entry.get("judge_gpt35"),
        "chatgpt": entry.get("judge_chatgpt"),
        "gpt4": entry.get("judge_gpt4"),
        "newbing": entry.get("judge_newbing")
    }

    # Calculate metrics using model predictions
    for model, context in predictions.items():
        if context:
            # Use the pipeline to get model's answer for the context
            model_QA_input = {
                'question': question,
                'context': context
            }
            model_res = nlp(model_QA_input)
            predicted_answer = model_res['answer']

            best_model_scores = {}
            for golden_answer in golden_answers:
              # Evaluate the current prediction-golden pair
              current_results = evaluator.evaluate([predicted_answer], [golden_answer])

              # Compare with best scores for this prediction
              for metric_name, scores in current_results.items():
                  if isinstance(scores, list):
                      # Handle case where scores is a list
                      if scores:  # Ensure the list is not empty
                          max_score = max(scores)
                          if metric_name not in best_model_scores:
                              best_model_scores[metric_name] = max_score
                          else:
                              best_model_scores[metric_name] = max(best_model_scores[metric_name], max_score)
                  elif isinstance(scores, dict):
                      # Handle case where scores is a dictionary
                      if metric_name not in best_model_scores:
                          best_model_scores[metric_name] = scores
                      else:
                          # Compare nested dictionaries
                          for sub_metric, sub_score in scores.items():
                              if sub_metric not in best_model_scores[metric_name]:
                                  best_model_scores[metric_name][sub_metric] = sub_score
                              else:
                                  if isinstance(sub_score, (int, float)) and isinstance(best_model_scores[metric_name][sub_metric], (int, float)):
                                      best_model_scores[metric_name][sub_metric] = max(
                                          best_model_scores[metric_name][sub_metric], sub_score
                                      )
                                  else:
                                      best_model_scores[metric_name][sub_metric] = sub_score
                  else:
                      # Handle scalar values
                      if metric_name not in best_model_scores:
                          best_model_scores[metric_name] = scores
                      else:
                          if isinstance(scores, (int, float)) and isinstance(best_model_scores[metric_name], (int, float)):
                              best_model_scores[metric_name] = max(best_model_scores[metric_name], scores)
                          else:
                              best_model_scores[metric_name] = scores


            results_with_predictions.append({
                "question": question,
                "model": model,
                "predicted_answer": predicted_answer,
                "gold_answer": golden_answers,
                "best_model_scores": best_model_scores,
                "judge": judges[model]
            })
            #print( results_with_predictions)

    # Calculate metrics without using model predictions
    for model, raw_answer in predictions.items():
        if raw_answer:
            best_model_scores = {}

            # Evaluate the raw answer against the golden answers
            current_results = evaluator.evaluate([raw_answer], [golden_answer])

            # Compare with best scores for this prediction
            for metric_name, scores in current_results.items():
                if isinstance(scores, list):
                    max_score = max(scores)
                    if metric_name not in best_model_scores:
                        best_model_scores[metric_name] = max_score
                    else:
                        best_model_scores[metric_name] = max(best_model_scores[metric_name], max_score)
                else:
                    if metric_name not in best_model_scores:
                        best_model_scores[metric_name] = scores
                    else:
                        best_model_scores[metric_name] = max(best_model_scores[metric_name], scores)

            results_without_predictions.append({
                "question": question,
                "model": model,
                "raw_answer": raw_answer,
                "gold_answer": golden_answers,
                "best_model_scores": best_model_scores,
                "judge": judges[model]
            })
            #print(results_without_predictions)

# Save the results to files
with open(metrics_with_predictions_file_path, 'w', encoding='utf-8') as metrics_file:
    json.dump(results_with_predictions, metrics_file, indent=4, ensure_ascii=False)

print(f"Evaluation results with model predictions saved to {metrics_with_predictions_file_path}")

with open(metrics_without_predictions_file_path, 'w', encoding='utf-8') as no_metrics_file:
    json.dump(results_without_predictions, no_metrics_file, indent=4, ensure_ascii=False)

print(f"Evaluation results with raw predictions saved to {metrics_without_predictions_file_path}")
