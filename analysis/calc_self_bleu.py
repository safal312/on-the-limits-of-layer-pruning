import json
import argparse
import sys
import numpy as np
import nltk

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Ensure tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading nltk punkt tokenizer...", file=sys.stderr)
    nltk.download('punkt', quiet=True)

def get_responses_from_data(data):
    """
    Extracts responses from various JSON structures.
    Returns a dictionary: { "dataset_name": [list of responses], ... }
    """
    all_responses = {}
    
    # Case 1: Evaluation script format (Flat with 'items' or 'dataset_type')
    if 'items' in data and isinstance(data['items'], list):
        dataset_name = data.get('dataset_type', 'unknown_dataset')
        responses = [item.get('response', '') for item in data['items'] if 'response' in item]
        all_responses[dataset_name] = responses
        
    # Case 2: Pruning script format (Nested under 'datasets')
    elif 'datasets' in data and isinstance(data['datasets'], dict):
        for ds_name, ds_data in data['datasets'].items():
            if 'items' in ds_data:
                responses = [item.get('response', '') for item in ds_data['items'] if 'response' in item]
                all_responses[ds_name] = responses
                
    # Case 3: Direct list of items
    elif isinstance(data, list):
        responses = [item.get('response', '') for item in data if isinstance(item, dict) and 'response' in item]
        if responses:
            all_responses['list_data'] = responses

    return all_responses

def calculate_intra_self_bleu(responses, ngram=4):
    """
    Calculates Intra-Self-BLEU score (commonly used in dialog/story generation).
    For each response, measures similarity between its own sentences.
    Higher score = lower diversity (more redundancy within the response).
    """
    if not responses:
        return 0.0

    scores = []
    smoothing = SmoothingFunction().method1
    
    # Weights for BLEU-N
    if ngram == 2:
        weights = (0.5, 0.5)
    elif ngram == 3:
        weights = (0.33, 0.33, 0.33)
    elif ngram == 4:
        weights = (0.25, 0.25, 0.25, 0.25)
    else:
        weights = tuple([1.0/ngram] * ngram)

    for text in responses:
        # Split into sentences (simple rough split by newline or punctuation could also work, 
        # but punkt sentence tokenizer is standard)
        try:
            sentences = nltk.sent_tokenize(text.lower())
        except:
            continue
            
        if len(sentences) < 2:
            continue
            
        # Tokenize each sentence
        tokenized_sents = [nltk.word_tokenize(s) for s in sentences]
        
        # Calculate BLEU for each sentence against other sentences in the same response
        doc_scores = []
        for i in range(len(tokenized_sents)):
            hypothesis = tokenized_sents[i]
            references = tokenized_sents[:i] + tokenized_sents[i+1:]
            
            # Remove empty/too short references
            references = [ref for ref in references if len(ref) >= ngram]
            
            if not hypothesis or len(hypothesis) < ngram or not references:
                continue
                
            try:
                score = sentence_bleu(references, hypothesis, weights=weights, smoothing_function=smoothing)
                doc_scores.append(score)
            except:
                pass
        
        if doc_scores:
            scores.append(np.mean(doc_scores))

    return np.mean(scores) if scores else 0.0

def calculate_repetition_rate(responses, n=4):
    """
    Calculates Repetition Rate (Rep-N) to measure within-sequence degeneration.
    Rep-N = 100 * (1.0 - unique_ngrams / total_ngrams)
    
    Higher Score = More Repetitive (Degenerated)
    Lower Score = Less Repetitive (Natural)
    """
    if not responses:
        return 0.0
    
    rep_rates = []
    
    for text in responses:
        tokens = nltk.word_tokenize(text.lower())
        if len(tokens) < n:
            continue
            
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        total_ngrams = len(ngrams)
        unique_ngrams = len(set(ngrams))
        
        if total_ngrams > 0:
            # Repetition rate for this single response
            rep_rate = 1.0 - (unique_ngrams / total_ngrams)
            rep_rates.append(rep_rate)
            
    # Return average percentage
    return np.mean(rep_rates) * 100 if rep_rates else 0.0

def calculate_avg_tokens(responses):
    """
    Calculates average number of tokens per response.
    """
    if not responses:
        return 0.0
    
    total_tokens = 0
    for r in responses:
        total_tokens += len(nltk.word_tokenize(r))
        
    return total_tokens / len(responses)

def main():
    parser = argparse.ArgumentParser(description="Calculate Repetition Rate and Intra-Self-BLEU to measure degeneration.")
    parser.add_argument("file_path", type=str, help="Path to the JSON file containing responses")
    args = parser.parse_args()

    print(f"Reading file: {args.file_path}")
    try:
        with open(args.file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    datasets_responses = get_responses_from_data(data)
    
    if not datasets_responses:
        print("No 'response' fields found in the file structure.")
        return

    print(f"{'='*110}")
    print(f"{'Dataset':<20} | {'Samples':<8} | {'Avg Tokens':<10} | {'Rep-2':<8} | {'Rep-3':<8} | {'Rep-4':<8} | {'Intra-BLEU-4':<12}")
    print(f"{'='*110}")

    for name, responses in datasets_responses.items():
        count = len(responses)
        if count == 0:
            print(f"{name:<20} | {'0':<8} | {'N/A':<10} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'N/A':<12}")
            continue
            
        rep2 = calculate_repetition_rate(responses, 2)
        rep3 = calculate_repetition_rate(responses, 3)
        rep4 = calculate_repetition_rate(responses, 4)
        intra_bleu4 = calculate_intra_self_bleu(responses, ngram=4)
        avg_tokens = calculate_avg_tokens(responses)
        
        print(f"{name:<20} | {count:<8} | {avg_tokens:<10.1f} | {rep2:5.2f}%   | {rep3:5.2f}%   | {rep4:5.2f}%   | {intra_bleu4:.4f}")
    print(f"{'='*110}")
    print("Note: Higher Rep/Intra-BLEU scores indicate more degeneration (repetition).")

if __name__ == "__main__":
    main()
