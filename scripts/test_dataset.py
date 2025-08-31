from nnsight import LanguageModel
import argparse
import json
from transformers import AutoTokenizer

def test_token_indices(dataset, model_name):

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)

    for item in dataset:
        toks = tokenizer.encode(item['prompt_source'])
        tok_str = [tokenizer.decode([t]) for t in toks]

        for name, idx in sorted(item['token_indices'].items()):
            # A value of -1 is expected for nouns not present in the analogy cue
            if idx != -1 and idx < len(tok_str):
                print(f"{name:<15}: index={idx:<4} token='{tok_str[idx]}")
            else:
                print(f"{name:<15}: NOT FOUND (index={idx})")

def test_baseline_accuracy(dataset, model_name, batch_size=32):
    """
    Args:
        dataset: List of dataset items
        model_name: Name of the model to load
        batch_size: Number of items to process in each batch
    """
    model = LanguageModel(model_name)
    tokenizer = model.tokenizer
    
    correct = 0
    total = 0

    # Process dataset in batches
    for batch_start in range(0, len(dataset), batch_size):
        batch_end = min(batch_start + batch_size, len(dataset))
        batch_items = dataset[batch_start:batch_end]
        
        # Extract prompts for the batch
        batch_prompts = [item['prompt_source'] for item in batch_items]
        
        # Process the entire batch at once
        with model.trace(batch_prompts):
            batch_token_ids = model.lm_head.output.argmax(dim=-1).save()
        
        # Process results for each item in the batch
        for i, item in enumerate(batch_items):
            # Get the logits at the last position and find the most likely token
            source_pred = tokenizer.decode(batch_token_ids[i][-1])
            
            is_correct = source_pred.strip() in item['answer']

            if is_correct:
                correct += 1
            total += 1

            print(f"Item {total}: Predicted: '{source_pred}' | Ground Truth: '{item['answer']}' | Correct: {is_correct}")

        accuracy = correct / total if total > 0 else 0
        print(f"\nAccuracy: {correct}/{total} = {accuracy:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='../data/patching_dataset_mixed.jsonl')
    parser.add_argument('--model_name', default='meta-llama/Llama-3.3-70B-Instruct')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for processing multiple prompts at once')
    args = parser.parse_args()

    # Load the dataset
    dataset = []
    with open(args.dataset, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    
    # Run the accuracy method
    test_baseline_accuracy(dataset, args.model_name, args.batch_size)
    