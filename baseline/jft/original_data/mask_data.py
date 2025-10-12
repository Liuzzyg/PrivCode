import argparse
import json
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
import tqdm

class PIIMasker:
    def __init__(self, cache_dir):
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the NER model for PII detection
        self.tokenizer = AutoTokenizer.from_pretrained(
            "bigcode/starpii",
            cache_dir=cache_dir
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            "bigcode/starpii",
            cache_dir=cache_dir
        ).to(self.device)

        self.ner_pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=self.device
        )

    def mask_pii(self, text):
        """Detect PII in text and replace with <mask>."""
        try:
            # Detect PII entities
            entities = self.ner_pipeline(text)
            
            # Sort entities by start index in descending order to avoid index shifting
            entities = sorted(entities, key=lambda x: x['start'], reverse=True)
            
            # Replace PII with <mask>
            masked_text = text
            for entity in entities:
                start, end = entity['start'], entity['end']
                masked_text = masked_text[:start] + '<mask>' + masked_text[end:]
            
            return masked_text, len(entities)
        except Exception as e:
            print(f"Error processing text: {e}")
            return text, 0

    def process_dataset(self, dataset, output_path):
        """Process dataset, mask PII in solutions, and save results."""
        total_entities = 0
        processed_data = []

        print("Processing dataset for PII detection and masking...")
        for sample in tqdm.tqdm(dataset, desc="Processing samples"):
            # Mask PII in solution
            masked_solution, entity_count = self.mask_pii(sample['solution'])
            total_entities += entity_count
            
            # Create new sample with masked solution
            processed_sample = {
                'problem': sample['problem'],
                'solution': masked_solution
            }
            processed_data.append(processed_sample)

        print(f"Total PII entities detected and masked: {total_entities}")

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Save processed dataset
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)

        return total_entities

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mask PII in Magicoder dataset solutions.")
    parser.add_argument("--cache_dir", type=str, default=".../.cache/huggingface", 
                       help="Cache directory for models and datasets.")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Seed for reproducibility.")
    parser.add_argument("--original_dataset", type=str, 
                       default=".../dpcode/canary/origin_data/pii_instruction_dataset_canary_rep100.jsonl",
                       help="Path to save the masked dataset.")
    parser.add_argument("--output_path", type=str, 
                       default="baseline/jft/original_data/masked_dataset_canary_rep100.json",
                       help="Path to save the masked dataset.")

    args = parser.parse_args()

    # Load dataset
    if args.original_dataset == 'ise-uiuc/Magicoder-OSS-Instruct-75K':
        dataset = load_dataset(
            args.original_dataset,
            split="train",
            # use_auth_token=True,
            cache_dir=os.path.join(args.cache_dir, 'datasets')
        )
    elif 'canary' in args.original_dataset:
        dataset = load_dataset(
                "json", 
                data_files=args.original_dataset,
                split="train",
                cache_dir='.../.cache/huggingface/datasets'
            )

    # Split dataset (keeping most as train)
    dataset = dataset.train_test_split(test_size=0.00001, seed=args.seed)
    train_data = dataset['train']

    # Initialize PII masker
    masker = PIIMasker(cache_dir=os.path.join(args.cache_dir, 'hub'))

    # Process dataset and save results
    total_entities = masker.process_dataset(
        dataset=train_data,
        output_path=args.output_path
    )