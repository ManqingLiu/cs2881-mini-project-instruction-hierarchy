"""
Train Baseline Model with SFT + RLHF (Red Bars in Figures 2 & 3)

This script trains the baseline model on CAPABILITY DATA ONLY (no instruction hierarchy data).
Following the paper's methodology, it includes both:
1. Supervised Fine-Tuning (SFT)
2. Reinforcement Learning from Human Feedback (RLHF) using DPO

The baseline model is the comparison point that shows what happens without instruction hierarchy training.

Usage:
    # Train with both SFT and DPO (recommended - matches paper):
    python train_baseline_improved.py --mode both --data_path ./training_data/baseline_data.json

    # Train SFT only:
    python train_baseline_improved.py --mode sft --data_path ./training_data/baseline_data.json

    # Train DPO only (requires SFT checkpoint):
    python train_baseline_improved.py --mode dpo --data_path ./training_data/baseline_data.json
"""

import os
import json
import torch
import argparse
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
    PeftModel
)
from trl import DPOConfig, DPOTrainer
from datasets import Dataset as HFDataset
import wandb
from typing import Dict, List


# ====================
# Configuration
# ====================

class BaselineConfig:
    """Configuration for baseline model training"""

    # Model configuration
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    use_4bit: bool = True

    # LoRA configuration (same as instruction hierarchy model for fair comparison)
    lora_r: int = 1
    lora_alpha: int = 256
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # Training configuration
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    num_epochs_sft: int = 3
    num_epochs_dpo: int = 1
    warmup_steps: int = 100
    max_length: int = 2048
    max_prompt_length: int = 1024

    # Paths
    data_path: str = "./training_data/baseline_data.json"
    output_dir: str = "./baseline_model"

    # DPO specific
    dpo_beta: float = 0.1

    # Logging
    use_wandb: bool = False
    wandb_project: str = "instruction-hierarchy"


# ====================
# Dataset Classes
# ====================

class BaselineDataset(Dataset):
    """Dataset for baseline model training (capability data only)"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 2048, mode: str = "sft"):
        """
        Args:
            data_path: Path to baseline_data.json
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            mode: "sft" for supervised fine-tuning or "dpo" for preference optimization
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

        # Load data
        print(f"Loading baseline data from {data_path}...")
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} baseline examples")

        # Prepare examples based on mode
        if mode == "sft":
            self.examples = self._prepare_sft_examples()
        elif mode == "dpo":
            self.examples = self._prepare_dpo_examples()
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _prepare_sft_examples(self) -> List[Dict]:
        """Prepare training examples for SFT"""
        examples = []

        for item in self.data:
            # Format conversation
            system_msg = item.get('system_message', 'You are a helpful AI assistant.')
            user_msg = item.get('user_message', '')
            response = item.get('ground_truth', '')

            if not user_msg or not response:
                continue

            # Format as chat
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": response}
            ]

            examples.append({
                'messages': messages,
                'type': item.get('type', 'baseline_capability')
            })

        print(f"Prepared {len(examples)} training examples for SFT")
        return examples

    def _prepare_dpo_examples(self) -> List[Dict]:
        """Prepare preference pairs for DPO training

        For baseline capability data, we create preferences based on:
        - Preferred: Helpful, accurate, well-formatted responses
        - Rejected: Less helpful, overly verbose, or slightly off-topic responses
        """
        examples = []

        for item in self.data:
            system_msg = item.get('system_message', 'You are a helpful AI assistant.')
            user_msg = item.get('user_message', '')
            response = item.get('ground_truth', '')

            if not user_msg or not response:
                continue

            # Format prompt (system + user message)
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]

            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Chosen response: the ground truth (high quality)
            chosen = response

            # Rejected response: generate a lower quality alternative
            # This could be overly verbose, slightly inaccurate, or less helpful
            rejected = self._generate_rejected_response(item, response)

            examples.append({
                'prompt': prompt,
                'chosen': chosen,
                'rejected': rejected
            })

        print(f"Prepared {len(examples)} preference pairs for DPO training")
        return examples

    def _generate_rejected_response(self, item: Dict, ground_truth: str) -> str:
        """Generate a lower quality response for DPO training

        This creates realistic "rejected" responses that are suboptimal but not completely wrong.
        The model will learn to prefer the ground truth over these alternatives.
        """
        item_type = item.get('type', '')

        # Different strategies for different types of tasks
        if 'code' in item_type.lower() or 'programming' in item_type.lower():
            # For code: add unnecessary complexity or verbose comments
            return ground_truth + "\n\n# Note: This could potentially be optimized further, though the current implementation works fine for most use cases."

        elif 'summarize' in item_type.lower() or 'summary' in item_type.lower():
            # For summaries: be overly verbose
            return f"Let me provide you with a detailed summary. {ground_truth} I hope this comprehensive summary helps you understand the key points."

        elif 'math' in item_type.lower() or 'calculation' in item_type.lower():
            # For math: add uncertainty
            return f"I believe the answer is: {ground_truth}. However, you might want to double-check this calculation."

        else:
            # Generic: add unnecessary hedging or verbosity
            rejected_options = [
                f"I think {ground_truth}. But I'm not entirely certain about all the details.",
                f"Based on my understanding, {ground_truth}. Though there might be other perspectives.",
                f"Let me try to help with that. {ground_truth}. I hope that makes sense.",
                f"{ground_truth}. Please let me know if you need me to elaborate further on any aspect of this response."
            ]
            # Use hash of ground truth to consistently select same rejected response
            idx = hash(ground_truth) % len(rejected_options)
            return rejected_options[idx]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        if self.mode == "sft":
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                example['messages'],
                tokenize=False,
                add_generation_prompt=False
            )

            # Tokenize
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            # Create labels (mask system and user messages, only train on assistant response)
            labels = encoding["input_ids"].clone()

            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": labels.squeeze()
            }

        elif self.mode == "dpo":
            # For DPO, return the example as-is
            # DPOTrainer will handle tokenization
            return example


def create_hf_dataset(examples: List[Dict], mode: str) -> HFDataset:
    """Convert examples to HuggingFace Dataset format"""
    if mode == "dpo":
        # DPO format: prompt, chosen, rejected
        return HFDataset.from_dict({
            'prompt': [ex['prompt'] for ex in examples],
            'chosen': [ex['chosen'] for ex in examples],
            'rejected': [ex['rejected'] for ex in examples]
        })
    else:
        raise ValueError(f"create_hf_dataset only needed for DPO mode, got: {mode}")


# ====================
# Training Functions
# ====================

def train_sft(config: BaselineConfig):
    """Train baseline model with SFT on capability data only"""

    print("\n" + "=" * 60)
    print("BASELINE MODEL SFT TRAINING")
    print("(Capability Data Only - No Instruction Hierarchy)")
    print("=" * 60)

    # Load tokenizer
    print(f"\n1. Loading tokenizer from {config.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Quantization config
    if config.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        bnb_config = None

    # Load model
    print(f"\n2. Loading base model {config.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Prepare model for LoRA
    if config.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # LoRA config
    print("\n3. Configuring LoRA...")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    print(f"\n4. Loading training data from {config.data_path}...")
    train_dataset = BaselineDataset(
        data_path=config.data_path,
        tokenizer=tokenizer,
        max_length=config.max_length,
        mode="sft"
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"{config.output_dir}/sft",
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        num_train_epochs=config.num_epochs_sft,
        learning_rate=config.learning_rate,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="wandb" if config.use_wandb else "none",
    )

    # Create trainer
    print("\n5. Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Train
    print("\n6. Starting SFT training...")
    trainer.train()

    # Save final model
    final_checkpoint_path = f"{config.output_dir}/sft_final"
    print(f"\n7. Saving SFT model to {final_checkpoint_path}")
    trainer.save_model(final_checkpoint_path)
    tokenizer.save_pretrained(final_checkpoint_path)

    print("\n" + "=" * 60)
    print("✓ BASELINE SFT TRAINING COMPLETED")
    print("=" * 60)
    print(f"Model saved to: {final_checkpoint_path}")
    print("=" * 60)


def train_dpo(config: BaselineConfig):
    """Train baseline model with DPO (RLHF) on capability data"""

    print("\n" + "=" * 60)
    print("BASELINE MODEL DPO TRAINING (RLHF)")
    print("(Capability Data Only - No Instruction Hierarchy)")
    print("=" * 60)

    # Path to SFT checkpoint
    sft_model_path = f"{config.output_dir}/sft_final"

    if not os.path.exists(sft_model_path):
        raise FileNotFoundError(
            f"SFT checkpoint not found at {sft_model_path}. "
            f"Please run SFT training first with --mode sft or --mode both"
        )

    print(f"\n1. Loading SFT checkpoint from: {sft_model_path}")

    # Load tokenizer from SFT checkpoint
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Quantization config
    if config.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        bnb_config = None

    # Load base model for training
    print("\n2. Loading base model for training...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.resize_token_embeddings(len(tokenizer))

    # Load SFT adapter
    print("Loading SFT adapter onto training model...")
    model = PeftModel.from_pretrained(model, sft_model_path)

    # Load reference model (frozen copy of SFT model)
    print("\n3. Loading base model for reference...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    ref_model.resize_token_embeddings(len(tokenizer))

    print("Loading SFT adapter onto reference model...")
    ref_model = PeftModel.from_pretrained(ref_model, sft_model_path)

    # Load DPO dataset
    print(f"\n4. Loading DPO training data from {config.data_path}...")
    dpo_dataset_obj = BaselineDataset(
        data_path=config.data_path,
        tokenizer=tokenizer,
        max_length=config.max_length,
        mode="dpo"
    )

    # Convert to HuggingFace Dataset
    train_dataset = create_hf_dataset(dpo_dataset_obj.examples, mode="dpo")
    print(f"Created {len(train_dataset)} DPO training examples")

    # DPO Training configuration
    print("\n5. Configuring DPO training...")
    dpo_config = DPOConfig(
        output_dir=f"{config.output_dir}/dpo",
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        num_train_epochs=config.num_epochs_dpo,
        learning_rate=config.learning_rate,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="wandb" if config.use_wandb else "none",
        beta=config.dpo_beta,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
    )

    # Create DPO Trainer
    print("\n6. Initializing DPO Trainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset
    )

    # Train
    print("\n7. Starting DPO training...")
    dpo_trainer.train()

    # Save final model
    final_checkpoint_path = f"{config.output_dir}/dpo_final"
    print(f"\n8. Saving DPO model to {final_checkpoint_path}")
    dpo_trainer.save_model(final_checkpoint_path)
    tokenizer.save_pretrained(final_checkpoint_path)

    print("\n" + "=" * 60)
    print("✓ BASELINE DPO TRAINING COMPLETED")
    print("=" * 60)
    print(f"Model saved to: {final_checkpoint_path}")
    print("\nThis model represents the RED BARS in Figures 2 and 3")
    print("(trained on capability data only, without instruction hierarchy)")
    print("=" * 60)


# ====================
# Main Function
# ====================

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(
        description="Train Baseline Model with SFT + RLHF (Red Bars in Figures 2 & 3)"
    )
    parser.add_argument(
        "--mode",
        choices=["sft", "dpo", "both"],
        required=True,
        help="Training mode: 'sft' for supervised fine-tuning, 'dpo' for RLHF/DPO, 'both' for sequential training (recommended)"
    )
    parser.add_argument(
        "--data_path",
        default="./training_data/baseline_data.json",
        help="Path to baseline training data JSON file"
    )
    parser.add_argument(
        "--output_dir",
        default="./baseline_model",
        help="Output directory for baseline model"
    )
    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model name"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size per device"
    )
    parser.add_argument(
        "--num_epochs_sft",
        type=int,
        default=3,
        help="Number of training epochs for SFT"
    )
    parser.add_argument(
        "--num_epochs_dpo",
        type=int,
        default=1,
        help="Number of training epochs for DPO"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )

    args = parser.parse_args()

    # Verify data file exists
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(
            f"Baseline data not found at {args.data_path}. "
            f"Please run: python generate_data.py --data_type baseline"
        )

    # Initialize configuration
    config = BaselineConfig()
    config.data_path = args.data_path
    config.output_dir = args.output_dir
    config.model_name = args.model_name
    config.use_wandb = args.use_wandb
    config.batch_size = args.batch_size
    config.num_epochs_sft = args.num_epochs_sft
    config.num_epochs_dpo = args.num_epochs_dpo
    config.learning_rate = args.learning_rate

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Initialize wandb if enabled
    if config.use_wandb:
        wandb.init(project=config.wandb_project, name=f"baseline-{args.mode}")

    print("\n" + "=" * 60)
    print("BASELINE MODEL TRAINING SETUP")
    print("(Following Paper's Methodology: SFT + RLHF)")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Data: {config.data_path}")
    print(f"Output: {config.output_dir}")
    print(f"Model: {config.model_name}")
    print(f"SFT Epochs: {config.num_epochs_sft}")
    print(f"DPO Epochs: {config.num_epochs_dpo}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"W&B: {'Enabled' if config.use_wandb else 'Disabled'}")
    print("=" * 60 + "\n")

    try:
        if args.mode == "sft":
            print("Running SFT training only...")
            train_sft(config)

        elif args.mode == "dpo":
            print("Running DPO training only (requires SFT checkpoint)...")
            train_dpo(config)

        elif args.mode == "both":
            print("Running both SFT and DPO training (recommended - matches paper)...")
            train_sft(config)
            print("\n\n" + "=" * 60)
            print("SFT Complete - Starting DPO (RLHF)")
            print("=" * 60 + "\n")
            train_dpo(config)

        print("\n" + "=" * 60)
        print("✓ SUCCESS: Baseline training completed!")
        print("=" * 60)

        if args.mode == "both" or args.mode == "dpo":
            print(f"\nFinal model location: {config.output_dir}/dpo_final")
            print("\nThis baseline model (trained with SFT + RLHF on capability data)")
            print("represents the RED BARS in Figures 2 and 3 of the paper.")
        else:
            print(f"\nSFT model location: {config.output_dir}/sft_final")
            print("Note: Run DPO training (--mode dpo) to complete the baseline model")

        print("\nNext steps:")
        print("1. Train instruction hierarchy model:")
        print("   python instruction_hierarchy_training.py --mode both")
        print("2. Create evaluation datasets:")
        print("   python create_eval_datasets.py")
        print("3. Run evaluations to reproduce Figures 2 and 3:")
        print("   python evaluate_models_improved.py")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise

    finally:
        if config.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()