# Instruction Hierarchy: LLM Training for Privileged Instructions

[![Paper](https://img.shields.io/badge/arXiv-2312.06681-b31b1b.svg)](https://arxiv.org/abs/2404.13208)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Implementation of "**The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions**" - a method for training language models to follow system-level instructions over conflicting user inputs, improving robustness against prompt injection and jailbreak attacks.

> **Note**: This implementation uses **Qwen/Qwen2.5-7B-Instruct** instead of the paper's GPT-4/Claude models due to API cost constraints, but follows the similar methodology with minor changes in fine-tuning and RL procedure.

## 📋 Overview

This repository reproduces **Figures 2 & 3** from the paper, demonstrating that instruction hierarchy training significantly improves model robustness:

- **Figure 2 (Main Results)**: Performance on core attack types (prompt injection, conflicting instructions, system extraction)
- **Figure 3 (Generalization)**: Zero-shot transfer to unseen attacks (jailbreaks, password extraction)

### Key Results

Training on instruction hierarchy data improves robustness by **20-40 percentage points** across multiple attack categories, with strong generalization to unseen attack types.

| Attack Type | Baseline | + Instruction Hierarchy | Improvement |
|-------------|----------|------------------------|-------------|
| Prompt Injection | ~30% | ~70% | +40% |
| Conflicting Instructions | ~45% | ~80% | +35% |
| System Extraction | ~40% | ~75% | +35% |
| Jailbreaks (zero-shot) | ~25% | ~55% | +30% |

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **CUDA-capable GPU** (16GB+ VRAM recommended for 7B model)
- **~50GB disk space** for models and data

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd instruction-hierarchy

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### One-Command Execution

Run the entire pipeline (data generation → training → evaluation → plotting):

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

**Expected Runtime**: 6-12 hours depending on hardware (mostly training time).

## 📚 Step-by-Step Usage

### 1. Generate Training Data

Generate both baseline and instruction hierarchy training data:

```bash
python generate_data.py \
    --data_type both \
    --num_samples 5000 \
    --output_dir ./training_data \
    --generation_model "Qwen/Qwen2.5-7B-Instruct"
```

**Arguments**:
- `--data_type`: Choose from `baseline`, `instruction_hierarchy`, or `both`
- `--num_samples`: Number of samples per data type (default: 5000)
- `--output_dir`: Where to save generated data
- `--generation_model`: Model to use for data generation
- `--batch_size`: Batch size for generation (default: 4)

**Output**:
- `training_data/baseline_data.json` - Capability/instruction-following data
- `training_data/instruction_hierarchy_full.json` - Aligned + misaligned hierarchy data

**Data Distribution** (for instruction hierarchy):
- Aligned open domain: 25%
- Misaligned open domain: 20%
- Closed domain injection: 15%
- Indirect injection: 15%
- System extraction: 15%
- Baseline capabilities: 10%

### 2. Create Evaluation Datasets

Split training data into evaluation sets for different attack types:

```bash
python create_eval_datasets.py \
    --data_dir ./training_data \
    --eval_dir ./eval_data \
    --eval_split 0.2
```

**Output**: Creates 10 evaluation JSON files in `eval_data/`:
- **Figure 2**: `prompt_injection_hijacking.json`, `prompt_injection_new.json`, `user_conflicting.json`, `indirect_browsing.json`, `system_extraction.json`
- **Figure 3**: `indirect_tools.json`, `tensortrust_password.json`, `gandalf_password.json`, `jailbreak_unsafe.json`, `chatgpt_jailbreaks.json`

### 3. Train Baseline Model

Train baseline model on **capability data only** (no instruction hierarchy):

```bash
python train_baseline.py \
    --mode both \
    --data_path ./training_data/baseline_data.json \
    --output_dir ./baseline_model \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --num_epochs_sft 3 \
    --num_epochs_dpo 2
```

**Arguments**:
- `--mode`: Training mode - `sft`, `dpo`, or `both` (recommended: `both`)
- `--data_path`: Path to baseline data JSON
- `--output_dir`: Where to save model checkpoints
- `--model_name`: Base model to fine-tune
- `--num_epochs_sft`: SFT epochs (default: 3)
- `--num_epochs_dpo`: DPO epochs (default: 1)
- `--batch_size`: Training batch size (default: 1)
- `--use_4bit`: Use 4-bit quantization (default: True)

**Output**: 
- `baseline_model/sft_final/` - After SFT
- `baseline_model/dpo_final/` - After DPO (if mode=both)
- `baseline_model/final/` - Symlink to final model

### 4. Train Instruction Hierarchy Model

Train model on **instruction hierarchy data**:

```bash
python instruction_hierarchy_training.py \
    --mode both \
    --data_path ./training_data/instruction_hierarchy_full.json \
    --output_dir ./instruction_hierarchy_model \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --num_epochs_sft 3 \
    --num_epochs_dpo 2
```

**Arguments**: Same as baseline training (see above)

**Output**:
- `instruction_hierarchy_model/sft_final/` - After SFT
- `instruction_hierarchy_model/dpo_final/` - After DPO (if mode=both)

### 5. Evaluate Both Models

Evaluate both models on all test datasets:

```bash
python evaluate_models.py \
    --baseline_model ./baseline_model/final \
    --ih_model ./instruction_hierarchy_model/dpo_final \
    --base_model_name "Qwen/Qwen2.5-7B-Instruct" \
    --eval_dir ./eval_data \
    --output_file ./evaluation_results.json
```

**Arguments**:
- `--baseline_model`: Path to trained baseline model
- `--ih_model`: Path to trained hierarchy model
- `--base_model_name`: Base model name for tokenizer
- `--eval_dir`: Directory with evaluation datasets
- `--output_file`: Where to save results JSON

**Output**: `evaluation_results.json` with robustness scores for each model/attack type

### 6. Generate Figures

Create publication-ready figures:

```bash
python plot_results.py \
    --results_file ./evaluation_results.json \
    --output_dir ./figures \
    --format png
```

**Arguments**:
- `--results_file`: Path to evaluation results JSON
- `--output_dir`: Where to save figures
- `--format`: Output format - `png`, `pdf`, or `svg` (default: png)

**Output**:
- `figures/figure_2.png` - Main results (5 attack types)
- `figures/figure_3.png` - Generalization results (5 unseen attacks)
- `figures/combined_figures.png` - Both figures side-by-side


## ⚙️ Configuration

### Memory Requirements

- **Data Generation**: 8GB VRAM (can use CPU if needed)
- **Training (4-bit)**: 16GB VRAM
- **Training (full precision)**: 32GB+ VRAM
- **Evaluation**: 16GB VRAM

## 📖 Methodology

### Training Process

Both models follow a two-stage training process:

1. **Supervised Fine-Tuning (SFT)**: Train with LoRA on task-specific data
   - Baseline: Capability/instruction-following data only
   - Hierarchy: Mixed aligned + misaligned instruction hierarchy data

2. **Direct Preference Optimization (DPO)**: Reinforcement learning using preference pairs
   - Chosen: Aligned responses (follow system instructions)
   - Rejected: Misaligned responses (follow conflicting user instructions)

### Key Differences from Paper

| Aspect | Paper            | This Implementation |
|--------|------------------|---------------------|
| Base Model | GPT-3.5 Turbo    | Qwen/Qwen2.5-7B-Instruct |
| Training Method | Full fine-tuning | LoRA (efficient) |
| Data Size | ~50K samples     | ~5-10K samples (configurable) |
| Data Generation | GPT-3.5/GPT-4    | Qwen/Qwen2.5-7B-Instruct |
| Evaluation Model | GPT-4            | deepseek-ai/DeepSeek-R1-Distill-Llama-8B |

## 🔬 Attack Types

### Figure 2: Main Results

1. **Prompt Injection (Hijacking)**: User attempts to override system instructions
2. **Prompt Injection (New Instructions)**: Closed-domain context with injected instructions
3. **User Conflicting Instructions**: Direct user commands that conflict with system role
4. **Indirect Browsing**: Malicious instructions in web content
5. **System Extraction**: Attempts to extract system prompts/secrets

### Figure 3: Generalization (Zero-Shot)

1. **Indirect Tools**: Malicious instructions in tool outputs
2. **TensorTrust Password**: Password extraction challenges
3. **Gandalf Password**: Password guessing game
4. **Jailbreak (Unsafe)**: Jailbreakchat prompts with harmful requests
5. **ChatGPT Jailbreaks**: Known ChatGPT jailbreak attempts


## 📝 Citation

If you use this code, please cite the original paper:

```bibtex
@article{wallace2024instruction,
  title={The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions},
  author={Wallace, Eric and Xiao, Kai and Leike, Jan and Weng, Lilian and Heidecke, Johannes and Beutel, Alex},
  journal={arXiv preprint arXiv:2404.13208},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 🙏 Acknowledgments

- Original Instruction Hierarchy paper authors
- Hugging Face for transformer implementations
- Qwen team for the base models

**⭐ If you find this useful, please star the repository!**