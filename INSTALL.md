# TravelPlanner Installation Guide

This guide explains how to install TravelPlanner as a Python package with CLI scripts.

## Prerequisites

1. **Download the database**: Download the [database](https://drive.google.com/file/d/1pF1Sw6pBmq2sFkJvm-LzJOqrmfWoQgxE/view?usp=drive_link) and unzip it to the `TravelPlanner` directory. The database folder should contain CSV files for flights, accommodations, restaurants, attractions, etc.

2. **Python 3.9+**: Ensure you have Python 3.9 or higher installed.

## Installation

### Using uv (Recommended)

```bash
# Install TravelPlanner in editable mode
uv pip install -e .
```

### Using pip

```bash
# Install TravelPlanner in editable mode
pip install -e .
```

## Available Scripts

After installation, the following CLI scripts will be available:

### Evaluation

#### `tp-eval` - Evaluate Generated Plans

Evaluate TravelPlanner generated plans against constraints.

**Usage:**
```bash
tp-eval --set_type validation --evaluation_file_path /path/to/submission.jsonl
```

**Arguments:**
- `--set_type`: Dataset split to evaluate (choices: `train`, `validation`, `test`)
- `--evaluation_file_path`: Path to the evaluation file in JSONL format (required)

**Example:**
```bash
# Evaluate a validation set submission
tp-eval --set_type validation --evaluation_file_path ./results/submission_validation.jsonl

# Evaluate a test set submission
tp-eval --set_type test --evaluation_file_path ./results/submission_test.jsonl
```

### Postprocessing Pipeline

The postprocessing pipeline converts natural language plans to structured JSON format for evaluation. Run these scripts in sequence:

#### `tp-parse` - Parse Natural Language Plans to JSON

Use GPT-4 to convert natural language travel plans to structured JSON format.

**Usage:**
```bash
tp-parse --set_type validation --model_name gpt-3.5-turbo-1106 --mode two-stage --output_dir ./output --tmp_dir ./tmp
```

**Arguments:**
- `--set_type`: Dataset split (default: `validation`)
- `--model_name`: Model name used to generate plans (default: `gpt-3.5-turbo-1106`)
- `--mode`: Planning mode: `two-stage` or `sole-planning` (default: `two-stage`)
- `--strategy`: Strategy for sole-planning mode (default: `direct`)
- `--output_dir`: Directory containing generated plan JSON files (default: `./`)
- `--tmp_dir`: Directory to save parsed output (default: `./`)

**Note:** Requires `OPENAI_API_KEY` environment variable.

#### `tp-extract` - Extract Elements from Parsed Plans

Extract and validate structured elements from parsed plans.

**Usage:**
```bash
tp-extract --set_type validation --model_name gpt-3.5-turbo-1106 --mode two-stage --output_dir ./output --tmp_dir ./tmp
```

**Arguments:** Same as `tp-parse`.

#### `tp-combine` - Combine Plans into Submission File

Combine parsed plans with queries into a JSONL submission file for evaluation.

**Usage:**
```bash
tp-combine --set_type validation --model_name gpt-3.5-turbo-1106 --mode two-stage --output_dir ./output --submission_file_dir ./submissions
```

**Arguments:**
- `--set_type`: Dataset split (default: `validation`)
- `--model_name`: Model name (default: `gpt-3.5-turbo-1106`)
- `--mode`: Planning mode (default: `two-stage`)
- `--strategy`: Strategy for sole-planning (default: `direct`)
- `--output_dir`: Directory with generated plan JSONs (default: `./`)
- `--submission_file_dir`: Directory to save submission JSONL (default: `./`)

#### `tp-format-check` - Validate Plan Formats

Check format validity of generated travel plans.

**Usage:**
```bash
tp-format-check --evaluation_file_path /path/to/submission.jsonl
```

**Arguments:**
- `--evaluation_file_path`: Path to JSONL file containing plans to check (required)

### Complete Postprocessing Example

```bash
# Set up environment
export OPENAI_API_KEY=your_api_key

# 1. Parse natural language plans using GPT-4
tp-parse --set_type validation --model_name gpt-3.5-turbo-1106 --mode two-stage \
  --output_dir ./output --tmp_dir ./tmp

# 2. Extract structured elements
tp-extract --set_type validation --model_name gpt-3.5-turbo-1106 --mode two-stage \
  --output_dir ./output --tmp_dir ./tmp

# 3. Combine into submission file
tp-combine --set_type validation --model_name gpt-3.5-turbo-1106 --mode two-stage \
  --output_dir ./output --submission_file_dir ./submissions

# 4. Optional: Check format validity
tp-format-check --evaluation_file_path ./submissions/validation_gpt-3.5-turbo-1106_two-stage_submission.jsonl

# 5. Evaluate the submission
tp-eval --set_type validation --evaluation_file_path ./submissions/validation_gpt-3.5-turbo-1106_two-stage_submission.jsonl
```

## Package Structure

After installation, TravelPlanner modules can be imported directly:

```python
from evaluation.eval import eval_score
from agents.tool_agents import ReactAgent
from tools.flights.apis import Flights
```

## Troubleshooting

### Database Files Missing

If you see errors like `FileNotFoundError: '../database/flights/clean_Flights_2022.csv'`, ensure you have:
1. Downloaded the database from the link above
2. Extracted it to the TravelPlanner root directory
3. The `database/` folder contains subdirectories like `flights/`, `accommodations/`, `restaurants/`, etc.

### Dependency Conflicts

If you encounter dependency issues:

```bash
# Reinstall with exact dependencies
uv pip install -e . --force-reinstall

# Or manually fix specific packages
uv pip install 'huggingface-hub<1.0.0'
```

## Development

To install in development mode with additional tools:

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"
```

This includes:
- pytest for testing
- black for code formatting
- ruff for linting

## Environment Variables

Some scripts require environment variables:

```bash
export OPENAI_API_KEY=your_openai_key
export GOOGLE_API_KEY=your_google_key  # Optional, use "1" if not using Google models
```

## Next Steps

- See [README.md](README.md) for usage examples
- Check the [evaluation/](evaluation/) directory for evaluation modules
- Check the [agents/](agents/) directory for agent implementations
