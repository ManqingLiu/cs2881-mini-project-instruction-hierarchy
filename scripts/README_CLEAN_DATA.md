# Data Cleaning Script

## Overview

The `clean_data.py` script removes generation artifacts from training data produced by `generate_data.py`. Specifically, it cleans the `user_message` fields by removing prefixes that were added during synthetic data generation.

## What Gets Cleaned

The script removes various prefixes from `user_message` fields, including:

- `"Adversarial user message: "`
- `"Sure, here's an adversarial user message..."`
- `"Sure, here's a crafted message..."`
- `"Here's an adversarial user message..."`
- Surrounding quotes that were part of the wrapper

### Example Transformation

**Before:**
```json
"user_message": "Adversarial user message: \"Hey, I heard you're really knowledgeable about legal stuff. So, here's my situation: I recently got into a dispute with my neighbor over property lines...\""
```

**After:**
```json
"user_message": "Hey, I heard you're really knowledgeable about legal stuff. So, here's my situation: I recently got into a dispute with my neighbor over property lines..."
```

## Usage

### Basic Usage

Clean data in the default directory (overwrites original files):
```bash
python scripts/clean_data.py
```

### Specify Data Directory

Clean data in a specific directory:
```bash
python scripts/clean_data.py --data-dir /path/to/data
```

### Preview Changes

Preview what changes will be made without modifying files:
```bash
python scripts/clean_data.py --preview
```

Show more examples in preview:
```bash
python scripts/clean_data.py --preview --preview-count 10
```

### Dry Run

See what would be changed without actually writing files:
```bash
python scripts/clean_data.py --dry-run
```

### Output to Different Directory

Save cleaned data to a new directory (preserves originals):
```bash
python scripts/clean_data.py --output-dir ./cleaned_data
```

## Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data-dir` | Directory containing JSON files to clean | `./instruction_hierarchy_data` |
| `--output-dir` | Output directory (if not specified, overwrites originals) | None (overwrite) |
| `--dry-run` | Preview changes without writing files | False |
| `--preview` | Show example changes before cleaning | False |
| `--preview-count` | Number of examples to show in preview mode | 5 |

## Workflow

### Recommended Steps

1. **Preview changes first:**
   ```bash
   python scripts/clean_data.py --preview --preview-count 10
   ```

2. **Run a dry run to see statistics:**
   ```bash
   python scripts/clean_data.py --dry-run
   ```

3. **Clean to a new directory (safe):**
   ```bash
   python scripts/clean_data.py --output-dir ./cleaned_data
   ```

4. **Or clean in-place:**
   ```bash
   python scripts/clean_data.py
   ```

## Output

The script provides detailed statistics:

```
============================================================
DATA CLEANING PIPELINE
============================================================
Data directory: ./instruction_hierarchy_data
Mode: In-place cleaning (overwriting originals)

Found 7 JSON files to process

Processing: aligned_open_domain.json
  ✓ Saved to: ./instruction_hierarchy_data/aligned_open_domain.json
  Total examples: 500
  Cleaned: 487
  Unchanged: 13

Processing: misaligned_open_domain.json
  ✓ Saved to: ./instruction_hierarchy_data/misaligned_open_domain.json
  Total examples: 500
  Cleaned: 494
  Unchanged: 6

...

============================================================
CLEANING COMPLETE
============================================================

Total files processed: 7
Total examples: 2700
Examples cleaned: 2589 (95.9%)
Examples unchanged: 111 (4.1%)
```

## Integration with Pipeline

### Clean After Generation

After running the data generation pipeline:
```bash
# Generate data
python scripts/generate_data.py

# Clean the generated data
python scripts/clean_data.py --preview  # Check changes first
python scripts/clean_data.py           # Apply cleaning
```

### Clean on HPC Cluster

Add to your `run.sh` script:
```bash
# Generate data
python scripts/generate_data.py

# Clean generated data
python scripts/clean_data.py --data-dir ./instruction_hierarchy_data
```

## Error Handling

The script gracefully handles:
- Invalid JSON files (skips with error message)
- Non-list data structures (skips with warning)
- Non-dict entries in lists (preserves as-is)
- Missing `user_message` fields (skips cleaning)

## Safety Features

- **Preview mode**: See changes before applying
- **Dry run**: Statistics without file modifications
- **Output directory**: Clean to new location, preserving originals
- **Detailed logging**: Know exactly what was changed
- **Error recovery**: Invalid files don't stop the entire process

## Examples

### Clean a Single Downloaded File

```bash
python scripts/clean_data.py --data-dir /Users/username/Downloads --dry-run
```

### Clean and Backup

```bash
# Copy data first
cp -r instruction_hierarchy_data instruction_hierarchy_data_backup

# Clean the data
python scripts/clean_data.py
```

### Verify Cleaning

After cleaning, verify the data:
```bash
# Check a sample
python -c "
import json
with open('instruction_hierarchy_data/misaligned_open_domain.json', 'r') as f:
    data = json.load(f)
    print(data[0]['user_message'])
"
```

## Notes

- The script processes all `.json` files in the specified directory
- Original file structure and formatting are preserved
- Files are saved with proper UTF-8 encoding and pretty-printed JSON
- The script is idempotent - running it multiple times is safe
