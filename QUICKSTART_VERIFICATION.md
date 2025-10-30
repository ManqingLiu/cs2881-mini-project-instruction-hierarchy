# Quick Start: Using Verification and Rejection Sampling

## What's New?

Ground truth generation now includes **AI verification** and **rejection sampling** to ensure responses properly resist prompt injections.

## How to Use

### Option 1: Default (Recommended)

The verification is **enabled by default**. Just run normally:

```bash
python scripts/generate_data.py
```

The pipeline will automatically:
- ✓ Generate responses with multiple attempts
- ✓ Verify each response resists injections
- ✓ Use fallbacks when needed
- ✓ Display quality statistics

### Option 2: Testing on Small Dataset

Test with just a few examples:

```bash
python scripts/test_verification.py
```

This will:
- Test 3 hand-crafted examples
- Show verification process in detail
- Generate 5 full examples
- Display sample output

### Option 3: Quick Structure Check (No Model Loading)

Verify the code structure without loading models:

```bash
python3 scripts/test_verification_simple.py
```

## Configuration

### Adjust Max Attempts

Edit [scripts/generate_data.py:657](scripts/generate_data.py#L657):

```python
ground_truth, attempts = self.generate_ground_truth_with_rejection_sampling(
    task=task,
    original_text=sample_text,
    injection=injection,
    max_attempts=3,           # Change this (default: 3)
    use_ai_verification=True
)
```

- `max_attempts=1`: Fastest, no retry (not recommended)
- `max_attempts=2`: Balanced (faster, slightly lower quality)
- `max_attempts=3`: Default (recommended)
- `max_attempts=5`: Slowest (highest quality)

### Disable Verification (For Speed Testing)

```python
ground_truth, attempts = self.generate_ground_truth_with_rejection_sampling(
    task=task,
    original_text=sample_text,
    injection=injection,
    max_attempts=1,
    use_ai_verification=False  # Disable verification
)
```

**Warning:** This may produce lower quality data.

## Understanding the Output

### During Generation

You'll see progress like this:

```
Step 4: Generating ground truth responses with rejection sampling...
This will verify each response and retry if it follows the injection.

Generating verified ground truths: 100%|████████| 500/500 [25:00<00:00,  3.00s/it]

  Attempt 1 failed: Appears to follow injection. Retrying...
  Attempt 2 failed: Does not perform task. Retrying...
  All 3 attempts failed. Using fallback response.
```

### Statistics Summary

After generation, you'll see:

```
✓ Generated 500 verified ground truth responses
  Statistics:
    - Succeeded on first attempt: 350  (70%)
    - Succeeded after retry: 120       (24%)
    - Used fallback: 30                (6%)
```

**What this means:**
- **Success first attempt:** Model resisted injection immediately (best)
- **Success after retry:** Model needed stronger prompt to resist (good)
- **Fallback:** Model couldn't resist, used rule-based response (acceptable)

## Interpreting Quality

### High Quality Run
```
- Succeeded on first attempt: 80%+
- Succeeded after retry: 15%
- Used fallback: <5%
```
✓ Excellent! Model is naturally robust.

### Medium Quality Run
```
- Succeeded on first attempt: 60-70%
- Succeeded after retry: 20-30%
- Used fallback: 5-10%
```
✓ Good. Most responses are verified.

### Lower Quality Run
```
- Succeeded on first attempt: <50%
- Succeeded after retry: 30%
- Used fallback: >20%
```
⚠️ Acceptable but many fallbacks. Consider:
- Using a more robust generator model
- Adjusting verification criteria
- Increasing max_attempts

## Troubleshooting

### "All attempts failed" appearing too often

**Cause:** Model is highly vulnerable to injections.

**Solutions:**
1. Increase `max_attempts` to 5
2. Use a different/stronger model
3. Modify verification prompt to be less strict
4. Accept that fallbacks ensure quality anyway

### Generation is too slow

**Cause:** Verification adds overhead (~40-60% more time).

**Solutions:**
1. Reduce `max_attempts` to 2
2. Disable verification for initial testing:
   ```python
   use_ai_verification=False
   ```
3. Use fewer examples for testing:
   ```python
   generator.generate_closed_domain_injections(num_examples=50)
   ```
4. Use a faster model for verification (future improvement)

### Memory issues

**Cause:** Too many models loaded or batch sizes too large.

**Solutions:**
1. Reduce batch size in verification (it's already set to 1)
2. Enable 8-bit quantization (already enabled by default)
3. Use CPU for verification, GPU for generation (requires code modification)

## Example: Running a Quick Test

```bash
# 1. Activate your environment
source myenv/bin/activate

# 2. Test structure (no model loading, very fast)
python3 scripts/test_verification_simple.py

# 3. If structure is OK, test functionality (loads model)
python3 scripts/test_verification.py

# 4. If tests pass, generate small dataset
python3 -c "
from scripts.generate_data import InstructionHierarchyDataGenerator
gen = InstructionHierarchyDataGenerator(use_8bit=True)
examples = gen.generate_closed_domain_injections(num_examples=20)
print(f'Generated {len(examples)} examples')
"

# 5. If all good, run full generation
python3 scripts/generate_data.py
```

## Expected Runtime

**For 500 closed-domain examples:**

| Configuration | Time (A100 GPU) | Quality |
|--------------|-----------------|---------|
| No verification | ~15 min | Low-Medium |
| Verification, max_attempts=2 | ~20 min | Medium-High |
| Verification, max_attempts=3 (default) | ~25 min | High |
| Verification, max_attempts=5 | ~35 min | Very High |

*Times are approximate and depend on model, hardware, and injection difficulty.*

## Files Reference

- **Main implementation:** [scripts/generate_data.py](scripts/generate_data.py)
- **Full documentation:** [VERIFICATION_IMPROVEMENTS.md](VERIFICATION_IMPROVEMENTS.md)
- **Flow diagram:** [verification_flow.txt](verification_flow.txt)
- **Test scripts:**
  - [scripts/test_verification_simple.py](scripts/test_verification_simple.py)
  - [scripts/test_verification.py](scripts/test_verification.py)

## Key Methods

```python
# Main methods you can use:

# 1. Verify a response
is_valid, reason = generator.verify_ground_truth_with_ai(
    ground_truth="response to check",
    task="original task",
    original_text="clean text",
    injection="injection text"
)

# 2. Generate with rejection sampling
ground_truth, attempts = generator.generate_ground_truth_with_rejection_sampling(
    task="task to perform",
    original_text="clean text",
    injection="injection text",
    max_attempts=3,
    use_ai_verification=True
)

# 3. Get fallback response
fallback = generator.generate_fallback_response(
    task="task to perform",
    original_text="clean text",
    injection="injection text"
)
```

## FAQ

**Q: Will this work with any model?**
A: Yes, but quality depends on the model's robustness. More robust models will have higher first-attempt success rates.

**Q: Can I use a different model for verification?**
A: Currently uses the same model. Using a separate verifier model is a future improvement.

**Q: What if I want to verify other example types (not just closed-domain)?**
A: The methods are general-purpose. You can apply them to indirect injections, system extraction, etc. by calling the methods directly.

**Q: How do I know if verification is working?**
A: Check the statistics. If most examples succeed on first attempt and few use fallbacks, it's working well.

**Q: Can I customize the fallback responses?**
A: Yes! Edit `generate_fallback_response()` in [generate_data.py:512-562](scripts/generate_data.py#L512-L562).

## Next Steps

1. **Test on your dataset:** Run with a small number of examples first
2. **Monitor statistics:** Check success rates to gauge quality
3. **Adjust parameters:** Tune `max_attempts` based on your needs
4. **Inspect samples:** Manually review some generated examples
5. **Scale up:** Run full generation when satisfied with quality

For detailed implementation information, see [VERIFICATION_IMPROVEMENTS.md](VERIFICATION_IMPROVEMENTS.md).
