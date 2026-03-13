# 🚀 Quick Start Guide: Distill Granite-Docling into Gemma 3 1B

## Choose Your Path

### Path 1: Just Want to Understand the Concept? 
**→ Start with `minimal_example.py`**

```bash
# Install dependencies
pip install torch transformers pillow

# Run the minimal example
python minimal_example.py
```

This shows you the core distillation concept in ~200 lines with detailed explanations.

---

### Path 2: Ready to Train? (Simple Approach)
**→ Use Response-Based Distillation**

```bash
# Step 1: Generate teacher responses
python granite_to_gemma_distillation.py --mode simple

# Step 2: Fine-tune student on generated data
# (Use any standard fine-tuning method like LoRA)
```

**Pros:** Simple, easy to debug, works with limited resources  
**Cons:** Loses some "dark knowledge" from teacher

---

### Path 3: Want Best Quality? (Advanced)
**→ Use Logit-Based Distillation**

```bash
# Install all dependencies
pip install -r requirements.txt

# Run full distillation
python production_training.py \
    --data_path ./your_documents.json \
    --epochs 3 \
    --batch_size 2
```

**Pros:** Best quality, preserves teacher's probability distributions  
**Cons:** More complex, needs more GPU memory

---

## What You Need

### Minimum Requirements
- 1x GPU with 24GB VRAM (e.g., RTX 4090, A5000)
- PyTorch 2.0+
- 100GB disk space

### Recommended
- 2x GPUs with 40GB+ VRAM
- Document dataset (1000+ examples)
- 500GB disk space (for checkpoints)

---

## Your Document Dataset Format

Create a JSON file with your documents:

```json
[
  {
    "image_path": "invoices/invoice_001.png",
    "query": "Extract all line items and prices",
    "ground_truth": "Item A: $100, Item B: $50"
  },
  {
    "image_path": "forms/form_002.png",
    "query": "What is the customer name?",
    "ground_truth": "John Doe"
  }
]
```

**Don't have data?** Use public datasets:
- [DocVQA](https://huggingface.co/datasets/nielsr/docvqa_1200_examples)
- [InfographicVQA](https://huggingface.co/datasets/lmms-lab/InfographicsVQA)
- [VisualMRC](https://huggingface.co/datasets/lmms-lab/VisualMRC)

---

## Quick Training Commands

### Start Training
```bash
python production_training.py \
    --data_path documents.json \
    --epochs 3 \
    --batch_size 2 \
    --lr 2e-5
```

### Resume from Checkpoint
```bash
python production_training.py \
    --data_path documents.json \
    --resume_from checkpoints/checkpoint-5000
```

### Monitor Progress
```bash
# In another terminal
tensorboard --logdir ./checkpoints
```

---

## Testing Your Distilled Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your distilled model
model = AutoModelForCausalLM.from_pretrained("./distilled_model/final")
tokenizer = AutoTokenizer.from_pretrained("./distilled_model/final")

# Test it
prompt = "Extract line items from this invoice: ..."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
```

---

## Common Issues & Quick Fixes

### Out of Memory?
```python
# Option 1: Smaller batch size
--batch_size 1 --gradient_accumulation_steps 16

# Option 2: Enable 8-bit quantization
# Edit DistillationConfig in production_training.py:
self.use_8bit = True
```

### Training Too Slow?
```python
# Use gradient checkpointing (trades speed for memory)
self.gradient_checkpointing = True

# Or use a smaller subset of data first to test
```

### Poor Results?
```python
# Try adjusting temperature
--temperature 1.5  # Lower = sharper distributions

# Or increase distillation weight
# Edit DistillationConfig:
self.alpha = 0.8  # More weight on teacher
self.beta = 0.2   # Less weight on ground truth
```

---

## Expected Results

After training on ~5000 documents for 3 epochs:

| Metric | Teacher (Granite) | Student (Gemma) | Improvement |
|--------|------------------|-----------------|-------------|
| Accuracy | 85% | 60-70% | N/A |
| Speed | 1x (slow, needs images) | 3-5x faster | 🚀 |
| Memory | 2GB (with images) | 0.5GB (text only) | 💾 |
| Deployment | Needs GPU | Can run on CPU | ✅ |

---

## Next Steps

1. ✅ Understand the concept: `minimal_example.py`
2. ✅ Read the guide: `README.md`
3. ✅ Prepare your data
4. ✅ Start training: `production_training.py`
5. ✅ Evaluate on test set
6. ✅ Deploy to production

---

## Files in This Package

| File | Purpose | When to Use |
|------|---------|-------------|
| `minimal_example.py` | Learn the concept | Start here! |
| `granite_to_gemma_distillation.py` | Full implementation | Reference code |
| `production_training.py` | Production training | Actually train |
| `README.md` | Complete guide | Detailed learning |
| `requirements.txt` | Dependencies | Installation |

---

## Getting Help

If you run into issues:

1. Check the detailed comments in each file
2. Read the README.md for deep explanations
3. Try the minimal example first to verify setup
4. Start with small batch size and few examples

---

## Performance Tips

### For Maximum Speed
- Use fp16 training
- Enable gradient checkpointing
- Use larger batch sizes with gradient accumulation
- Use multiple GPUs with distributed training

### For Maximum Quality
- Higher temperature (T=2.5 or T=3.0)
- More training epochs
- Larger dataset (10K+ documents)
- Match teacher and student sequence lengths

### For Limited Resources
- Use response-based distillation (simpler)
- 8-bit quantization
- Batch size = 1
- Smaller document subset for testing

---

## Success Checklist

Before starting production training:

- [ ] Verified setup with minimal_example.py
- [ ] Prepared document dataset (1000+ examples)
- [ ] Tested on small subset first
- [ ] Set up checkpointing and logging
- [ ] Allocated sufficient GPU memory
- [ ] Have evaluation metrics ready

---

## That's It! 🎉

You now have everything you need to distill Granite-Docling's document understanding into Gemma 3 1B.

Start with the minimal example, then scale up to production training.

Good luck! 🚀
