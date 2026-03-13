# Distilling Granite-Docling-258M into Gemma 3 1B
## Complete Implementation Guide

## Overview

This guide shows how to transfer document understanding capabilities from IBM's specialized Granite-Docling-258M vision-language model into Google's Gemma 3 1B general-purpose language model.

**Why This Matters:**
- Granite-Docling excels at understanding document structure (tables, forms, layouts)
- Gemma 3 1B is larger and better at reasoning but lacks specialized document knowledge
- Distillation combines the best of both worlds

---

## Quick Start

### 1. Installation

```bash
pip install torch transformers datasets pillow accelerate bitsandbytes
pip install -U huggingface_hub
```

### 2. Choose Your Approach

**Method A: Logit-Based Distillation** (Advanced, better quality)
- Transfers probability distributions from teacher to student
- Requires more computational resources
- Better preserves nuanced knowledge

```bash
python granite_to_gemma_distillation.py --mode logit
```

**Method B: Response-Based Distillation** (Simpler, faster)
- Teacher generates responses, student learns to mimic them
- Standard fine-tuning approach
- Easier to implement and debug

```bash
python granite_to_gemma_distillation.py --mode simple
```

---

## Detailed Walkthrough

### Method A: Logit-Based Distillation

#### Step 1: Understand the Architecture

```
┌─────────────────────────────────────┐
│  Teacher: Granite-Docling-258M      │
│  - Vision encoder (sees images)     │
│  - Language decoder                 │
│  - Outputs: logits (probabilities)  │
└─────────────────────────────────────┘
                 ↓
         Soft Targets (logits)
                 ↓
┌─────────────────────────────────────┐
│  Student: Gemma 3 1B                │
│  - Text-only (no vision)            │
│  - Learns to match teacher's        │
│    probability distributions        │
└─────────────────────────────────────┘
```

#### Step 2: Prepare Your Document Dataset

You need documents with:
- Image files (invoices, forms, receipts, etc.)
- Queries about those documents
- Optionally: ground truth answers

```python
documents = [
    {
        'image_path': 'invoice.png',
        'query': 'Extract line items with prices',
        'ground_truth': 'Item A: $100, Item B: $50'  # optional
    },
    # ... more documents
]
```

**Recommended Datasets:**
- DocVQA: Visual question answering on documents
- InfographicVQA: Infographic understanding
- Your own proprietary documents

#### Step 3: The Distillation Process

```python
# For each document:
# 1. Teacher sees image + query → produces logits
teacher_logits = teacher_model(image, query)  # Shape: [seq_len, vocab_size]

# 2. Student sees text-only prompt → produces logits
student_prompt = create_text_description(query)
student_logits = student_model(student_prompt)

# 3. KL Divergence Loss forces student to match teacher
loss = KL_divergence(student_logits, teacher_logits)
```

**Key Hyperparameters:**
- **Temperature (T=2.0)**: Higher = softer probabilities, more "dark knowledge" transfer
- **Alpha (α=0.7)**: Weight for distillation loss
- **Beta (β=0.3)**: Weight for hard target loss (if using ground truth)

#### Step 4: Training Configuration

```python
training_args = {
    'num_epochs': 3,
    'batch_size': 4,
    'gradient_accumulation': 4,  # Effective batch size = 16
    'learning_rate': 2e-5,
    'fp16': True,  # Use mixed precision
}
```

**Hardware Requirements:**
- Minimum: 1x GPU with 24GB VRAM (e.g., RTX 4090, A5000)
- Recommended: 2x GPUs with 40GB+ VRAM each
- Training time: ~6-12 hours for 10K documents

---

### Method B: Response-Based Distillation (Simpler Alternative)

This is more practical if you have limited resources:

#### Step 1: Generate Teacher Responses

```python
# Teacher generates text responses for each document
for document in dataset:
    teacher_response = teacher_model.generate(
        image=document['image'],
        query=document['query']
    )
    
    # Save as training pair
    training_data.append({
        'input': document['query'],
        'output': teacher_response
    })
```

#### Step 2: Fine-Tune Student on Generated Data

```bash
# Use standard fine-tuning (like LoRA or full fine-tuning)
# This is just supervised learning on teacher's outputs
```

**Advantages:**
- No need to handle logits or probability distributions
- Can use standard fine-tuning pipelines
- Much simpler to debug

**Disadvantages:**
- Loses "dark knowledge" (uncertainty information in logits)
- May not capture teacher's full understanding

---

## Advanced Techniques

### 1. Multi-Stage Distillation

```python
# Stage 1: Distill on synthetic document-free descriptions
# Stage 2: Distill on actual document tasks
# Stage 3: Fine-tune on hard examples
```

### 2. Feature Matching

In addition to logit matching, match intermediate representations:

```python
# Match hidden states between teacher and student
teacher_hidden = teacher_model(..., output_hidden_states=True)
student_hidden = student_model(..., output_hidden_states=True)

hidden_loss = MSE(student_hidden[-1], teacher_hidden[-1])
total_loss = distillation_loss + 0.1 * hidden_loss
```

### 3. Data Augmentation

Since the student doesn't see images, create diverse text descriptions:

```python
def augment_prompt(query):
    templates = [
        f"Document analysis request: {query}",
        f"Extract from document: {query}",
        f"Based on the document structure, {query}",
        # ... more variations
    ]
    return random.choice(templates)
```

---

## Evaluation

Test your distilled model:

```python
def evaluate_model(model, test_set):
    metrics = {
        'exact_match': [],
        'f1_score': [],
        'bleu': []
    }
    
    for doc in test_set:
        prediction = model.generate(doc['query'])
        ground_truth = doc['answer']
        
        # Compute metrics
        metrics['exact_match'].append(exact_match(prediction, ground_truth))
        metrics['f1_score'].append(f1_score(prediction, ground_truth))
    
    return metrics
```

---

## Common Issues & Solutions

### Issue 1: Out of Memory

**Solution:**
```python
# Use gradient checkpointing
model.gradient_checkpointing_enable()

# Reduce batch size
per_device_batch_size = 1
gradient_accumulation_steps = 16

# Use 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True
)
```

### Issue 2: Student Doesn't Learn

**Solution:**
- Lower the temperature (try T=1.5 or T=1.0)
- Increase alpha (weight for distillation loss)
- Check that teacher and student tokenizers align
- Verify logits are properly scaled

### Issue 3: Distribution Mismatch

The student sees text-only while teacher sees images.

**Solution:**
```python
# Create rich text descriptions of documents
def create_document_description(doc_metadata):
    return f"""
    Document Type: {doc_metadata['type']}
    Layout: {doc_metadata['layout_description']}
    Key Sections: {', '.join(doc_metadata['sections'])}
    
    Query: {doc_metadata['query']}
    """
```

---

## Production Deployment

Once distillation is complete:

```python
# Load your distilled model
model = AutoModelForCausalLM.from_pretrained(
    "./gemma-docling-distilled-final"
)

# Quantize for faster inference
from optimum.bettertransformer import BetterTransformer
model = BetterTransformer.transform(model)

# Deploy with vLLM or TGI for production
```

**Expected Performance:**
- 50-80% of teacher's accuracy on document tasks
- 3-5x faster inference (teacher requires image processing)
- Can run on CPU (with quantization)

---

## Further Reading

- [Knowledge Distillation in NLP](https://arxiv.org/abs/1910.01108) - DistilBERT paper
- [Distilling Vision-Language Models](https://arxiv.org/abs/2212.10559)
- [Gemma Model Documentation](https://ai.google.dev/gemma)
- [Granite-Docling Paper](https://huggingface.co/IBM/granite-docling-258M)

---

## License & Citation

If you use this distillation approach in research, please cite:

```bibtex
@misc{granite-gemma-distillation,
  title={Distilling Granite-Docling into Gemma for Document Understanding},
  author={Your Name},
  year={2024}
}
```
