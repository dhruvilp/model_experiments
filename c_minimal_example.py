"""
Minimal Distillation Example
=============================

This is a simplified version showing the core concept of distilling
Granite-Docling into Gemma without all the production complexity.

Perfect for understanding the basics before using the full implementation.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from PIL import Image

# ============================================================================
# STEP 1: Load Models
# ============================================================================

print("Loading models...")

# Teacher: Granite-Docling (vision-language model)
teacher_model = AutoModelForCausalLM.from_pretrained(
    "IBM/granite-docling-258M",
    torch_dtype=torch.float16,
    device_map="auto"
)
teacher_processor = AutoProcessor.from_pretrained("IBM/granite-docling-258M")
teacher_model.eval()

# Student: Gemma 3 1B (text-only model)
student_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-1b-it",
    torch_dtype=torch.float16,
    device_map="auto"
)
student_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-1b-it")

print(f"✓ Teacher loaded ({sum(p.numel() for p in teacher_model.parameters())/1e6:.0f}M params)")
print(f"✓ Student loaded ({sum(p.numel() for p in student_model.parameters())/1e6:.0f}M params)")


# ============================================================================
# STEP 2: Prepare One Example
# ============================================================================

# Example document and query
document_image = Image.open("sample_invoice.png")  # Replace with your image
query = "Extract all line items and their prices from this invoice"

print(f"\nQuery: {query}")


# ============================================================================
# STEP 3: Get Teacher's "Dark Knowledge" (Logits)
# ============================================================================

print("\n[Teacher] Processing image and generating logits...")

with torch.no_grad():
    # Teacher processes image + text
    teacher_inputs = teacher_processor(
        text=query,
        images=document_image,
        return_tensors="pt"
    ).to(teacher_model.device)
    
    # Get teacher's logits (probability distributions over vocabulary)
    teacher_outputs = teacher_model(**teacher_inputs)
    teacher_logits = teacher_outputs.logits  # Shape: [batch, seq_len, vocab_size]
    
    print(f"Teacher logits shape: {teacher_logits.shape}")
    print(f"These represent probability distributions over {teacher_logits.shape[-1]} tokens")


# ============================================================================
# STEP 4: Prepare Student Input (Text-Only)
# ============================================================================

print("\n[Student] Processing text-only input...")

# Since student can't see images, create a descriptive text prompt
student_prompt = f"""You are a document analysis expert. 

Task: {query}

Provide a detailed structured response extracting the requested information.

Response:"""

student_inputs = student_tokenizer(
    student_prompt,
    return_tensors="pt"
).to(student_model.device)

print(f"Student will learn from text prompt: '{student_prompt[:100]}...'")


# ============================================================================
# STEP 5: Compute Distillation Loss
# ============================================================================

print("\n[Distillation] Computing loss...")

# Forward pass through student
student_outputs = student_model(**student_inputs)
student_logits = student_outputs.logits

# Temperature for softening probability distributions
temperature = 2.0

# Soft targets from teacher (softened probabilities)
soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)

# Soft predictions from student (softened log probabilities)
soft_student = F.log_softmax(student_logits / temperature, dim=-1)

# KL Divergence loss (how different are the distributions?)
distillation_loss = F.kl_div(
    soft_student,
    soft_teacher,
    reduction='batchmean'
) * (temperature ** 2)

print(f"Distillation loss: {distillation_loss.item():.4f}")
print(f"(Lower is better - means student matches teacher's distribution)")


# ============================================================================
# STEP 6: Training Step (Conceptual)
# ============================================================================

print("\n[Training] What happens next...")
print("""
In actual training, you would:

1. Backward pass:
   distillation_loss.backward()
   
2. Optimizer step:
   optimizer.step()
   optimizer.zero_grad()
   
3. Repeat for thousands of documents
   
4. Student gradually learns to mimic teacher's probability distributions
   
5. Result: Gemma learns document understanding without seeing images!
""")


# ============================================================================
# STEP 7: Visualize What's Being Transferred
# ============================================================================

print("\n[Visualization] Teacher's 'Dark Knowledge':")

# Get top-5 most likely next tokens from teacher
teacher_probs = F.softmax(teacher_logits[0, -1, :], dim=-1)
top_teacher_tokens = torch.topk(teacher_probs, k=5)

print("\nTeacher's top 5 next token predictions:")
for prob, token_id in zip(top_teacher_tokens.values, top_teacher_tokens.indices):
    token = teacher_processor.tokenizer.decode([token_id])
    print(f"  '{token}': {prob.item()*100:.2f}%")

print("\n✨ The student learns these nuanced probability distributions!")
print("Not just the single most likely token, but the full distribution.")
print("This is the 'dark knowledge' that makes distillation powerful.")


# ============================================================================
# STEP 8: Compare Before/After (Conceptual)
# ============================================================================

print("\n" + "="*60)
print("UNDERSTANDING THE BENEFIT:")
print("="*60)

print("""
BEFORE DISTILLATION:
  Student (Gemma): "I can't analyze this document because I can't see images"
  
DURING DISTILLATION:
  Teacher: "For documents like this, the next tokens should be..."
  Student: "I'll learn those patterns even without seeing the image"
  
AFTER DISTILLATION:
  Student (Gemma): "Based on the query, here are the extracted line items:
                     Item A: $100, Item B: $50..."
                     
The student learns document understanding patterns through text-only training,
guided by the teacher's visual understanding!
""")


# ============================================================================
# Key Insights
# ============================================================================

print("\n" + "="*60)
print("KEY INSIGHTS:")
print("="*60)

print("""
1. TEACHER SEES IMAGES → Student learns from teacher's outputs (logits)
   
2. TEMPERATURE MATTERS:
   - T=1: Sharp distributions (one token dominates)
   - T=2-3: Soft distributions (reveals teacher's uncertainty)
   - Higher T transfers more "dark knowledge"
   
3. WHY LOGITS NOT TEXT?
   - Text: "Item A: $100" (single answer)
   - Logits: P("Item"=0.7, "Line"=0.2, "Product"=0.1)
   - Logits reveal teacher's reasoning process!
   
4. MULTI-MODAL → TEXT-ONLY:
   - Teacher: Vision encoder → Language decoder
   - Student: Pure language model (faster, more flexible)
   - Student approximates teacher's behavior without images!
""")


# ============================================================================
# Next Steps
# ============================================================================

print("\n" + "="*60)
print("NEXT STEPS TO BUILD FULL SYSTEM:")
print("="*60)

print("""
1. Collect document dataset (1000+ examples)
   - DocVQA, InfographicVQA, or your own data
   
2. Generate teacher logits for all documents
   - Save them (they're large!) or compute on-the-fly
   
3. Set up training loop:
   - DataLoader, Optimizer, Learning rate schedule
   - Track loss convergence
   
4. Train for 3-5 epochs
   
5. Evaluate on held-out test set
   - Compare with teacher's accuracy
   - Expect 60-80% of teacher's performance
   
6. Deploy distilled model
   - Faster (no image processing)
   - Smaller memory footprint
   - Can run on CPU with quantization
   
See granite_to_gemma_distillation.py for full implementation!
""")
