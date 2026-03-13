"""
Model Distillation: Granite-Docling-258M → Gemma 3 1B
=====================================================

This script demonstrates how to distill document understanding capabilities
from the specialized Granite-Docling-258M model into Gemma 3 1B.

Architecture Overview:
- Teacher: granite-docling-258M (specialized VLM for document understanding)
- Student: Gemma 3 1B (general-purpose LLM)
- Goal: Transfer document structure understanding to the larger model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple
import json

# ============================================================================
# PART 1: Load Teacher and Student Models
# ============================================================================

class DistillationSetup:
    """Setup teacher and student models for distillation"""
    
    def __init__(
        self,
        teacher_model_name: str = "IBM/granite-docling-258M",
        student_model_name: str = "google/gemma-2-1b-it",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        
        # Load teacher model (Granite-Docling)
        print("Loading teacher model (Granite-Docling-258M)...")
        self.teacher_processor = AutoProcessor.from_pretrained(teacher_model_name)
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.teacher_model.eval()  # Teacher is always in eval mode
        
        # Load student model (Gemma 3 1B)
        print("Loading student model (Gemma 3 1B)...")
        self.student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
        self.student_model = AutoModelForCausalLM.from_pretrained(
            student_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print(f"Teacher parameters: {sum(p.numel() for p in self.teacher_model.parameters()) / 1e6:.1f}M")
        print(f"Student parameters: {sum(p.numel() for p in self.student_model.parameters()) / 1e6:.1f}M")


# ============================================================================
# PART 2: Prepare Distillation Dataset
# ============================================================================

class DocumentDistillationDataset(Dataset):
    """
    Dataset for document understanding distillation.
    Pairs document images with their structured outputs from the teacher model.
    """
    
    def __init__(
        self,
        documents: List[Dict],  # List of {image_path, query, ground_truth}
        teacher_processor,
        student_tokenizer,
        teacher_model,
        device: str = "cuda"
    ):
        self.documents = documents
        self.teacher_processor = teacher_processor
        self.student_tokenizer = student_tokenizer
        self.teacher_model = teacher_model
        self.device = device
        
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        doc = self.documents[idx]
        
        # Load document image
        image = Image.open(doc['image_path']).convert('RGB')
        query = doc['query']
        
        # Get teacher's logits (with no gradient)
        with torch.no_grad():
            # Process image + text for teacher (VLM)
            teacher_inputs = self.teacher_processor(
                text=query,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Get teacher's output logits
            teacher_outputs = self.teacher_model(**teacher_inputs, output_hidden_states=True)
            teacher_logits = teacher_outputs.logits  # Shape: [batch, seq_len, vocab_size]
            
        # Prepare student input (text-only)
        # We create a text-only prompt that describes the document task
        student_prompt = self._create_text_prompt(doc)
        student_inputs = self.student_tokenizer(
            student_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        return {
            'student_input_ids': student_inputs['input_ids'].squeeze(),
            'student_attention_mask': student_inputs['attention_mask'].squeeze(),
            'teacher_logits': teacher_logits.squeeze().cpu(),  # Soft targets
            'labels': doc.get('ground_truth', None)  # Hard targets (optional)
        }
    
    def _create_text_prompt(self, doc: Dict) -> str:
        """
        Convert document understanding task to text-only prompt for student.
        Since student doesn't see images, we need to describe the task clearly.
        """
        return f"""Document Understanding Task:

Query: {doc['query']}

Analyze the document and provide a detailed response based on the query. Focus on extracting structured information like:
- Tables and their contents
- Headers and sections
- Key-value pairs
- Lists and enumerations
- Document layout and structure

Response:"""


# ============================================================================
# PART 3: Distillation Loss Functions
# ============================================================================

class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation:
    - KL Divergence for soft targets (teacher logits)
    - Cross Entropy for hard targets (ground truth)
    """
    
    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.7,  # Weight for distillation loss
        beta: float = 0.3    # Weight for hard target loss
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            student_logits: [batch, seq_len, vocab_size]
            teacher_logits: [batch, seq_len, vocab_size]
            labels: [batch, seq_len] (optional, for hard target loss)
        """
        # Soft target distillation loss (KL divergence)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        distillation_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Hard target loss (if ground truth available)
        if labels is not None:
            hard_loss = self.ce_loss(
                student_logits.view(-1, student_logits.size(-1)),
                labels.view(-1)
            )
            total_loss = self.alpha * distillation_loss + self.beta * hard_loss
        else:
            total_loss = distillation_loss
        
        return total_loss


# ============================================================================
# PART 4: Custom Trainer for Distillation
# ============================================================================

class DistillationTrainer(Trainer):
    """Custom trainer that implements distillation loss"""
    
    def __init__(self, *args, temperature=2.0, alpha=0.7, **kwargs):
        super().__init__(*args, **kwargs)
        self.distillation_loss = DistillationLoss(
            temperature=temperature,
            alpha=alpha
        )
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Override to use distillation loss"""
        
        # Forward pass through student model
        student_outputs = model(
            input_ids=inputs['student_input_ids'],
            attention_mask=inputs['student_attention_mask'],
        )
        student_logits = student_outputs.logits
        
        # Get teacher logits (already computed and stored in dataset)
        teacher_logits = inputs['teacher_logits'].to(student_logits.device)
        
        # Compute distillation loss
        loss = self.distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=inputs.get('labels')
        )
        
        return (loss, student_outputs) if return_outputs else loss


# ============================================================================
# PART 5: Main Training Pipeline
# ============================================================================

def prepare_document_dataset(dataset_name: str = "doc-vqa-sample") -> List[Dict]:
    """
    Prepare a document understanding dataset.
    Replace with your own document dataset.
    """
    # Example structure - replace with real data
    documents = [
        {
            'image_path': 'documents/invoice_001.png',
            'query': 'Extract all line items from this invoice',
            'ground_truth': 'Line items: Item A: $100, Item B: $200, Item C: $150'
        },
        {
            'image_path': 'documents/form_002.png',
            'query': 'What is the customer name and address?',
            'ground_truth': 'Customer: John Doe, Address: 123 Main St'
        },
        # Add more documents...
    ]
    
    # Or load from Hugging Face
    # dataset = load_dataset("nielsr/docvqa_1200_examples_donut")
    
    return documents


def train_distilled_model():
    """Main training function"""
    
    # 1. Setup models
    setup = DistillationSetup()
    
    # 2. Prepare dataset
    print("Preparing dataset...")
    documents = prepare_document_dataset()
    
    train_dataset = DocumentDistillationDataset(
        documents=documents[:int(0.8 * len(documents))],
        teacher_processor=setup.teacher_processor,
        student_tokenizer=setup.student_tokenizer,
        teacher_model=setup.teacher_model
    )
    
    val_dataset = DocumentDistillationDataset(
        documents=documents[int(0.8 * len(documents)):],
        teacher_processor=setup.teacher_processor,
        student_tokenizer=setup.student_tokenizer,
        teacher_model=setup.teacher_model
    )
    
    # 3. Training arguments
    training_args = TrainingArguments(
        output_dir="./gemma-docling-distilled",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        save_total_limit=3,
        fp16=True,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        report_to="tensorboard"
    )
    
    # 4. Initialize trainer
    trainer = DistillationTrainer(
        model=setup.student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        temperature=2.0,  # Distillation temperature
        alpha=0.7         # Distillation loss weight
    )
    
    # 5. Train!
    print("Starting distillation training...")
    trainer.train()
    
    # 6. Save the distilled model
    trainer.save_model("./gemma-docling-distilled-final")
    setup.student_tokenizer.save_pretrained("./gemma-docling-distilled-final")
    
    print("Distillation complete!")


# ============================================================================
# PART 6: Inference with Distilled Model
# ============================================================================

def test_distilled_model():
    """Test the distilled Gemma model on document tasks"""
    
    # Load distilled model
    model = AutoModelForCausalLM.from_pretrained(
        "./gemma-docling-distilled-final",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("./gemma-docling-distilled-final")
    
    # Test prompt
    test_prompt = """Document Understanding Task:

Query: Extract the total amount from this invoice

Analyze the document and provide a detailed response based on the query. Focus on extracting structured information like:
- Tables and their contents
- Headers and sections
- Key-value pairs
- Lists and enumerations
- Document layout and structure

Response:"""
    
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Distilled model output:\n{response}")


# ============================================================================
# ALTERNATIVE: Simpler Response-Based Distillation
# ============================================================================

def simple_response_distillation():
    """
    Simpler approach: Generate responses from teacher on documents,
    then fine-tune student on those text pairs (no logits).
    """
    
    print("Generating teacher responses...")
    
    # Setup
    setup = DistillationSetup()
    documents = prepare_document_dataset()
    
    # Generate training pairs
    training_pairs = []
    
    for doc in documents:
        # Get teacher's response
        image = Image.open(doc['image_path']).convert('RGB')
        inputs = setup.teacher_processor(
            text=doc['query'],
            images=image,
            return_tensors="pt"
        ).to(setup.device)
        
        with torch.no_grad():
            outputs = setup.teacher_model.generate(
                **inputs,
                max_new_tokens=256
            )
        
        teacher_response = setup.teacher_processor.decode(outputs[0], skip_special_tokens=True)
        
        # Create training pair for student
        student_prompt = f"""Document Understanding Task:

Query: {doc['query']}

Response: {teacher_response}"""
        
        training_pairs.append({
            'text': student_prompt
        })
    
    # Save as JSON for fine-tuning
    with open('distillation_training_data.json', 'w') as f:
        json.dump(training_pairs, f, indent=2)
    
    print(f"Generated {len(training_pairs)} training pairs")
    print("Now fine-tune Gemma on this data using standard fine-tuning")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Distill Granite-Docling into Gemma 3 1B")
    parser.add_argument(
        '--mode',
        choices=['logit', 'simple', 'test'],
        default='logit',
        help='Distillation mode: logit (advanced), simple (response-based), or test'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'logit':
        train_distilled_model()
    elif args.mode == 'simple':
        simple_response_distillation()
    elif args.mode == 'test':
        test_distilled_model()
