import os
import json
import math
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)

from peft import LoraConfig, get_peft_model
from accelerate import Accelerator


# ===============================
# CONFIG
# ===============================

TEACHER_MODEL_NAME = "ibm-granite/granite-docling-258m"
STUDENT_MODEL_NAME = "google/gemma-3-1b"

DATA_DIR = "./documents"
PROMPT = "Extract structured fields from this document as JSON."

OUTPUT_DIR = "./distilled_gemma"
EPOCHS = 3
BATCH_SIZE = 2
LR = 2e-5
ALPHA = 0.4
TEMPERATURE = 3.0
MAX_LENGTH = 1024
USE_LORA = True


# ===============================
# DATASET
# ===============================

class DocumentDataset(Dataset):
    def __init__(self, image_dir):
        self.paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(("png", "jpg", "jpeg"))
        ]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        return {"image": image, "path": self.paths[idx]}


# ===============================
# DISTILLER
# ===============================

class DistillationTrainer:

    def __init__(self):

        self.accelerator = Accelerator(mixed_precision="fp16")

        # Teacher (frozen)
        self.teacher_processor = AutoProcessor.from_pretrained(TEACHER_MODEL_NAME)
        self.teacher = AutoModelForVision2Seq.from_pretrained(
            TEACHER_MODEL_NAME,
            torch_dtype=torch.float16
        )
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Student
        self.tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_NAME)
        self.student = AutoModelForCausalLM.from_pretrained(
            STUDENT_MODEL_NAME,
            torch_dtype=torch.float16
        )

        if USE_LORA:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.student = get_peft_model(self.student, lora_config)

        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=LR)

    # ===========================
    # TEACHER FORWARD
    # ===========================

    @torch.no_grad()
    def get_teacher_outputs(self, image):

        inputs = self.teacher_processor(
            images=image,
            text=PROMPT,
            return_tensors="pt"
        ).to(self.accelerator.device)

        outputs = self.teacher(
            **inputs,
            output_hidden_states=False
        )

        logits = outputs.logits
        generated_ids = torch.argmax(logits, dim=-1)

        text_output = self.teacher_processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return text_output, logits

    # ===========================
    # STUDENT FORWARD
    # ===========================

    def compute_loss(self, teacher_text, teacher_logits):

        # Tokenize teacher output in student vocab space
        student_inputs = self.tokenizer(
            teacher_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(self.accelerator.device)

        labels = student_inputs["input_ids"].clone()

        student_outputs = self.student(
            **student_inputs,
            labels=labels
        )

        student_logits = student_outputs.logits

        # Align teacher logits length
        teacher_logits = teacher_logits[:, :student_logits.size(1), :]

        # Resize teacher logits to student vocab if mismatch
        if teacher_logits.size(-1) != student_logits.size(-1):
            teacher_logits = teacher_logits[:, :, :student_logits.size(-1)]

        # KL Loss
        T = TEMPERATURE

        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)

        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean"
        ) * (T * T)

        ce_loss = student_outputs.loss

        loss = ALPHA * ce_loss + (1 - ALPHA) * kl_loss

        return loss

    # ===========================
    # TRAIN LOOP
    # ===========================

    def train(self, dataloader):

        self.student.train()

        total_steps = len(dataloader) * EPOCHS
        scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=total_steps * 0.05,
            num_training_steps=total_steps
        )

        self.student, self.optimizer, dataloader = self.accelerator.prepare(
            self.student, self.optimizer, dataloader
        )

        for epoch in range(EPOCHS):

            loop = tqdm(dataloader)
            for batch in loop:

                image = batch["image"][0]

                teacher_text, teacher_logits = self.get_teacher_outputs(image)

                loss = self.compute_loss(teacher_text, teacher_logits)

                self.accelerator.backward(loss)
                self.optimizer.step()
                scheduler.step()
                self.optimizer.zero_grad()

                loop.set_description(f"Epoch {epoch}")
                loop.set_postfix(loss=loss.item())

        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.student)
        unwrapped_model.save_pretrained(OUTPUT_DIR)
        self.tokenizer.save_pretrained(OUTPUT_DIR)


# ===============================
# MAIN
# ===============================

def main():

    dataset = DocumentDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    trainer = DistillationTrainer()
    trainer.train(dataloader)


if __name__ == "__main__":
    main()
