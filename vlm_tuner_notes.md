You can implement this exact self-supervised "analysis-by-synthesis" loop using Group Relative Policy Optimization (GRPO) or Proximal Policy Optimization (PPO) paired with a deterministic rendering engine.
This approach works best when the Vision-Language Model (VLM) outputs structured markup—such as HTML, LaTeX, or SVG with bounding-box coordinates—so a headless browser or compiler can re-render the prediction back into an image.

1. The RL Training Pipeline (Analysis-by-Synthesis)

Because standard rendering engines (like Playwright/Chromium or pdflatex) are non-differentiable, you cannot backpropagate pixel errors directly into the VLM. Reinforcement Learning solves this by treating the pixel similarity score as a black-box reward:
• Rollout Generation (GRPO): For a given document image \(I_{orig}\), sample \(G\) different structured text outputs \(\{y_1, y_2, \dots, y_G\}\) from your VLM policy.
• Deterministic Re-rendering: Pass each generated markup string \(y_{i}\) through a fast rendering engine to produce a reconstructed image \(\^{I}_{i}\).
• Reward Calculation: Compare \(\^{I}_{i}\) against \(I_{orig}\) to compute a scalar reward \(R_{i}\).
• Policy Update: Use GRPO to normalize rewards across the \(G\) samples and update the VLM weights, reinforcing outputs that visually reconstruct the original page most accurately.

2. Curriculum Learning: Layout-to-Word Progression

Starting at the layout level and progressing to word-level fidelity prevents sparse rewards early in training:
• Stage 1: Layout & Structural Alignment. Have the model predict high-level bounding boxes or empty HTML/CSS skeleton containers (tables, columns, headers). Render these blocks as solid gray or colored boxes and compare them against a binarized mask of the original document.
• Stage 2: Paragraph & Line Formatting. Introduce font sizes, line breaks, and alignment. At this stage, comparing blurred versions of \(I_{orig}\) and \(\^{I}_{i}\) helps the model lock in text density and block positioning before worrying about exact spelling.
• Stage 3: Word & Character Fidelity. Switch to high-resolution, sharp pixel comparisons. Any typo, missed word, or wrong symbol will cause a localized visual mismatch in the rendered text, forcing the VLM to correct its OCR accuracy.

3. Designing the Reward Function

Pure pixel-by-pixel subtraction (like L1/L2 loss) is too brittle because a 1-pixel shift in font rendering will ruin the score even if the text is 100% correct. Instead, combine multiple visual metrics into a composite reward:
\(R=w_{1}R_{\text{layout}}+w_{2}R_{\text{perceptual}}+w_{3}R_{\text{text\_edge}}-w_{4}P_{\text{syntax}}\)
• Layout IoU (\(R_{\text{layout}}\)): Convert both \(I_{orig}\) and \(\^{I}_{i}\) into blurred grayscale masks (using a Gaussian blur) and compute the Intersection over Union (IoU) or Dice score. This rewards the model for putting text in the right regions, even if the exact typeface differs slightly.
• Perceptual Similarity (\(R_{\text{perceptual}}\)): Use SSIM (Structural Similarity Index) or LPIPS (feature distance from a lightweight frozen vision encoder like CLIP or DINOv2) on cropped text patches. LPIPS is especially effective at catching wrong characters without penalizing minor sub-pixel anti-aliasing differences.
• Multi-Scale Edge / Gradient Similarity (\(R_{\text{text\_edge}}\)): Compare Sobel edge maps of the two images. Since text is defined by high-contrast strokes, edge matching sharply penalizes hallucinated words or missing punctuation.
• Syntax & Length Penalty (\(P_{\text{syntax}}\)): Apply a negative reward if the generated HTML/LaTeX fails to compile/render, or if the model cheats by generating excessively verbose markup.

4. Key Practical Challenges to Watch For

• Font Mismatch: If the original document uses a proprietary font and your renderer uses Arial, pixel-level comparison will hit a ceiling. You can mitigate this by having the VLM also predict basic font attributes (serif vs. sans-serif, bold, approximate font size) or by applying slight spatial tolerances (like chamfer distance) during comparison.
• Rendering Bottleneck: RL requires thousands of rollouts per step. Keep rendering fast by running a pool of persistent headless browser workers in memory or rendering directly to raw pixel buffers using lightweight libraries like Skia or Cairo.
If you'd like to dive deeper into setting this up, let me know:
• What target output format you plan to use (HTML/CSS, LaTeX, Markdown + bounding boxes?)
• The types of documents you are parsing (invoices, dense academic papers, forms, handwritten notes?)
I can help you write the exact reward calculation script or structure the GRPO training loop.

Using libraries like pdfplumber or PyMuPDF (fitz) to extract ground-truth bounding boxes and text from digital PDFs is a massive shortcut. It completely eliminates the rendering bottleneck and gives you exact, sub-pixel supervision without needing a headless browser.
However, pdfplumber is a parser, not a renderer—so depending on whether you have the original digital PDF or only a raw image/scan, you will use it in two slightly different ways.

1. When You Have the Original Digital PDF (Direct Ground-Truth RL)

If your training dataset starts from digital PDFs, you can render the PDF page to an image \(I_{\text{orig}}\) as the input to your VLM, while using pdfplumber in the background to extract the exact ground-truth list of words and bounding boxes:
\(GT=\{(w_{1},b_{1}),(w_{2},b_{2}),\dots ,(w_{n},b_{n})\}\)
Instead of re-rendering the VLM's output to pixels, you have the VLM output a structured list of (text, [x0, y0, x1, y1]) (e.g., in JSON or XML format) and compare it directly against pdfplumber's ground truth.
• Speed Advantage: Comparing JSON/lists in Python takes microseconds instead of the tens of milliseconds required to render an image in a browser.
• No Font Mismatch: You don't have to worry about Arial vs. Times New Roman pixel differences because you are comparing coordinates and strings directly.

2. When You Only Have Scanned Images (Ultra-Fast Canvas Re-rendering)

If your goal is to train on scanned documents or photos where no digital PDF exists (your original self-supervised "re-render and compare" idea), you still don't need HTML or a browser.
Instead, have the VLM output (text, [x0, y0, x1, y1]) and use a lightweight 2D drawing library like Pillow (PIL), OpenCV, or PyCairo to stamp the predicted text onto a blank white canvas at those exact bounding boxes:
1. VLM Output: {"text": "Total:", "bbox": [100, 500, 180, 525]}
2. Canvas Draw: Stretch/scale the font so "Total:" fits tightly inside [100, 500, 180, 525] on a blank image Î.
3. Pixel Comparison: Compare Î against the scanned document \(I_{\text{orig}}\). Drawing 500 words onto a NumPy array with Pillow/Cairo takes < 2 milliseconds—orders of magnitude faster than Chromium.

3. The Reward Function for pdfplumber Ground Truth

When comparing VLM predictions \(\hat{Y} = \{(\hat{w}_j, \hat{b}_j)\}\) directly against pdfplumber's extracted ground truth \(Y = \{(w_i, b_i)\}\), your reward function becomes a clean bipartite matching problem (using the Hungarian algorithm / scipy.optimize.linear_sum_assignment):
\(R=w_{1}R_{\text{box\_IoU}}+w_{2}R_{\text{text\_edit}}+w_{3}R_{\text{reading\_order}}-w_{4}P_{\text{hallucination}}\)
• Spatial Alignment (\(R_{\text{box\_IoU}}\)): Compute the Generalized IoU (GIoU) between predicted bounding boxes \(\^{b}_{j}\) and ground-truth boxes \(b_{i}\). GIoU gives a smooth gradient even when predicted boxes don't overlap with the target yet.
• Word Accuracy (\(R_{\text{text\_edit}}\)): For matched boxes, compute \(1 - \text{Normalized Levenshtein Distance}(\hat{w}_j, w_i)\). This rewards partial character matches (e.g., "Invoce" vs "Invoice" gets 0.85 instead of 0.0).
• Reading Order (\(R_{\text{reading\_order}}\)): Compare the sequence order of predicted words against pdfplumber's extracted order using Kendall's Tau or Normalized Spearman Footrule. This ensures the VLM reads multi-column layouts in the right sequence.
• Precision/Recall Penalty (\(P_{\text{hallucination}}\)): Penalize unmatched predicted boxes (false positives/hallucinations) and missed ground-truth boxes (false negatives).

4. Curriculum Learning with pdfplumber

pdfplumber naturally exposes hierarchical objects (rects, lines, curves, chars, words), making your curriculum stages effortless:
• Stage 1 (Layout Blocks): Cluster pdfplumber words into paragraphs/tables (using simple whitespace/distance heuristics or pdfplumber's .extract_tables()). Train the VLM to predict only block-level bounding boxes.
• Stage 2 (Line Detection): Drop down to line-level bounding boxes and rough word counts.
• Stage 3 (Word + Text): Require exact word-level coordinates and character-accurate text strings.

Watch Out for One pdfplumber Gotcha

Digital PDFs sometimes contain hidden OCR text layers underneath scanned images, or overlapping/invisible text used for search indexing. Before using pdfplumber output as RL ground truth, filter out characters where non_stroking_color is transparent/white on white, or verify that the bounding box actually contains dark pixels in the rendered PDF image.
Would you like a Python snippet showing how to build the fast Hungarian-matching reward function from pdfplumber output, or a Pillow-based canvas re-renderer for scanned images?


Here are both implementations so you can combine them or switch between them depending on whether your training batch has digital PDF ground truth or raw scanned images.

1. Fast Hungarian-Matching Reward (pdfplumber Ground Truth)

This script matches predicted (text, bbox) pairs against pdfplumber ground truth using the Hungarian algorithm (linear_sum_assignment). It computes GIoU for box accuracy, Levenshtein similarity for text accuracy, and Kendall's Tau for reading order.
python
import numpy as np
from difflib import SequenceMatcher
from scipy.optimize import linear_sum_assignment
from scipy.stats import kendalltau

def compute_giou_matrix(boxes1, boxes2):
    """
    boxes1: (N, 4) [x0, y0, x1, y1]
    boxes2: (M, 4) [x0, y0, x1, y1]
    Returns (N, M) GIoU matrix in [-1, 1]
    """
    b1 = boxes1[:, None, :]  # (N, 1, 4)
    b2 = boxes2[None, :, :]  # (1, M, 4)

    # Intersection
    inter_x0 = np.maximum(b1[..., 0], b2[..., 0])
    inter_y0 = np.maximum(b1[..., 1], b2[..., 1])
    inter_x1 = np.minimum(b1[..., 2], b2[..., 2])
    inter_y1 = np.minimum(b1[..., 3], b2[..., 3])
    inter_area = np.maximum(0, inter_x1 - inter_x0) * np.maximum(0, inter_y1 - inter_y0)

    # Union
    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
    union_area = area1 + area2 - inter_area + 1e-6
    iou = inter_area / union_area

    # Enclosing box
    enc_x0 = np.minimum(b1[..., 0], b2[..., 0])
    enc_y0 = np.minimum(b1[..., 1], b2[..., 1])
    enc_x1 = np.maximum(b1[..., 2], b2[..., 2])
    enc_y1 = np.maximum(b1[..., 3], b2[..., 3])
    enc_area = np.maximum(0, enc_x1 - enc_x0) * np.maximum(0, enc_y1 - enc_y0) + 1e-6

    return iou - (enc_area - union_area) / enc_area

def text_similarity(s1, s2):
    if not s1 and not s2:
        return 1.0
    return SequenceMatcher(None, s1, s2).ratio()

def compute_rl_reward(preds, gts, w_box=0.4, w_text=0.4, w_order=0.1, w_halluc=0.1):
    """
    preds / gts: list of dicts [{"text": str, "bbox": [x0, y0, x1, y1]}, ...]
    Coordinates should be normalized to.
    """
    if len(preds) == 0 and len(gts) == 0:
        return 1.0
    if len(preds) == 0 or len(gts) == 0:
        return -1.0

    pred_boxes = np.array([p["bbox"] for p in preds], dtype=np.float32)
    gt_boxes = np.array([g["bbox"] for g in gts], dtype=np.float32)

    # 1. Cost matrix combining spatial GIoU and text similarity
    giou_mat = compute_giou_matrix(pred_boxes, gt_boxes)
    text_mat = np.zeros((len(preds), len(gts)), dtype=np.float32)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            # Only compute string edit distance if boxes are remotely close
            if giou_mat[i, j] > -0.5:
                text_mat[i, j] = text_similarity(p["text"], g["text"])

    # Hungarian algorithm minimizes cost -> negate similarity
    cost_matrix = -(0.5 * giou_mat + 0.5 * text_mat)
    pred_idx, gt_idx = linear_sum_assignment(cost_matrix)

    # 2. Compute component rewards over matched pairs
    matched_giou = np.mean((giou_mat[pred_idx, gt_idx] + 1.0) / 2.0)  # Scale to [0, 1]
    matched_text = np.mean(text_mat[pred_idx, gt_idx])

    # 3. Reading order reward (Kendall's Tau on matched sequence)
    if len(gt_idx) > 1:
        tau, _ = kendalltau(pred_idx, gt_idx)
        order_reward = (tau + 1.0) / 2.0 if not np.isnan(tau) else 0.5
    else:
        order_reward = 1.0

    # 4. Hallucination / Recall F1 penalty
    precision = len(pred_idx) / len(preds)
    recall = len(gt_idx) / len(gts)
    f1_count = 2 * precision * recall / (precision + recall + 1e-6)

    total_reward = (
        w_box * matched_giou
        + w_text * matched_text
        + w_order * order_reward
        + w_halluc * f1_count
    )
    return float(total_reward)
Use code with caution.

2. Ultra-Fast Pillow Canvas Re-Renderer (For Scanned Images)

When you only have a raw image (no PDF text layer), use this function to stamp the VLM's predicted (text, bbox) pairs onto a white canvas in < 2ms and compare against the original scan using multi-scale blurred MSE + edge matching.
python
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Load default font once in memory
FONT = ImageFont.load_default()

def render_predictions_to_canvas(preds, width, height):
    """
    Renders predicted text scaled to fit inside each predicted bounding box.
    preds: list of dicts [{"text": str, "bbox": [x0, y0, x1, y1]}, ...] in [0, 1000]
    """
    canvas = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(canvas)

    for item in preds:
        text = item.get("text", "").strip()
        if not text:
            continue

        # Convert normalized [0, 1000] coords to pixel coords
        x0 = int(item["bbox"][0] * width / 1000)
        y0 = int(item["bbox"][1] * height / 1000)
        x1 = int(item["bbox"][2] * width / 1000)
        y1 = int(item["bbox"][3] * height / 1000)

        box_w, box_h = max(1, x1 - x0), max(1, y1 - y0)

        # Render word onto a tight temporary patch, then resize to fit bbox
        tmp_bbox = draw.textbbox((0, 0), text, font=FONT)
        tw, th = max(1, tmp_bbox[2] - tmp_bbox[0]), max(1, tmp_bbox[3] - tmp_bbox[1])

        word_patch = Image.new("L", (tw, th), color=255)
        ImageDraw.Draw(word_patch).text((-tmp_bbox[0], -tmp_bbox[1]), text, fill=0, font=FONT)

        # Stretch word patch to exact predicted bounding box dimensions
        word_patch = word_patch.resize((box_w, box_h), Image.Resampling.BILINEAR)
        canvas.paste(word_patch, (x0, y0))

    return np.array(canvas, dtype=np.float32) / 255.0

def compute_visual_reconstruction_reward(orig_gray_norm, rendered_gray_norm):
    """
    Compares original document image [0, 1] with rendered canvas.
    Uses multi-scale Gaussian blur so minor font differences aren't over-penalized.
    """
    # 1. Coarse layout similarity (heavy blur -> checks block alignment)
    orig_coarse = cv2.GaussianBlur(orig_gray_norm, (21, 21), 0)
    rend_coarse = cv2.GaussianBlur(rendered_gray_norm, (21, 21), 0)
    layout_score = 1.0 - np.mean(np.abs(orig_coarse - rend_coarse))

    # 2. Fine stroke/character density similarity (light blur -> checks word/char shape)
    orig_fine = cv2.GaussianBlur(orig_gray_norm, (3, 3), 0)
    rend_fine = cv2.GaussianBlur(rendered_gray_norm, (3, 3), 0)
    stroke_score = 1.0 - np.mean(np.abs(orig_fine - rend_fine))

    return float(0.4 * layout_score + 0.6 * stroke_score)
Use code with caution.

How to Combine Both in Curriculum Training

• Curriculum Weight Scheduling: During early GRPO steps, set w_box=0.7, w_text=0.1 (and use heavier blur in the visual renderer) so the model learns layout and coordinates first. Gradually shift to w_box=0.3, w_text=0.5 as box IoU stabilizes above 0.75.
• Coordinate Normalization: Always prompt your VLM to output coordinates normalized to a 0–1000 integer grid (like Qwen2.5-VL and PaliGemma do) rather than raw pixel values—this makes your reward function completely invariant to image resolution.
Would you like to see how to integrate this reward function into a GRPO training loop (e.g., using TRL / Unsloth / verl) or how to format the VLM system prompt and JSON schema so the model outputs clean coordinates without syntax errors?

Integrating firecrawl/pdf-inspector into your VLM RL pipeline is a major upgrade over pdfplumber. Because it is written in Rust with native Python bindings (pip install pdf-inspector), it solves both the speed bottleneck and several data-quality gaps that make pdfplumber tricky for RL ground truth.

1. pdf-inspector vs. pdfplumber: Why Switch for RL?


Feature / Gap	pdfplumber (Python)	firecrawl/pdf-inspector (Rust)	Impact on Your RL Training
Speed	~50–200ms per page	~2–10ms per page (200 docs in 0.47s)	Extracts ground truth on the fly during RL without slowing down GPU steps.
Scan vs. Digital Routing	No built-in detection; silently returns empty strings on scans.	Built-in classification (TextBased, Scanned, Mixed) + pages_needing_ocr.	Automatically routes digital pages to fast Hungarian Ground-Truth Reward and scanned pages to Pillow Visual Reconstruction Reward.
Reading Order	Raw PDF stream order (often scrambled in multi-column layouts).	Automatic multi-column & RTL reading order (0.915 NID benchmark score).	Gives you reliable ground-truth ordering for Kendall's Tau rewards without custom heuristics.
Rotated Text	Often collapses bounding boxes or splits characters weirdly.	True axis-aligned bounding box + rotation angle (TextItem).	Prevents false negatives on margin stamps, chart axes, and vertical table headers.
Broken Font Encodings	Extracts garbled □□□ characters silently.	Automatically flags broken encodings (ToUnicode CMap checks).	Prevents poisoning your VLM with corrupted ground-truth strings during RL.
Granularity	Exposes chars, words, lines, rects.	Exposes positioned TextItems (spans/runs), lines, layout blocks, and Markdown.	pdf-inspector outputs line/span items rather than individual words, so you split spans by whitespace if you want word-level boxes.


2. VLM System Prompt & Structured Output Schema

To prevent the VLM from wasting tokens on markdown chatter or generating invalid JSON during GRPO rollouts, enforce a strict 0–1000 normalized coordinate grid and a compact JSON schema.

System Prompt

text
You are a precision document layout and OCR parser. Extract all visible text elements from the document image in natural reading order.

Output ONLY a valid JSON array of objects. Each object must strictly follow this schema:
{"text": "<exact string>", "bbox": [x0, y0, x1, y1], "type": "<heading|text|table|list|caption>"}

Rules:
1. Coordinates [x0, y0, x1, y1] must be integers normalized to a 0 to 1000 scale, where (0,0) is top-left and (1000,1000) is bottom-right.
2. Preserve exact spelling, punctuation, and capitalization.
3. Do not include markdown code fences, explanations, or extra text outside the JSON array.
Use code with caution.
(Note: For Stage 1 Layout Curriculum, simply modify the prompt to omit the "text" key so the VLM only predicts {"bbox": [x0, y0, x1, y1], "type": "..."}.)

3. Hybrid Data Router (pdf-inspector + Dual Reward)

Use pdf-inspector in your dataset preprocessing step to classify each page and attach either exact ground-truth items (for digital pages) or flag it for visual canvas re-rendering (for scanned/corrupted-font pages):
python
import pdf_inspector

def prepare_rl_sample(pdf_path: str, page_num: int):
    """
    Uses pdf-inspector to route digital pages to exact GT reward
    and scanned/broken-encoding pages to visual canvas reward.
    """
    result = pdf_inspector.process_pdf(pdf_path)
    
    # Check if page needs OCR (scanned, image-based, or broken font encoding)
    if result.pdf_type in ("scanned", "image_based") or page_num in result.pages_needing_ocr:
        return {
            "mode": "visual_canvas",  # Use Pillow re-renderer + pixel similarity reward
            "gt_items": None
        }
    
    # Extract positioned TextItems (normalized to 0-1000)
    # Note: CLI `--items-json` or Rust/Python item bindings expose page width/height & item rects
    page_w, page_h = result.page_dimensions[page_num]
    gt_items = []
    for item in result.get_page_items(page_num):
        # Filter out invisible/white-on-white hidden text
        if item.render_mode == 3:  # PDF render mode 3 = invisible text
            continue
        norm_bbox = [
            int(item.x0 / page_w * 1000),
            int(item.y0 / page_h * 1000),
            int(item.x1 / page_w * 1000),
            int(item.y1 / page_h * 1000),
        ]
        gt_items.append({"text": item.text, "bbox": norm_bbox, "type": item.item_type})
        
    return {
        "mode": "hungarian_gt",  # Use fast Hungarian GIoU + Levenshtein reward
        "gt_items": gt_items
    }
Use code with caution.

4. GRPO Training Loop with trl (Hugging Face)

Using GRPOTrainer from Hugging Face trl, you can plug both reward functions directly into a VLM like Qwen2.5-VL-7B or Qwen3-VL:
python
import json
import numpy as np
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# Import the two reward functions from the previous step:
# compute_rl_reward, render_predictions_to_canvas, compute_visual_reconstruction_reward

def parse_vlm_json(completion_text: str):
    """Safely parses VLM output and applies a syntax penalty if malformed."""
    try:
        # Strip accidental markdown fences if present
        cleaned = completion_text.strip().removeprefix("```json").removesuffix("```").strip()
        data = json.loads(cleaned)
        if not isinstance(data, list):
            return None
        valid_items = []
        for entry in data:
            if isinstance(entry, dict) and "bbox" in entry and len(entry["bbox"]) == 4:
                valid_items.append({
                    "text": str(entry.get("text", "")),
                    "bbox": [max(0, min(1000, int(c))) for c in entry["bbox"]]
                })
        return valid_items
    except Exception:
        return None

def hybrid_document_reward_func(completions, mode, gt_items, orig_images, **kwargs):
    """
    TRL GRPO reward function called on every batch of G rollouts.
    """
    rewards = []
    for comp, sample_mode, gt, img in zip(completions, mode, gt_items, orig_images):
        # Extract text content from assistant message
        raw_text = comp[0]["content"] if isinstance(comp, list) else comp
        preds = parse_vlm_json(raw_text)
        
        # Heavy syntax penalty if the VLM outputs invalid JSON
        if preds is None:
            rewards.append(-1.5)
            continue

        if sample_mode == "hungarian_gt" and gt is not None:
            # Fast exact coordinate + text + reading order reward
            score = compute_rl_reward(preds, gt)
        else:
            # Self-supervised Pillow canvas re-rendering reward for scanned pages
            gray_norm = np.array(img.convert("L"), dtype=np.float32) / 255.0
            h, w = gray_norm.shape
            rendered = render_predictions_to_canvas(preds, w, h)
            score = compute_visual_reconstruction_reward(gray_norm, rendered)
            
        rewards.append(score)
    return rewards

# Configure GRPO
training_args = GRPOConfig(
    output_dir="./vlm-doc-parser-grpo",
    learning_rate=1e-6,
    per_device_train_batch_size=2,
    num_generations=8,          # G=8 rollouts per document image
    max_completion_length=2048,
    bf16=True,
    logging_steps=10,
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    reward_funcs=hybrid_document_reward_func,
    args=training_args,
    train_dataset=your_hf_dataset,  # Contains: "prompt", "image", "mode", "gt_items"
)

trainer.train()
Use code with caution.

Summary of Where Each Tool Shines in Your Pipeline

1. Before Training (Data Prep): Run pdf-inspector across your PDF corpus. It filters out corrupted fonts, separates digital pages from scanned pages in milliseconds, and gives you reading-order-sorted ground truth.
2. During RL Training (Digital Pages): Use the Hungarian Ground-Truth Reward on pdf-inspector items—zero rendering overhead, sub-millisecond evaluation, and exact reading order supervision.
3. During RL Training (Scanned Pages / Images): Use your Pillow Canvas Re-Renderer—gives you true self-supervised pixel-level feedback on hard scanned documents where no digital text layer exists.
Would you like to adjust the schema or reward function to also supervise table cell structure (using pdf-inspector's built-in table/rect detection) or curriculum stage scheduling inside the GRPOTrainer callback?


To build a complete end-to-end RL training setup, you can pair pdf-inspector (for blazing-fast layout, reading order, tables, and text spans) with pdfplumber (as a surgical fallback to split spans into exact word-level bounding boxes when pdf-inspector groups them into larger lines), and use a TRL TrainerCallback to automatically advance your curriculum stages.

1. Bridging the Granularity Gap (pdf-inspector + pdfplumber)

While pdf-inspector excels at reading order, multi-column detection, and table structure, its TextItem objects represent text runs/spans rather than individual words. If your VLM curriculum requires strict word-level bounding boxes in Stage 3, you can merge both libraries:
• Use pdf-inspector for: Document classification (TextBased vs. Scanned), broken font filtering, reading order, table grids/cells, and semantic element types (heading, list, code, caption).
• Use pdfplumber fallback for: Exact character/word-level bounding boxes inside complex spans, or extracting vector curves/lines when pdf-inspector falls back to heuristic table mode.
python
import json
import subprocess
import pdfplumber

def extract_hybrid_ground_truth(pdf_path: str, page_num: int):
    """
    Combines pdf-inspector's layout, reading order, and table structure
    with pdfplumber's exact word-level bounding boxes.
    """
    # 1. Run pdf-inspector CLI (--items-json gives axis-aligned boxes, font, paint, & item types)
    cmd = ["pdf2md", pdf_path, "--items-json", "--select-pages", str(page_num + 1)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    inspector_data = json.loads(proc.stdout)

    page_w = inspector_data["pages"][0]["width"]
    page_h = inspector_data["pages"][0]["height"]
    items = inspector_data["pages"][0]["items"]

    # 2. Extract word-level boxes from pdfplumber for fine-grained Stage 3 alignment
    with pdfplumber.open(pdf_path) as pdf:
        p_page = pdf.pages[page_num]
        plumber_words = p_page.extract_words(keep_blank_chars=False, use_text_flow=True)

    # Normalize pdf-inspector spans (preserves reading order & semantic types)
    span_gt = []
    for idx, item in enumerate(items):
        # Skip invisible render modes (mode 3) or transparent text
        if item.get("render_mode") == 3:
            continue
        x0, y0, x1, y1 = item["bbox"]
        span_gt.append({
            "id": idx,
            "text": item["text"],
            "bbox": [int(x0 / page_w * 1000), int(y0 / page_h * 1000),
                     int(x1 / page_w * 1000), int(y1 / page_h * 1000)],
            "type": item.get("item_type", "text"),  # heading, table, list, code, caption
            "rotation": item.get("rotation", 0.0)
        })

    # Normalize pdfplumber words for Stage 3 word-level supervision
    word_gt = []
    for w in plumber_words:
        word_gt.append({
            "text": w["text"],
            "bbox": [int(w["x0"] / page_w * 1000), int(w["top"] / page_h * 1000),
                     int(w["x1"] / page_w * 1000), int(w["bottom"] / page_h * 1000)],
            "type": "word"
        })

    return {"spans": span_gt, "words": word_gt}
Use code with caution.

2. Upgraded Schema & Table-Aware Reward Function

To teach the VLM table cell structure alongside text and coordinates, expand your JSON schema to include optional row and col indices (or Markdown table tags) whenever "type": "table":
json
[
  {"text": "Quarterly Revenue", "bbox":, "type": "heading"},
  {"text": "Q1 2026", "bbox":, "type": "table", "row": 0, "col": 1},
  {"text": "$4.2M", "bbox":, "type": "table", "row": 1, "col": 1}
]
Use code with caution.
Add two extra terms to your Hungarian reward function to supervise semantic type and table cell grid alignment:
python
def compute_structure_and_table_reward(preds, gts, pred_idx, gt_idx):
    """
    Computes rewards for semantic label accuracy (heading/list/table)
    and exact (row, col) grid placement inside tables.
    """
    type_matches = []
    table_cell_matches = []

    for p_i, g_i in zip(pred_idx, gt_idx):
        p_item, g_item = preds[p_i], gts[g_i]
        
        # 1. Semantic type reward (heading, list, table, code, caption)
        type_matches.append(1.0 if p_item.get("type") == g_item.get("type") else 0.0)

        # 2. Table cell grid reward (only evaluated on table items)
        if g_item.get("type") == "table":
            row_ok = (p_item.get("row") == g_item.get("row"))
            col_ok = (p_item.get("col") == g_item.get("col"))
            if row_ok and col_ok:
                table_cell_matches.append(1.0)
            elif row_ok or col_ok:
                table_cell_matches.append(0.5)  # Partial credit for right row or column
            else:
                table_cell_matches.append(0.0)

    type_score = float(np.mean(type_matches)) if type_matches else 1.0
    table_score = float(np.mean(table_cell_matches)) if table_cell_matches else 1.0
    return type_score, table_score
Use code with caution.

3. Automated 3-Stage Curriculum Scheduler (GRPOTrainer Callback)

Instead of manually stopping and restarting training runs, use a TrainerCallback in Hugging Face trl to dynamically shift reward weights, switch between span/word ground truth, and tighten the Pillow visual blur radius as training progresses:
python
from transformers import TrainerCallback

class DocumentCurriculumState:
    def __init__(self):
        self.stage = 1
        self.w_box = 0.70
        self.w_text = 0.05
        self.w_struct = 0.15
        self.w_order = 0.10
        self.blur_kernel = (25, 25)  # Heavy blur for Stage 1 layout blocks
        self.granularity = "spans"   # Start with coarse blocks/spans

curriculum = DocumentCurriculumState()

class CurriculumSchedulerCallback(TrainerCallback):
    """
    Automatically advances the VLM from:
      Stage 1 (0% - 25% steps): Coarse Layout & Bounding Boxes
      Stage 2 (25% - 60% steps): Line/Span Text, Tables & Reading Order
      Stage 3 (60% - 100% steps): Exact Word-Level & Character Fidelity
    """
    def on_step_begin(self, args, state, control, **kwargs):
        progress = state.global_step / max(1, state.max_steps)

        if progress < 0.25:
            curriculum.stage = 1
            curriculum.w_box, curriculum.w_text, curriculum.w_struct, curriculum.w_order = 0.70, 0.05, 0.15, 0.10
            curriculum.blur_kernel = (25, 25)
            curriculum.granularity = "spans"

        elif progress < 0.60:
            curriculum.stage = 2
            curriculum.w_box, curriculum.w_text, curriculum.w_struct, curriculum.w_order = 0.40, 0.30, 0.15, 0.15
            curriculum.blur_kernel = (9, 9)
            curriculum.granularity = "spans"

        else:
            curriculum.stage = 3
            curriculum.w_box, curriculum.w_text, curriculum.w_struct, curriculum.w_order = 0.30, 0.45, 0.10, 0.15
            curriculum.blur_kernel = (3, 3)  # Sharp pixel comparison for exact spelling
            curriculum.granularity = "words"
Use code with caution.

4. Plugging It All Into GRPOTrainer

Your reward function simply reads the live curriculum state on every step:
python
def dynamic_curriculum_reward_func(completions, mode, gt_spans, gt_words, orig_images, **kwargs):
    rewards = []
    # Select span-level or word-level GT based on current curriculum stage
    active_gts = gt_words if curriculum.granularity == "words" else gt_spans

    for comp, sample_mode, gt, img in zip(completions, mode, active_gts, orig_images):
        raw_text = comp["content"] if isinstance(comp, list) else comp
        preds = parse_vlm_json(raw_text)

        if preds is None:
            rewards.append(-1.5)
            continue

        if sample_mode == "hungarian_gt" and gt:
            score = compute_rl_reward(
                preds, gt,
                w_box=curriculum.w_box,
                w_text=curriculum.w_text,
                w_order=curriculum.w_order
            )
        else:
            gray_norm = np.array(img.convert("L"), dtype=np.float32) / 255.0
            h, w = gray_norm.shape
            rendered = render_predictions_to_canvas(preds, w, h)
            # Apply current curriculum blur kernel for coarse-to-fine pixel matching
            orig_blur = cv2.GaussianBlur(gray_norm, curriculum.blur_kernel, 0)
            rend_blur = cv2.GaussianBlur(rendered, curriculum.blur_kernel, 0)
            score = float(1.0 - np.mean(np.abs(orig_blur - rend_blur)))

        rewards.append(score)
    return rewards

# Pass the callback into GRPOTrainer
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    reward_funcs=dynamic_curriculum_reward_func,
    args=training_args,
    train_dataset=your_hf_dataset,
    callbacks=[CurriculumSchedulerCallback()],
)
Use code with caution.

One Pro-Tip for Faster Training

Pre-compute gt_spans and gt_words offline using multiprocessing before starting GRPOTrainer, and save them directly in your Hugging Face Arrow/Parquet dataset. That way, pdf-inspector and pdfplumber never block your GPU training loop, and each GRPO reward evaluation runs in under 1 millisecond.
Would you like to see how to handle multi-page PDF batching / image resolution scaling (e.g., configuring min_pixels and max_pixels in Qwen-VL so dense small-font text stays readable without blowing up VRAM)?






