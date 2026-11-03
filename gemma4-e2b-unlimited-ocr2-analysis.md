# Unlimited-OCR-2 vs Gemma-4-E2B-it — Architecture, Weights & Transfer Analysis

> **Status: analysis complete; empirical validation done** — weight autopsy, training-free surgery, and ParseBench results included.
> Analysis date: 2026-11-03. All facts below are fetched from the HF repos, configs, modeling source, raw safetensors, and arXiv papers — not from prior knowledge (both models are post-cutoff releases).

---

## Table of contents

1. [TL;DR](#1-tldr)
2. [Unlimited-OCR-2 (baidu)](#2-unlimited-ocr-2)
3. [Unlimited-OCR v1 → v2 diff](#3-v1--v2-diff)
4. [What makes Unlimited-OCR-2 fast](#4-what-makes-uo2-fast)
5. [Gemma-4-E2B-it (google)](#5-gemma-4-e2b-it)
6. [Head-to-head: sliding-window techniques](#6-head-to-head-sliding-window)
7. [Head-to-head: weights & core config](#7-head-to-head-weights--config)
8. [Applying Unlimited-OCR-2's good ideas to Gemma-4-E2B](#8-transfer-plan)
9. [Empirical: autopsy, surgery validation, ParseBench](#9-empirical)
10. [Appendix: verbatim configs & references](#10-appendix)
11. [Replication guide — from scratch to the validated numbers](#11-replication-guide--from-scratch-to-the-validated-numbers)

---

## 1. TL;DR

- **Unlimited-OCR-2** is a 3.34B DeepSeek-V2-style MoE (64 routed + 2 shared experts, 6/token → **~0.73B active**) with MLA disabled, a dual SAM+CLIP vision tower, and — new in v2 — **DASH/SR-SWA learned block-sparse attention**: every decode step reads only top-12 32-token blocks ∪ 128 local ≈ **~512 fixed reference tokens** regardless of document length, giving claimed 7.0×/5.6× per-step speedups over Full/R-SWA at 100 pages. v2 = v1 weights + 109 tensors (+7.87M params, all router/indexer).
- **Gemma-4-E2B-it** is a 5.12B dense multimodal model (2.3B "effective" + 2.35B lookup-only PLE table + vision/audio towers), 35 layers in a 4:1 sliding:full pattern (window 512), MQA, per-layer-type RoPE (local θ=10k; global p-RoPE 25% rotary/75% NoPE), KV-sharing for layers 15–34. Its 7 global layers read **everything** every step — the exact cost UO2 eliminates.
- **Transfer without training** (validated here): monkeypatch the eager attention in transformers 5.x (§8.3 gotcha: a custom `attn_implementation` string silently kills the causal mask) to give the 7 global layers **Quest-style top-12-blocks ∪ 128-local** decode reads — SR-SWA's ~512-token budget with zero weight changes. Verified bit-safe (prefill cos 1.0, 100% greedy match, sparse path confirmed firing on all 7 layers).
- **ParseBench (gold-rule) results** on a deterministic 24-doc text subset — baseline vs surgery in §9.
- **Not portable without training**: DASH's learned router itself, page2d RoPE, MoE upcycling. Gemma-4's MRCR@128k = 19.1% warns against trusting sparsified global layers at extreme length without re-checking.

---

## 2. Unlimited-OCR-2

`UnlimitedOCRForCausalLM`, `model_type: "unlimited-ocr"`, custom code (`modeling_unlimitedocr.py`), MIT license, BF16, 2 safetensors shards. Lineage: built on DeepSeek-OCR (acknowledges DeepSeek-OCR, DeepSeek-OCR-2, PaddleOCR). Repo created 2026-10-26; v1 (`baidu/Unlimited-OCR`) 2026-06-19. All facts below verified against config.json, safetensors headers, and the custom code — full raw notes in `uo2-notes.md` alongside this file.

### 2.1 Language backbone — DeepSeek-V2-style MoE with MLA **disabled**

| Field | UO2 value |
|---|---|
| layers / hidden | **12 / 1280** |
| attention | **plain MHA, 10 Q / 10 KV heads** (no GQA), v_head_dim 128 → attn inner dim 1280; `use_mla: false`, all LoRA-rank fields null → **no MLA** despite the DeepSeek-V2 config lineage |
| dense MLP (layer 0 only) | intermediate 6848, SiLU |
| MoE (layers 1–11) | **64 routed + 2 shared experts, 6 routed/token**, expert intermediate 896, greedy top-k, softmax scoring, no topk norm, aux_loss 0.001 |
| vocab | 129,280, **untied** embed + lm_head |
| context | 131,072 via **YaRN ×4** from 32,768, θ=500,000 |
| norm / eps | RMSNorm 1e-6 |

Stock DeepSeek-V2 for contrast: hidden 4096, 30 layers, 32 heads, MLA on (kv_lora_rank 512). UO2 is a heavily shrunk, MLA-off, sliding-window-added variant.

### 2.2 Vision — `deeplip_b_l` dual tower (SAM-ViT-B + CLIP-L/14-224)

- **SAM ViT-B** branch: width 768, 12 layers, global attn at blocks [2,5,8,11] (window-14 elsewhere), 1024² input, patch 16 → 64×64; neck 1×1 conv→256 + 3×3 conv→256; two stride-2 convs → **16×16 = 256 tokens @1024 channels** (16:1 spatial compression).
- **CLIP-L/14-224** branch: 24 layers, width 1024, 16 heads, patch 14; pos-embed bicubic-resized for other grids; fused QKV SDPA.
- **Fusion**: concat SAM(1024)+CLIP(1024) → linear projector 2048→1280 (`model.projector.layers.weight` [1280, 2048]).
- Token layout per page: global view `16×17+1` with a learned [1280] `image_newline` row separator + learned `view_seperator`; page = concat(local_tiles, global_view, view_seperator) inserted via **masked_scatter_** into image positions. **273 image tokens per page**. Vision runs under `torch.no_grad()`, skipped during single-token decode.

### 2.3 The attention stack — three layered mechanisms

1. **R-SWA (Reference SWA, v1, arXiv:2606.23050)** — ring-buffer KV: prefill tokens are **pinned**; decode-time new KV overwrites a **128-slot ring** in place while attention still reads the full cache → "infinite prefill + W-sized overwrite ring", constant KV growth during decode. NOT a normal dropping sliding window.
2. **DASH (Differentiable Adaptive Sparse Hierarchical attention, arXiv:2605.18753)** — the learned sparse router, new in v2:
   - **Stage 0** `head_cls`: a learned summary query (zero-init→mean-pool) attends within each chunk to build block summary key `k̄_c`; block score = idx_q·k̄_c.
   - **Stage 2 (routing)**: **1.5-entmax** over candidate blocks (`entmax_topk_perlayer`) → sparse differentiable prior, then **hard top-k, k annealed 64→12** over ~1000 steps (final `dash_topk_blocks=12`); every query reads exactly min(k,#candidates) remote blocks/layer.
   - **Bias**: `bias=(log(clamp(p,floor))−μ)/σ`, mean-centered, added to selected blocks' logits (scale 128, σ=1.0, floor 1e-4).
   - **Per-layer closed loop**: each layer's entmax temperature adapts toward setpoint=current k; floor anneals geometrically.
   - **Trained end-to-end** via entmax from LM loss — "no KL teacher / no two-stage curriculum". Learned params per layer: q_proj,k_proj [128,1280], q/k_norm, head_cls [128] ≈ 0.66M/layer.
   - **Image-aware**: `dash_image_only=true` (candidates = blocks with ≥1 image token), `image_bucket_constraint=true` (page-aware per-query domains).
   - Budget: "12 blocks is ~2.3% of the KV at 16k."
3. **SR-SWA (v2 umbrella, arXiv:2610.39308)** = R-SWA's constant-reference idea + DASH learned selection. Keep-set = top-12 remote blocks (of 32-token blocks) ∪ 128-token local window ≈ **~512 fixed reference tokens** → per-step cost independent of document length.
   - **Anchor/Reuse machinery**: `sparse_attention_freq=[1×12]` = every layer an Anchor computing its own indices (freq==0 layers would reuse the previous Anchor's — shipped but unused).
   - **page2d RoPE**: positions become **(tok_pos, page_pos)** split in *frequency-pair* space (rotate_half pairs dim j with j+dim/2, so slicing raw dims would mix axes). Dedicated page θ=256 → monotone unambiguous range ~402 pages; `page_mod=2` folds page coordinate to a same/adjacent-page parity marker at angle time. `resample` freq mode: token axis re-derives inv_freq at reduced dim (56 vs 64 pairs) so it still sub-samples the whole spectrum. KV cache stores already-rotated K; decode advances `_page_rope_state` on `<PAGE>` tokens.
   - At serve time DASH **replaces** the ring window for reads (`configure_dash_inference` sets `_ring_window=None, sliding_window=None` — "no ring-buffer SWA, vLLM attends via DASH"); the ring is primarily a training/storage device.

### 2.4 Triton kernels & inference wiring

- `_gather_attn_fwd_kernel` (+ bwd dq/dkv): per token, gather only its top-k KV blocks (int32 indices, −1=pad) + local causal window, single online softmax, **no gathered-block buffer** — K/V tiles loaded via `tl.make_block_ptr` (coalesced) + `tl.dot` tensor-core MMAs. GQA head-group padded G=2→16 free ("memory-bound on K/V block loads; padded tensor-core lanes ~free"). Duplicate block ids deduped to −1.
- `_block_max_score_kernel`: fused QK-score + per-block amax — no dense [B,C,Sk] score tensor materialized.
- Decode loop extras: `_uocr_no_repeat_ngram` filter and `_uocr_page_count_guard` (suppresses EOS until all expected pages emitted; forces `<PAGE>`/EOS at page boundaries). Prompt format `"<PAGE><image>"*n + prompt` (default "Multi page parsing.").
- Ships a patched **sglang 0.5.20.dev13 wheel** + `packing_patch.py` (segment-aware masks for packed training; fused kernels bypassed when packed).

### 2.5 Weights census (from safetensors headers — 3,343,978,497 params, 2,819 tensors)

| Component | Params | % | tensors |
|---|---|---|---|
| MoE routed experts (64×11 layers) | 2,422,210,560 | **72.4%** | 2,112 |
| CLIP tower | 303,177,985 | 9.1% | 294 |
| embed_tokens [129280,1280] | 165,478,400 | 4.9% | 1 |
| lm_head [129280,1280] | 165,478,400 | 4.9% | 1 |
| SAM tower | 95,569,152 | 2.9% | 179 |
| main attention q/k/v/o | 78,643,200 | 2.4% | 48 |
| MoE shared experts | 75,694,080 | 2.3% | 33 |
| dense MLP (layer 0) | 17,530,880 | 0.5% | 2 |
| MoE routers | 9,666,560 | 0.3% | 12 |
| **DASH router (v2)** | **3,936,768** | 0.12% | 60 |
| **sparse indexer (v2)** | **3,935,232** | 0.12% | 48 |
| projector + norms + misc | ~2.7M | 0.08% | 29 |

**Active params/token ≈ 0.73B of 3.34B (~22%)** — embed read 165M + attn ~79M + MoE active (6 routed ×896×1280×3 ≈ 20.6M + shared 6.9M ≈ 27.5M/layer × 11 ≈ 302M) + dense L0 17.5M + lm_head 165M. This is UO2's dominant speed lever.

## 3. v1 → v2 diff

**v2 = v1 weights + 109 new tensors; zero removed; zero shape changes.** The ONLY substantive config change is the addition of `sparse_attention_config` (plus cosmetic `model_type` spelling and transformers 4.46.3→4.52.1).

| Area | v1 (2026-06-19) | v2 (2026-10-26) |
|---|---|---|
| Attention | R-SWA (ring SWA + fixed references) | **SR-SWA** = R-SWA + **DASH learned sparse retrieval** |
| `sparse_attention_config` | absent | present (dash_* + sparse_* + page2d rope) |
| Weight tensors / params | 2,710 / 3,336,106,240 | 2,819 / 3,343,978,497 (**+109 tensors**: 12 layers × (5 dash_router + 4 indexer) + 1 CLIP position_ids; **+7.87M params, +0.24%**) |
| Code files | modeling + deepencoder | **+ dash_attention.py, sparse_gather_triton.py, sparse_gather_e2e_triton.py, _sparse_base_impl.py, packing_patch.py** |
| sglang wheel | 0.0.0.dev11416 | 0.5.20.dev13 |
| Attention backend | `--attention-backend fa3` | DASH via sglang, `--tokenizer-backend fastokens` |
| Context claim | 32k, "dozens of pages" | 128k (YaRN ×4), "unlimited horizon", hundred-page benchmarks |
| Paper / speed metric | arXiv:2606.23050, "constant KV cache" | arXiv:2610.39308, **"7.0× / 5.6× per-step vs Full / R-SWA"** at 100 pages |
| Benchmarks | OmniDocBench | + MPDocBench, MarathonDocBench (SOTA) |

Unchanged: backbone dims (1280/12L/10h), MoE (64e/6/2sh), vision tower, vocab, YaRN rope, sliding_window_size 128, untied embed/lm_head.

## 4. What makes UO2 fast

Verbatim claims (sources): "keeping per-step decoding latency and memory footprint nearly constant regardless of document length"; "unbounded document parsing within a finite context window"; "Compresses text-visual attention interactions from ultra-long sequences down to a fixed-size set of working references (e.g., 512 tokens)"; "**7.0× and 5.6× per-step speedups over Full Attention and R-SWA**" at hundred-page scale (arXiv:2610.39308, A800-80GB). DASH paper: "comparable accuracy as full attention with 75% sparsity", "speedup over FlashAttention-3 at inference".

Synthesis, in order of leverage:

1. **MoE sparsity** — only ~22% of params active per token (top-6/64 experts). Training-time architecture, the single biggest FLOP lever.
2. **Constant-size decode working set** — top-12-of-32-token-blocks + 128 local ≈ ~512 reference tokens regardless of document length (vs O(N) full attention). This is the per-step *latency* lever and the one that transfers.
3. **Learned, image-aware block selection** — DASH retrieves only relevant page-image blocks; image-only candidate restriction slashes the candidate pool (visual tokens dominate OCR sequences).
4. **Fused Triton gather** — no dense score tensor, coalesced block loads, free padded lanes. Kernel-level constant factor.
5. **No MLA overhead** — small 1280-dim MHA projections.
6. **page2d RoPE** — keeps position deltas small and document-length-independent → stable retrieval at extreme length, no extra compute.
7. **High visual compression** — 273 tokens/page → fewer KV entries per page to select over.

---

## 5. Gemma-4-E2B-it

### 5.1 Positioning

- `google/gemma-4-E2B-it`, Apache-2.0, created 2026-03-02, `Gemma4ForConditionalGeneration`, `model_type: "gemma4"`, BF16, **5,123,178,979 params** across 5 safetensors shards (~10.2 GB; HF metadata says 5,123,178,051 — the 928-param delta is scalar buffers the metadata summary skips). Requires bleeding-edge `transformers==5.5.0.dev0`.
- "E2B" = **effective 2.3B params** — 5.12B total, of which a giant **per-layer embedding (PLE) table (2.35B)** is lookup-only, so per-token compute ≈ a 2.3B model (card: "These embedding tables are large but are only used for quick lookups, which is why the effective parameter count is much smaller than the total").
- Natively **any-to-any multimodal**: text + vision + audio towers (video via image path). Instruction-tuned (`-it`) finetune of `google/gemma-4-E2B`; thinking mode, function calling, native system role, 140+ languages.
- Family context (arXiv:2607.02770, Gemma 4 Technical Report, 2 Jul 2026): E2B, E4B (4.5B effective/8B total), 12B Unified (encoder-free), 26B-A4B MoE (8 active of 128), 31B; 128K context for small, 256K for medium models.

### 5.2 Core config (verbatim structure)

```json
{
  "architectures": ["Gemma4ForConditionalGeneration"],
  "model_type": "gemma4",
  "dtype": "bfloat16",
  "tie_word_embeddings": true,
  "transformers_version": "5.5.0.dev0",
  "image_token_id": 258880, "audio_token_id": 258881, "video_token_id": 258884,
  "boi_token_id": 255999, "eoi_token_id": 258882, "boa_token_id": 256000, "eoa_token_id": 258883,
  "eos_token_id": [1, 106],
  "vision_soft_tokens_per_image": 280,
  "text_config": {
    "model_type": "gemma4_text",
    "hidden_size": 1536, "num_hidden_layers": 35,
    "num_attention_heads": 8, "num_key_value_heads": 1,
    "head_dim": 256, "global_head_dim": 512,
    "intermediate_size": 6144, "use_double_wide_mlp": true,
    "hidden_activation": "gelu_pytorch_tanh",
    "vocab_size": 262144, "vocab_size_per_layer_input": 262144,
    "hidden_size_per_layer_input": 256,
    "num_kv_shared_layers": 20, "attention_k_eq_v": false,
    "enable_moe_block": false, "num_experts": null, "top_k_experts": null,
    "sliding_window": 512,
    "layer_types": ["sliding_attention"×4, "full_attention"] × 7  (35 layers; full at idx 4,9,14,19,24,29,34),
    "rope_parameters": {
      "full_attention":   {"rope_theta": 1000000.0, "rope_type": "proportional", "partial_rotary_factor": 0.25},
      "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
    },
    "max_position_embeddings": 131072,
    "final_logit_softcapping": 30.0,
    "attention_bias": false, "rms_norm_eps": 1e-06,
    "bos_token_id": 2, "eos_token_id": 1, "pad_token_id": 0
  },
  "vision_config": {
    "model_type": "gemma4_vision", "hidden_size": 768, "num_hidden_layers": 16,
    "num_attention_heads": 12, "num_key_value_heads": 12, "head_dim": 64,
    "intermediate_size": 3072, "patch_size": 16, "pooling_kernel_size": 3,
    "position_embedding_size": 10240,
    "rope_parameters": {"rope_theta": 100.0, "rope_type": "default"},
    "use_clipped_linears": true, "hidden_activation": "gelu_pytorch_tanh"
  },
  "audio_config": {
    "model_type": "gemma4_audio", "hidden_size": 1024, "num_hidden_layers": 12,
    "num_attention_heads": 8, "output_proj_dims": 1536,
    "attention_chunk_size": 12, "attention_context_left": 13, "attention_context_right": 0,
    "conv_kernel_size": 5, "subsampling_conv_channels": [128, 32],
    "attention_logit_cap": 50.0, "residual_weight": 0.5,
    "hidden_act": "silu", "use_clipped_linears": true
  }
}
```

`generation_config.json`: `bos=2, eos=[1,106,50], pad=0, do_sample=true, temperature=1.0, top_k=64, top_p=0.95`.
`processor_config.json`: image `image_seq_length=280`, `max_soft_tokens=280` (budgets 70–1120), video 32 frames × 70 tokens, audio 16 kHz / 128 mel / 40 ms per token / 750 tokens max (30 s).

### 5.3 Architecture highlights

- **Text decoder**: 35 layers, hidden 1536, GELU-tanh MLP, RMSNorm pre+post on attn & MLP, QK-Norm, `final_logit_softcapping=30`, per-layer scalar (`layer_scalar`) bounding activations for fp16 (paper §2.5).
- **MQA everywhere**: 8 Q heads, **1 KV head**. Local layers head_dim 256; global layers head_dim **512** (`global_head_dim`), separate V kept on E2B (`attention_k_eq_v=false`; E4B+ unify K=V).
- **KV sharing**: `num_kv_shared_layers=20` — the last 20 layers (15–34) **reuse** KV from two producer layers: **layer 13 (sliding)** and **layer 14 (full)**, stashed in a `shared_kv_states` dict. Global consumers 19/24/29/34 all read layer 14's K/V; sliding consumers read layer 13's. Layers 4 and 9 (global, pre-sharing) keep their own full-length caches. KV-shared layers are compensated with **double-wide MLPs** (intermediate 6144→12,288 on layers 15–34) — freed KV params reinvested into MLP capacity.
- **PLE (per-layer embeddings)**: packed table `[262144, 35×256]`, per-layer gate + projection 1536↔256, mixed at scale 2^−0.5 (Gemma 3n heritage). Embedding scale √1536.
- **Vision tower**: 16-layer ViT, hidden 768, 12 heads, patch 16, pooling kernel 3, 2D axial RoPE θ=100, **280 soft tokens/image**, projection 768→1536, clipped linears (fp16/W8A8 friendly).
- **Audio tower**: 12-layer USM Conformer, hidden 1024, 8 heads, 4× conv subsampling (channels [128,32]), chunked left-context attention (chunk 12, left 13, right 0 → streaming), logit cap 50, proj 1024→1536, 40 ms/token. Paper: 305M params, 55% smaller than Gemma 3n's 680M, on-disk −78%.
- **Context**: 131,072 max positions (128K advertised).

### 5.4 Weight-level breakdown (arithmetic reconciled to 5,123,178,979 by autopsy)

```
V=262144, h=1536, L=35, heads=8, hd_local=256, hd_global=512, kv=1, inter=6144
embedding (tied)                    = 262144×1536                =   402.65M
PLE table  [V, 35×256]              = 262144×8960                = 2,348.81M   ← lookup-only
PLE model projection 1536→8960      =                             13.76M
attention: local/layer              = h·(8·256)+2·h·256+(8·256)·h = 7.08M ×28 = 198.18M
           global/layer             = h·(8·512)+2·h·512+(8·512)·h =14.16M ×7  =  99.09M
MLP: non-shared layers (15)         = 3·h·6144                  = 28.31M ×15  = 424.67M
     consumer layers (20, 2× wide)  = 3·h·12288                 = 56.62M ×20  = 1,132.46M
PLE per-layer gate+proj (35)        = 35×(1536×256×2)             =  27.53M
norms                                                            ≈    0.22M
TEXT STACK TOTAL                                                 ≈ 4.647B
vision tower (16 layers + patch/pos embeds)                      ≈ 0.15B
vision projector 768→1536                                        ≈   1.2M
audio tower (paper §2.2)                                         ≈ 0.305B
audio projector 1024→1536                                        ≈   1.6M
GRAND TOTAL                                                      ≈ 5.11B ✓ (autopsy: 5,123,178,979; HF metadata: 5,123,178,051)
```

Tensor naming (transformers `gemma4`): `model.language_model.embed_tokens.weight`, `model.language_model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`, `.{q,k,v}_norm.weight`, `.mlp.{gate,up,down}_proj.weight`, 4 layernorms/layer, `.per_layer_input_gate`, `.per_layer_projection`, `model.language_model.embed_tokens_per_layer.weight` [262144, 8960], `.per_layer_model_projection.weight`, `model.vision_tower.*`, `model.audio_tower.*`, `model.embed_vision.embedding_projection`, tied top-level `lm_head`.

**Checkpoint autopsy (measured from raw safetensors, not the config):** a full per-tensor pass over the 5 shards (2,011 tensors, 5,123,178,979 params; `autopsy.py`/`autopsy.json` in the companion repo folder) confirmed the arithmetic above and surfaced three facts the config does not tell you:

1. **The checkpoint ships ~18.9M dead parameters.** KV-shared layers 15–34 **do** contain `k_proj.weight`, `v_proj.weight`, `k_norm.weight` in the safetensors (60 tensors = 20 layers × 3; 12,587,008 params on the 16 sliding consumers + 6,293,504 on the 4 global consumers). transformers lists them in `_keys_to_ignore_on_load_unexpected` and **drops them at load** (2,011 file tensors vs 1,951 loaded). They are leftovers from training before KV-sharing was frozen. Surgical consequence: zero — they are unreachable — but they explain why on-disk size ≠ runtime size, and they are free weight-space if one ever wanted to *unshare* selected layers without retraining from scratch.
2. **Layer types, verified at runtime**: `full_attention` = layers **4, 9, 14, 19, 24, 29, 34** (7 layers); all others `sliding_attention`. (An early reading of the config suggested layer 9 was a *sliding* KV producer — runtime introspection of `self_attn.layer_type` says otherwise: 9 is full.)
3. **Vision and audio towers use `Gemma4ClippableLinear`**: every linear carries four scalar buffers `input_max/input_min/output_max/output_min` (e.g. audio `feed_forward1.ffw_layer_1.input_max = 12.8125`). These are the W8A8/fp16 activation-clipping bounds from the paper's on-device quantization work — they exist in bf16 too and cost nothing, but they are the hooks you'd clip through for int8.

Per-layer param census (text stack, from the autopsy; matches §5.4 arithmetic exactly):

| layers | type | attn params | MLP params | PLE params | norms+scalar | total/layer |
|---|---|---|---|---|---|---|
| 0–14 excl. 4,9,14 (12) | sliding, own-KV (L13 = shared producer) | 7,078,400 | 28,311,552 | 787,968 | 6,145 | 36,184,065 |
| 4, 9, 14 (3) | global, own-KV (L14 = shared producer) | 14,156,800 | 28,311,552 | 787,968 | 6,145 | 43,262,465 |
| 15–34 sliding (16) | sliding, KV-consumer (←L13) | 6,291,712† | 56,623,104 | 787,968 | 6,145 | 63,708,929 |
| 19, 24, 29, 34 (4) | global, KV-consumer (←L14) | 12,583,424† | 56,623,104 | 787,968 | 6,145 | 70,000,641 |

† consumer attention **as loaded** = q_proj + o_proj + q_norm only (the file-level k/v tensors are the dead 18.9M counted separately above). Each layer also carries 5 RMSNorms + a `layer_scalar` (6,145 params). Towers: `language_model` 2.765B (of which PLE table 2,348.8M — lookup-only), `vision_tower` 167.4M / 658 tensors, `audio_tower` 304.8M / 751 tensors.

### 5.5 Sliding-window attention — complete spec

- **Window**: 512 tokens, causal (left 512 incl. self, right 0), implemented as `sliding_window_overlay(512)` AND-composed with `OR(causal, blockwise_vision)` in `create_masks_for_vision_model` (`modeling_gemma4.py`).
- **Pattern**: `layer_types` = **4 sliding : 1 full**, full layers at indices 4, 9, 14, 19, 24, 29, 34; **last layer forced global** (paper: "4-to-1 for E2B, 5-to-1 for the rest"; transformers default is 5:1 via `sliding_window_pattern=6`).
- **Per-layer-type RoPE**:
  - Sliding layers: default RoPE, **θ=10,000**, full 256-dim rotation.
  - Global layers: **proportional RoPE (p-RoPE), θ=1,000,000, partial_rotary_factor=0.25** on 512-dim heads → rotated angles = int(0.25 × 512/2) = 64 freqs → **128 dims rotated, remaining 384 dims get inv_freq=0 ⇒ cos=1, sin=0 ⇒ NoPE identity**. Global layers are effectively 25% RoPE / 75% NoPE (cites Barbero et al. 2025); shrinks the positional footprint of the global KV cache and helps long context.
- **Vision bidirectionality**: Gemma 4 **disables bidirectional attention on global layers entirely** (unlike Gemma 3). On sliding layers, image tokens attend bidirectionally **within** the 512 window via a blockwise overlay driven by `mm_token_type_ids`.
- **KV-cache math** (verified against `modeling_gemma4.py:1171-1244`):
  - 28 sliding layers: capped at 512 tokens × 1 KV head × 256 dim × K+V × bf16 → **≈14 MiB total, constant regardless of sequence length**.
  - **KV sharing mechanism**: `is_kv_shared_layer = layer_idx ≥ 35−20 = 15`. Producer per type = **last non-shared layer of that type before 15** — **layer 13 (sliding)** and **layer 14 (full)** — which stash K/V into a `shared_kv_states[layer_type]` dict (NOT the rolling cache, since sliding caches drop >512). Consumers 15–34 read that dict (`sliding` consumers ← L13, `full` consumers ← L14) and have **no k/v weights of their own at load time** (their dead k/v tensors in the file are dropped — §5.4).
  - Full-length stores at inference: layers **4, 9** (global, own caches) + **13, 14** (shared dict) = **4 stores × seq_len × 1 head × 512 dim × K+V × bf16 = 16 KiB/token → ≈2 GiB @128k** (dominates the cache). Unshared it would be 7 global stores ≈ 3.5 GiB; paper claims "up to 37.5% global KV cache reduction" (with K=V unification on larger models — E2B's config has no `use_alternative_attention`, so V stays separate here; the `v_norm` has no learned scale either, `with_scale=False`, which is why no `v_norm.weight` appears in the autopsy).
- **Known weakness**: long-context retrieval is mediocre on E2B — **MRCR v2 8-needle @128k = 19.1% avg** (model card). Relevant caveat when extending context.

### 5.6 Gemma 4 paper extras (arXiv:2607.02770)

- E2B bf16 = 4.6 GB weights; mobile quant 0.8 GB; int8 KV @32k = +0.05 GB.
- Vision W8A8: 2× forward memory cut (400→200 MB), "44% reduction in on-device latency relative to Gemma 3n".
- **MTP speculative drafter** (§2.6): 4-layer block (3 local + 1 global), cross-attends to main-model KV, drafter dim 256, draft vocab 4,096; weights NOT in this checkpoint.
- 26B-A4B: MoE 8-of-128 + 1 shared — "runs almost as fast as a 4B-parameter model".
- Benchmarks (card): MMLU-Pro 60.0%, OmniDocBench 1.5 edit-distance 0.290 (vs Gemma 3 27B 0.365), LCB v6 44%, AIME'26 37.5%.

---

## 6. Head-to-head: sliding-window techniques

| Aspect | Gemma-4-E2B | Unlimited-OCR-2 |
|---|---|---|
| **Sparsity type** | **Static hybrid**: fixed 4:1 local:global layer pattern (position-only mask) | **Dynamic block-sparse (DASH)**: content-based top-k block selection per token + 128-token local window |
| **Where sparsity lives** | 28/35 layers are local-only (window 512); 7 global layers see everything | all 12 layers run the same sparse-read scheme at serve time |
| **Window / budget** | 512-token fixed sliding window (local layers); global layers unbounded | ring window 128 + top-12 blocks × 32 (=384 selected) ≈ **~512 fixed reference tokens** — deliberately the same order as Gemma-4's window |
| **Selection** | none (mask is position-only) | learned router: head_cls block summaries + 1.5-entmax + hard top-k annealed 64→12; image-only candidates, page-bucket constrained |
| **Data-dependence** | none | full: selection depends on token content |
| **Position encoding** | dual RoPE: local θ=10k full-rotary 256-dim; global **p-RoPE θ=1M, 25% rotary / 75% NoPE** on 512-dim heads | YaRN ×4 (32k→128k) θ=500k; **page2d RoPE**: (tok,page) axes split in frequency-pair space, page θ=256, page_mod=2 parity, resample freq mode |
| **KV cache** | 14 MiB sliding (constant) + 16 KiB/token full-length (4 stores: L4, L9 own caches + L13, L14 shared dict; 20 consumer layers share) | prefill pinned + 128-slot decode ring (training); at serve, DASH reads over full pinned cache; cache grows with doc length but **reads** are constant-size |
| **Cache cost model** | cache size bounded (sliding) + shared (global stores: 4×); **reads still O(N) on all 7 global layers** (sharing saves memory, not compute) | cache size O(N) pinned; **reads O(512)** per token on every layer |
| **Kernel** | stock SDPA/eager in transformers 5.x | fused Triton gather kernels + patched sglang; eager fallback in HF code |
| **Training requirement** | architecture is the training | DASH router trained end-to-end via entmax (7.87M params); R-SWA ring needs no training |
| **Long-context weakness** | MRCR 8-needle @128k = 19.1% (global layers carry everything, NoPE-heavy) | hundred-page benchmarks (MarathonDocBench SOTA) — designed for exactly this |

**The fundamental difference**: Gemma-4 decides *by position* which tokens a layer may read (static, learned into the weights); UO2 decides *by content* at inference time (dynamic router). Gemma-4's global layers are the long-range channel and they read **everything** every step — that is precisely the cost UO2's SR-SWA eliminates by reading only ~512 selected tokens per step. Conversely, Gemma-4's local 512 window on 28 layers has no analog cost problem at all. The transfer target is therefore narrow and clear: **make Gemma-4's 7 global layers content-selective at decode time**, without touching anything else.

## 7. Head-to-head: weights & core config

| | Gemma-4-E2B-it | Unlimited-OCR-2 |
|---|---|---|
| Total params | 5,123,178,979 on disk (1,951 of 2,011 tensors loaded; 18.9M dead KV-shard weights dropped) | 3,343,978,497 |
| **Active params/token** | ~2.7B (dense; PLE lookup adds reads not FLOPs) | **~0.73B (~22%)** — top-6/64 experts |
| LM type | dense (MoE fields exist but null) | **MoE 64 routed + 2 shared, 6/token** |
| Layers / hidden | 35 / 1536 | 12 / 1280 |
| Attention | MQA (1 KV head), hd 256 local / 512 global, QK-Norm | MHA 10 Q / 10 KV, v_head_dim 128, no MLA |
| MLP | GELU-tanh, inter 6144 (12,288 on 20 KV-shared layers) | SiLU, inter 6848 (dense L0) / expert inter 896 |
| Vocab | 262,144 tied | 129,280 untied (BOS 0/EOS 1/pad 2) |
| Norm/eps | RMSNorm pre+post, 1e-6, logit softcap 30 | RMSNorm, 1e-6 |
| Modalities | text+vision(280 tok/img)+audio | text+vision (273 tok/page, OCR-specialized) |
| Vision | 16L ViT-768, patch16, pool3, 2D axial RoPE, ClippableLinear W8A8 hooks | dual SAM-ViT-B + CLIP-L/14-224 (deeplip_b_l), 1024² tiles |
| Audio | 12L USM Conformer 1024 | — |
| Context | 131,072 | 131,072 (YaRN ×4) |
| Serving stock | transformers 5.x | custom sglang wheel + Triton kernels |

## 8. Transfer plan: applying UO2's good ideas to Gemma-4-E2B

Constraints adopted for this project (from the task): **surgical changes only, no full finetuning, no LoRA — training-free methods, eval-driven** (ParseBench gold rules as the quality gate). That constraint sorts UO2's seven levers (§4) into three buckets:

| UO2 lever | Training-free portable? | Verdict |
|---|---|---|
| 1. MoE sparsity | ✖ | training-time architecture; not retrofittable to a dense model without distillation. Excluded. |
| 2. Constant-size decode working set (top-k blocks ∪ local) | ✔ | **the core transfer** — replace O(N) reads on the 7 global layers with a fixed ~512-token keep-set at decode |
| 3. Learned image-aware selection (DASH router) | ✖ as-is | the router is trained end-to-end (7.87M params, entmax, k-anneal). **Substitute**: training-free Quest-style selection (§8.2) |
| 4. Fused Triton gather | ✔ (perf only) | optional kernel upgrade; the eager reference is correct first |
| 5. No-MLA | n/a | Gemma-4 already plain MQA |
| 6. page2d RoPE | ✖ | needs a `<PAGE>`-delimited training distribution; Gemma-4's p-RoPE NoPE dims already give length stability. Not portable zero-shot. |
| 7. High visual compression | ✔ (budget knob) | Gemma-4's soft-token budget is a *config*, not a weight — tunable per workload (§8.4) |

### 8.1 The target: Gemma-4's 7 global layers

Only layers 4, 9, 14, 19, 24, 29, 34 ever read beyond a 512-token window. They are MQA (1 KV head). Of these, 4, 9, 14 hold their own full-length KV and 19, 24, 29, 34 read layer 14's K/V from the shared dict (§5.5) — but all seven still **attend over the full sequence every decode step**: the KV sharing saves memory, not compute. With 1 KV head, a per-token gather is a [S,dim] slice — memory-bound and cheap. This makes Gemma-4 a *better* gather target than UO2's own 10-KV-head MHA.

### 8.2 SR-SWA analog without training: Quest-style block selection (IMPLEMENTED)

DASH's learned `head_cls` block summaries cannot exist without training. The standard training-free surrogate is **Quest** (Tang et al. 2024): per block of 32 keys, keep the elementwise min/max over the key dims; score a block against a query as `sum_d max(q_d·kmin_d, q_d·kmax_d)` — an upper bound on the best logit inside the block. Keep-set per decode step:

```
keep(q) = top-12 blocks by Quest score  ∪  last 128 tokens (local window)
        ≈ 384 + 128 = ~512 reference tokens        # = SR-SWA's budget exactly
```

Implementation (`sparse_patch.py` in the companion folder), ~120 lines, **zero weight changes**:

- **Hook point**: monkeypatch `transformers.models.gemma4.modeling_gemma4.eager_attention_forward`. In transformers 5.x the attention callable is resolved per-forward as `ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation, eager_attention_forward)` — the eager function is read from the **module namespace every call**, so patching it diverts every Gemma-4 attention module through one dispatcher.
- **Dispatch rule**: sparsify only when *all* of: `module.layer_type == "full_attention"` ∧ class is `Gemma4TextAttention` ∧ MQA (1 KV head) ∧ decode (`q_len ≤ 8`) ∧ `S ≥ min_len` ∧ not training ∧ **no non-trivial mask**. Everything else (prefill, sliding layers, vision, audio) calls the original eager function untouched.
- **Selection**: per block min/max → query scores → union over heads (`amax`) → top-12 → indices ∪ `arange(cover, S)` (covers the gap region between the last full block and the local tail, plus local 128) → gather K/V → eager attention over the keep-set with a duplicate-masked softmax. Prefill stays dense, so the KV cache is bit-identical to stock.

### 8.3 The mask gotcha — read this before touching transformers 5.x masks ⚠️

The first implementation registered a **custom attention implementation** (`"quest_sparse"`) and set `text_config._attn_implementation = "quest_sparse"`. That silently corrupts the model: `masking_utils.py:805` (transformers 5.22.1) returns **early with `and_masks=None`** for any `attn_implementation` not in the known mask-function registry → **no causal mask is built at all** → prefill becomes **bidirectional**, future tokens leak into the KV cache, and decode degenerates into whitespace loops within ~30 tokens.

Diagnosis path (toggle_check.py): prefill logits patched-vs-stock cosine was 0.9754 (must be 1.0) and teacher-forced top-1 agreement 2.1%. After switching to the monkeypatch approach (keep `attn_implementation="eager"`, patch only the Python function): prefill cosine **0.999999**, greedy 48-token continuation **100% token match**, teacher-forced top-1 agreement **100%**, logit cosine 0.9999 — and a dedicated activation test shows the sparse path firing on all 7 global layers every decode step (S≥512). **Rule: in transformers 5.x, never invent an `attn_implementation` string; patch the eager callable in the module namespace instead.**

### 8.4 The vision-token budget knob (S1) — UO2 lever 7

Gemma-4 emits 280 soft vision tokens/page (pool3 from a pre-pool grid). UO2 uses 273 tokens/page — nearly identical — but Gemma-4's budget is set in the *processor config*, not the weights, so it can be raised without training. For OCR-dense pages, doubling to 560 tokens (pool grid 2×) trades ~2× vision prefill cost for finer text detail. **Important**: with 560 tokens the pre-pool vision sequence is ~5,040 patches and the vision tower's *eager* attention materializes a ~582 MB fp32 matrix — on a shared 22 GB GPU that OOMs. Switching only the vision tower to sdpa post-load (`vision_tower.config._attn_implementation="sdpa"`, resolved per-forward in 5.x) removes the O(L²) spike while text stays eager for the patch. This switch is itself a free speed/memory win for any multi-page OCR batch.

### 8.5 R-SWA ring buffer — the other training-free piece

UO2's v1 lever (pin prefill, ring-overwrite the last 128 decode slots) is a pure KV-cache-layout change and needs no trained parts. For Gemma-4 it matters only at extreme generation lengths (decode output ≫ prompt), which OCR transcription rarely produces (output ≈ input). Deferred — but it is a drop-in `Cache` subclass if long *generation* ever becomes the bottleneck.

### 8.5b Two memory one-liners from the empirical work

- **`logits_to_keep=1` on prefill**: with a 262,144 vocab, a naive `model(input_ids=ids)` materializes `[1, L, 262144]` logits — **786 MB at L=1500**, OOM on a shared GPU. Passing `logits_to_keep=1` computes logits for the last position only. `generate()` does this internally; any manual prefill/decode loop must do it explicitly.
- **Vision tower → sdpa** (§8.4): kills the O(L²) vision prefill matrix; required for batched multi-page OCR.

### 8.6 What NOT to port

- **page2d RoPE** — Gemma-4's global layers already run 75% NoPE; bolting a 2-axis position scheme onto untrained weights scrambles the position geometry the model learned. Zero-shot port = certain regression.
- **DASH bias/entmax** — without the trained router there is no prior to bias with; Quest selection already covers the retrieval.
- **MoE upcycling** — the config's MoE fields are null; splitting dense GELU MLPs into experts without distillation destroys them.
- **Tied-embedding / vocab tricks** — 262k tied vs 129k untied is cosmetic for OCR.
- **MRCR caveat**: Gemma-4's global layers carry all long-range retrieval and the card reports 19.1% @128k. Block-sparsifying them trades exactly this channel — expect no regression at ParseBench lengths (~1k) and re-verify before trusting at 100k+.

### 8.7 The validated recipe, step by step

1. **Load stock** `google/gemma-4-E2B-it` with `attn_implementation="eager"` (audio tower rejects sdpa; eager everywhere is the safe default).
2. **Switch only the vision tower to sdpa** post-load: `model.vision_tower.config._attn_implementation = "sdpa"`. transformers 5.x resolves the attention callable per-forward from each submodule's config, so this is a one-line change that kills the O(L²) vision prefill matrices (152 MB/page at 280 tokens; 582 MB at 560) without touching text behavior.
3. **Apply the sparse patch**: `import sparse_patch; sparse_patch.apply(model, topk=12, block=32, local=128, min_len=512)`. Zero weight changes; prefill stays dense; only decode-step reads on the 7 global layers change (top-12 32-token blocks by Quest score ∪ 128 local ≈ 512 tokens = SR-SWA's budget).
4. **(quality, optional) Raise the vision budget**: `processor.image_processor.max_soft_tokens = 560` for OCR-dense pages (needs step 2 to not OOM).
5. **(robustness, optional) Generation guard**: `no_repeat_ngram_size=4` or `repetition_penalty≈1.1` — Gemma-4 greedy can loop on long transcriptions; UO2 ships the same guard (`_uocr_no_repeat_ngram`).
6. **Verify before trusting**: prefill-logit identity (cos = 1.0) + a 48-token greedy A/B + a ParseBench subset score. All three harnesses are in `code/`. Never set a custom `attn_implementation` string to do this — see §8.3.

## 9. Empirical: autopsy, surgery validation, ParseBench

Everything in this section was run on the actual checkpoints and data — not inferred from configs.

### 9.1 Method / environment

- Stock `google/gemma-4-E2B-it` (5 shards, BF16) served with `AutoModelForMultimodalLM` on transformers 5.22.1, `attn_implementation="eager"`, vision tower switched to sdpa post-load (§8.4).
- Eval data: ParseBench gold rules (`s3://ai-datasets-hub/OPEN_SOURCE_FINAL/Open_Source_LlamaIndex_ParseBench/`), `text_content.jsonl` = 141k rules over 508 PDFs; deterministic 24-doc subset (even-spaced), 590 in-subset rules, all `missing_specific_word`. Scorer: case-insensitive whitespace-normalized substring recall, macro (per-doc) + micro (per-rule) means.
- Prompt: "Please transcribe this document page into Markdown exactly as it appears…", greedy + `repetition_penalty=1.1` (a rep-free decode guard like UO2's `_uocr_no_repeat_ngram`), max 2048 new tokens, 150 DPI renders.

### 9.2 Weight autopsy (raw safetensors, 2,011 tensors)

Full per-tensor stats in `autopsy.json` (mean/std/absmax/zeros per tensor). Findings already folded into §5.4: exact 5,123,178,979-param reconciliation, the 18.9M dead KV-shard params, runtime-verified layer types (full = 4,9,14,19,24,29,34), ClippableLinear quantization hooks in vision+audio towers, and the double-wide MLPs (56.6M vs 28.3M) on exactly the 20 KV-shared consumer layers.

### 9.3 Surgery validation (CPU, float32, in-process A/B)

| check | result | pass criterion |
|---|---|---|
| prefill logits, patch off vs on | cosine **0.999999** | 1.0 (prefill must be untouched) |
| greedy 48-token continuation, off vs on | **100.0%** token match | high |
| teacher-forced last-48, off vs on | top-1 agreement **100.0%**, logit cos 0.9999 | ~95%+ |
| sparse path actually fires | **all 7 global layers**, every decode step at S≥512 (49 sparse calls / 8 steps × 7 layers; first step uses dense as mask lands) | fires on global only |

### 9.4 ParseBench results

_Config: 24-doc deterministic subset, greedy, rep-penalty 1.1, 150 DPI._

| run | macro doc score | micro rule score | mean tok/s | notes |
|---|---|---|---|---|
| baseline (stock, 280 vision tok) | 21.25 | 13.9 | 19.2 | 1/24 docs degenerated into a loop |
| **+ SR-SWA/Quest global sparse** | **21.86 (+0.61)** | **14.41 (+0.51)** | ~17.0 | top-12 blocks ∪ 128 local, min_len 512 |
| **+ vision budget 560 (S1)** | **33.91 (+12.66, +60%)** | **28.81 (+14.91, +107%)** | ~17.6 | 2× vision tokens, vision-sdpa |

**Sparse run vs baseline per-doc**: 19/24 docs identical, 3 improved (k10 +10.5, edited +3.9, 2cc +2.5), 1 regressed (ssrn-3007048 −2.2). Net: **no quality regression — the training-free SR-SWA analog preserves OCR quality** (differences are within greedy-decode trajectory noise). Tok/s is ~11% lower at these short lengths (~1k tokens) because block-selection overhead exceeds the gather saving below ~2k context; the crossover is predicted at a few thousand tokens (§9.5 — not reachable on this shared box).

**S1 run vs baseline per-doc**: 15 improved / 8 unchanged / 1 regressed. Biggest: agenda 25.8→87.1 (+61.3), rep 12.5→56.2 (+43.8), k10 15.8→47.4 (+31.6), 3colceo 2.7→32.4 (+29.7), eu 73.7→100 (+26.3); the only regression is wclb 100→81.8 (−18.2). The four handwriting/Punjabi/German-heavy docs stay 0 in every config (script/language limits, not budget). **Verdict: the vision-token budget is the single binding quality constraint for OCR-dense pages; doubling it is a pure-config change that buys +60% macro.** Cost: vision prefill ~2× (needs the vision-sdpa switch to not OOM) and ~2× vision KV per page. Score reports: `base_rp11.score.json`, `quest_rp11.score.json`, `s1_560.score.json`.

### 9.5 Long-context decode speed (eager vs sparse)

Two measurements, both on the shared GPU (co-tenant ~11.8 GB — the box could not fit >4-page multi-image eager prefills, so the clean curve is text-only):

**(a) Multi-image ParseBench pages, `generate()` end-to-end** (decode tok/s includes some prefill in the denominator):

| pages | input tokens | eager decode tok/s | sparse decode tok/s |
|---|---|---|---|
| 4 | 1,091 | 16.51 | 14.39 |

**(b) Text-only PURE decode** (prefill once, manual argmax loop — decode-step latency only), eager vs sparse vs cache length:

| context | eager tok/s | sparse tok/s | sparse speedup |
|---|---|---|---|
| 1,500 | 19.96 | 16.59 | 0.83× |
| 1,750 | 19.68 | 16.75 | 0.85× |
| 2,000–4,500 | — | — | OOM in **both** modes (see below) |

**Why the curve stops at 1,750**: prefill is dense and identical in both modes; its transient (bf16 activations + fp32 softmax workspace over 35 layers) is ~310–580 MB at 1.5–3k tokens, while this shared box had only ~540 MB–1 GB free after the 9.77 GiB model load (co-tenant ~11.8 GB, fluctuating). 2,000/2,500/2,750/3,000/4,500 all OOM'd at prefill on every attempt — a **hardware limit, not a property of the patch** (eager OOMs at exactly the same lengths).

At ~1.5–1.75k context the sparse path is ~15–17% slower (selection overhead > gather saving). The crossover is expected at a few thousand tokens — eager decode reads grow linearly (7 global layers × S), sparse reads are pinned at ~512 — but **could not be demonstrated on this shared box**; it needs a GPU with ~2–3 GB free above the model to prefill ≥3k tokens. UO2's claimed 7× at 100 pages (~27k tokens) is the asymptote of the same curve; nothing measured here contradicts it, and the mechanism (pin 7 layers' reads at ~512) is verified firing in §9.3.

## 10. Appendix

- **Verbatim UO2 config + v1↔v2 field diff + weight census + Triton/kernel details + arXiv summaries**: `uo2-notes.md` (this folder).
- **Verbatim Gemma-4 config + sliding-window spec + param arithmetic + paper details**: `gemma4-notes.md` (this folder).
- **Raw evidence**: `autopsy.json`, `base_rp11.score.json` / `quest_rp11.score.json` / `s1_560.score.json`, `decode_bench.jsonl` (every bench row incl. the OOMs).
- **References**: HF repos `baidu/Unlimited-OCR-2`, `baidu/Unlimited-OCR`, `google/gemma-4-E2B-it`; arXiv:2607.02770 (Gemma 4), arXiv:2606.23050 (Unlimited-OCR v1 / R-SWA), arXiv:2610.39308 (Unlimited-OCR-2 / SR-SWA), arXiv:2605.18753 (DASH), Quest (Tang et al. 2024); ParseBench (LlamaIndex).
- **Full code listings + replication guide**: §11 (everything needed to reproduce from scratch, inlined below).

## 11. Replication guide — from scratch to the validated numbers

This section makes the analysis self-contained: exact environment, exact commands, and the **complete source** of every harness. Nothing is left in an external file that you must trust.

### 11.1 What "surgery on gemma-4-E2B" means here (zero weight changes)

No weight tensor in `model.safetensors` is ever modified. Three **runtime, reversible** changes are applied after loading the stock model:

| # | Change | Where | Effect | Reversible by |
|---|---|---|---|---|
| A | `model.vision_tower.config._attn_implementation = "sdpa"` | vision tower only | kills the O(L²) vision prefill matrix (152→0 MB/page @280 tok; 582 MB @560) | set it back to `"eager"` |
| B | monkeypatch `modeling_gemma4.eager_attention_forward` → Quest dispatcher | text global layers, decode only | 7 global layers read ~512 selected tokens/step instead of all S | `sparse_patch.unapply()` |
| C | `processor.image_processor.max_soft_tokens = 560` (optional, quality) | processor config | 2× vision tokens/page for OCR-dense pages | set back to 280 |

Change B is the SR-SWA/DASH analog (§8.2). Changes A and C are UO2 lever 7 (visual budget). All three are **config/callable patches**, not finetuning and not LoRA — exactly the project's constraint.

### 11.2 Environment (verbatim from this box)

```bash
# Python venv with access to system site-packages (torch already present here)
python3 -m venv --system-site-packages /tmp/g4venv
/tmp/g4venv/bin/pip install -q "transformers==5.22.1" pymupdf pillow safetensors accelerate

# Transformers must be 5.x (tested 5.22.1). The mask gotcha (§8.3) is 5.x-specific;
# on 4.x the monkeypatch approach is still correct but the gotcha text won't apply.
```

Model + dataset:

```bash
# model (9.77 GiB, single safetensors shard)
hf download google/gemma-4-E2B-it --local-dir /tmp/models/gemma-4-E2B-it

# ParseBench gold answers (LlamaIndex). The harness expects:
#   $PB/docs/text/*.pdf            (the pages to parse)
#   $PB/text_content.jsonl         (gold rules; {"pdf","page","type","rule",...})
aws s3 sync s3://ai-datasets-hub/OPEN_SOURCE_FINAL/Open_Source_LlamaIndex_ParseBench/ /tmp/parsebench
PB=/tmp/parsebench
```

GPU note: the model needs **10,498 MiB** free minimum (9.77 GiB). On a shared box, wait for a free-enough GPU (the driver script §11.9 does this automatically).

### 11.3 The core patch — `sparse_patch.py` (COMPLETE, 123 lines)

This is the only non-trivial piece. Read the docstring — it encodes the §8.3 gotcha.

```python
#!/usr/bin/env python3
"""Training-free block-sparse attention for Gemma-4 global (full_attention) layers.

Surgical analog of Unlimited-OCR-2's SR-SWA/DASH decode path:
  keep-set = top-k KV blocks (content-selected)  ∪  local window (128 tokens)
DASH's learned entmax router cannot be ported without training, so selection uses
Quest-style block summaries (elementwise min/max of K per block, scored against
each query) — the standard training-free surrogate (Tang et al., Quest 2024).

Hook design (v2, mask-safe): DO NOT change config._attn_implementation.
transformers' masking_utils skips mask creation entirely for unknown attention
implementations (masking_utils.py:805), which silently makes prefill
bidirectional. Instead we monkeypatch `modeling_gemma4.eager_attention_forward`:
it is read from the module namespace at EVERY attention call as the `default`
for attn_implementation=="eager" (modeling_gemma4.py:952/1246), so our dispatcher
is invoked by every gemma4 attention module (text sliding+global, vision, audio).
We only divert Gemma4TextAttention global layers at decode; everything else calls
the original function. Masks, KV sharing, caches stay stock.

Only decode-time queries (q_len<=8) on full_attention layers with S>=min_len are
sparsified; prefill stays dense. No weights are modified.
"""
import os

import torch

_STATE = {}


def apply(model, mode="quest", topk=12, block=32, local=128, min_len=1024):
    from transformers.models.gemma4 import modeling_gemma4 as g4

    if getattr(g4, "_quest_patched", False):
        _STATE.update(topk=topk, block=block, local=local, min_len=min_len)
        return _STATE

    orig_eager = g4.eager_attention_forward
    _STATE.update(topk=topk, block=block, local=local, min_len=min_len,
                  sparse_calls=0, eager_calls=0, keys_seen=0)
    debug = bool(os.environ.get("QUEST_DEBUG"))

    def quest_dispatch(module, query, key, value, attention_mask=None,
                       dropout=0.0, scaling=None, softcap=None, **kwargs):
        q_len = query.shape[-2]
        S = key.shape[-2]
        use_sparse = (
            getattr(module, "layer_type", None) == "full_attention"
            and module.__class__.__name__ == "Gemma4TextAttention"
            and key.shape[1] == 1                      # MQA single KV head
            and q_len <= 8                             # decode-only; prefill stays dense
            and S >= _STATE["min_len"]
            and not module.training
            and not (attention_mask is not None
                     and bool((attention_mask != 0).any()))
        )
        if not use_sparse:
            _STATE["eager_calls"] += 1
            return orig_eager(module, query, key, value, attention_mask,
                              dropout=dropout, scaling=scaling, softcap=softcap, **kwargs)

        if debug and _STATE.get("dbg", 0) < 8:
            _STATE["dbg"] = _STATE.get("dbg", 0) + 1
            print(f"    [quest] sparse decode layer={module.layer_idx} q_len={q_len} S={S}", flush=True)
        _STATE["sparse_calls"] += 1
        _STATE["keys_seen"] += S
        if scaling is None:
            scaling = module.head_dim ** -0.5
        topk, block, local = _STATE["topk"], _STATE["block"], _STATE["local"]

        k = key[:, 0]                                     # [B, S, D]
        v = value[:, 0]                                   # [B, S, D]
        B, D = k.shape[0], k.shape[-1]
        nblocks = max((S - local) // block, 1)            # local tail always kept verbatim
        cover = nblocks * block

        kb = k[:, :cover].view(B, nblocks, block, D)
        kmin = kb.amin(2)                                 # [B, nb, D]
        kmax = kb.amax(2)                                 # [B, nb, D]
        q = query                                         # [B, H, L, D]
        # Quest score: sum_d max(q_d*kmin_d, q_d*kmax_d) per (query, block); union over heads
        qx = q.unsqueeze(3)                               # [B, H, L, 1, D]
        prod_max = torch.maximum(qx * kmin.unsqueeze(1).unsqueeze(1),
                                 qx * kmax.unsqueeze(1).unsqueeze(1))     # [B,H,L,nb,D]
        qs = prod_max.sum(-1).amax(1)                     # [B, L, nb]
        ksel = min(topk, nblocks)
        topb = qs.topk(ksel, dim=-1).indices              # [B, L, ksel]

        arange_b = torch.arange(block, device=q.device)
        idx = (topb.unsqueeze(-1) * block + arange_b).flatten(-2)          # [B,L,ksel*block]
        # include everything from `cover` onward (gap + local tail) so coverage is complete
        tidx = torch.arange(cover, S, device=q.device).view(1, 1, -1).expand(B, q_len, -1)
        idx = torch.cat([idx, tidx], dim=-1).sort(dim=-1).values           # [B,L,M]
        dup = torch.zeros_like(idx, dtype=torch.bool)
        dup[..., 1:] = idx[..., 1:] == idx[..., :-1]
        keep = ~dup

        kg = torch.gather(k.unsqueeze(1).expand(-1, q_len, -1, -1), 2,
                          idx.unsqueeze(-1).expand(-1, -1, -1, D))         # [B,L,M,D]
        vg = torch.gather(v.unsqueeze(1).expand(-1, q_len, -1, -1), 2,
                          idx.unsqueeze(-1).expand(-1, -1, -1, D))

        attn = torch.einsum("bhld,blmd->bhlm", q, kg) * scaling
        attn = attn.masked_fill(~keep.unsqueeze(1), float("-inf"))
        attn = torch.softmax(attn.float(), dim=-1).to(q.dtype)
        out = torch.einsum("bhlm,blmd->bhld", attn, vg)
        return out.transpose(1, 2).contiguous(), attn

    g4.eager_attention_forward = quest_dispatch
    g4._quest_patched = True
    g4._quest_orig_eager = orig_eager
    return _STATE


def unapply(model=None):
    from transformers.models.gemma4 import modeling_gemma4 as g4
    if getattr(g4, "_quest_patched", False):
        g4.eager_attention_forward = g4._quest_orig_eager
        g4._quest_patched = False


def stats():
    return dict(_STATE)
```

**Line-by-line, the decisions that matter:**

- `apply()` patches `g4.eager_attention_forward` **once** (`_quest_patched` guard); a second `apply()` only updates the hyperparameters. `unapply()` restores the original — this is what makes the change surgical and A/B-testable in one process.
- `use_sparse` is the safety net: **any** deviation (a real padding mask, training mode, a non-text module, prefill `q_len>8`, short context) falls through to `orig_eager` untouched. The `attention_mask != 0` check bails on any non-trivial mask so we never sparsify a masked/padded step.
- `nblocks = (S - local) // block`: the last 128 tokens are **always** kept verbatim and never enter block selection (they're the local window). `cover` is the end of the last complete block.
- `kmin/kmax` are the Quest block summaries (elementwise over the block dim). `qs = prod_max.sum(-1).amax(1)` scores each block per query and takes the **union over heads** (`amax` over the head axis) — one head finding a block important is enough to keep it.
- The keep-set is `top-12 blocks ∪ arange(cover, S)`: the second term covers both the partial-block gap and the 128-token local tail, so coverage is complete and causal order is preserved (`.sort()`), with duplicates masked to `-inf` before softmax.
- Softmax runs in fp32 (`.float()`) then casts back — matches stock eager numerics, which is why the validation cosines are ~1.0.

### 11.4 Eval harness — `eval_gemma4.py` (COMPLETE)

```python
#!/usr/bin/env python3
"""Eval harness: run gemma-4-E2B-it as a document parser on ParseBench PDFs.

Usage: python -I eval_gemma4.py --model-dir DIR --pb-dir DIR --split text \
         --n-docs N --out OUT.jsonl [--max-soft-tokens 280] [--dpi 150] \
         [--max-new-tokens 2048] [--sparse-global {none,quest}] [--sparse-topk 12] [--sparse-block 32]

Renders each PDF page with pymupdf, prompts for markdown transcription,
times generation, writes one JSONL row per page.
Untrusted-data safe: only reads files passed as args; run with python -I.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import pymupdf
from PIL import Image

PROMPT = ("Please transcribe this document page into Markdown exactly as it appears. "
          "Preserve the reading order, all text, tables, headers and footers. "
          "Output only the Markdown, no commentary.")


def render_pages(pdf_path, dpi):
    doc = pymupdf.open(pdf_path)
    imgs = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        imgs.append(img)
    doc.close()
    return imgs


def pick_docs(pb_dir, split, n_docs):
    """Deterministic even-spaced doc sample that has rules in the split."""
    rulefile = {"text": "text_content.jsonl", "table": "table.jsonl",
                "chart": "chart.jsonl", "layout": "layout.jsonl",
                "text_formatting": "text_formatting.jsonl"}[split]
    docs = sorted({json.loads(l)["pdf"] for l in open(os.path.join(pb_dir, rulefile))})
    if n_docs >= len(docs):
        return docs
    step = len(docs) / n_docs
    return [docs[int(i * step)] for i in range(n_docs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--pb-dir", default="/tmp/parsebench")
    ap.add_argument("--split", default="text")
    ap.add_argument("--n-docs", type=int, default=40)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-soft-tokens", type=int, default=280)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--sparse-global", default="none", choices=["none", "quest"])
    ap.add_argument("--sparse-topk", type=int, default=12)
    ap.add_argument("--sparse-block", type=int, default=32)
    ap.add_argument("--sparse-min-len", type=int, default=512)
    ap.add_argument("--sparse-local", type=int, default=128)
    ap.add_argument("--rep-penalty", type=float, default=1.0)
    ap.add_argument("--no-repeat-ngram", type=int, default=0)
    ap.add_argument("--vision-sdpa", action="store_true",
                    help="switch vision tower to sdpa attention post-load "
                         "(text stays eager; avoids O(L^2) vision prefill matrices)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoProcessor, AutoModelForMultimodalLM

    torch.manual_seed(args.seed)
    proc_kwargs = {}
    processor = AutoProcessor.from_pretrained(args.model_dir, **proc_kwargs)
    # vision-token budget knob (S1 surgery): try common attribute locations
    if args.max_soft_tokens != 280:
        for obj_name in ("image_processor",):
            obj = getattr(processor, obj_name, None)
            if obj is not None and hasattr(obj, "max_soft_tokens"):
                setattr(obj, "max_soft_tokens", args.max_soft_tokens)
                print(f"[cfg] image_processor.max_soft_tokens -> {args.max_soft_tokens}")
        if hasattr(processor, "max_soft_tokens"):
            processor.max_soft_tokens = args.max_soft_tokens
            print(f"[cfg] processor.max_soft_tokens -> {args.max_soft_tokens}")

    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="eager")
    model.eval()

    if args.vision_sdpa:
        vt = getattr(model, "vision_tower", None) or getattr(
            getattr(model, "model", None), "vision_tower", None)
        if vt is not None:
            vt.config._attn_implementation = "sdpa"
            print("[cfg] vision_tower._attn_implementation -> sdpa")
        else:
            print("[warn] vision_tower not found; --vision-sdpa ignored")

    if args.sparse_global != "none":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import sparse_patch
        sparse_patch.apply(model, mode=args.sparse_global, topk=args.sparse_topk,
                           block=args.sparse_block, local=args.sparse_local,
                           min_len=args.sparse_min_len)
        print(f"[cfg] sparse-global={args.sparse_global} topk={args.sparse_topk} "
              f"block={args.sparse_block} local={args.sparse_local} min_len={args.sparse_min_len}")

    docs = pick_docs(args.pb_dir, args.split, args.n_docs)
    print(f"[cfg] {len(docs)} docs, split={args.split}")

    n_done = 0
    with open(args.out, "w") as out:
        for pdf_rel in docs:
            pdf_path = os.path.join(args.pb_dir, pdf_rel)
            if not os.path.exists(pdf_path):
                print(f"[skip] missing {pdf_path}")
                continue
            try:
                pages = render_pages(pdf_path, args.dpi)
            except Exception as e:
                print(f"[skip] render fail {pdf_rel}: {e}")
                continue
            for pno, img in enumerate(pages):
                messages = [{"role": "user", "content": [
                    {"type": "image"}, {"type": "text", "text": PROMPT}]}]
                try:
                    text = processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False)
                    inputs = processor(text=text, images=[img],
                                       return_tensors="pt").to(0, torch.bfloat16)
                except Exception as e:
                    print(f"[skip] process fail {pdf_rel} p{pno}: {e}")
                    continue
                t0 = time.perf_counter()
                gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=False)
                if args.rep_penalty != 1.0:
                    gen_kwargs["repetition_penalty"] = args.rep_penalty
                if args.no_repeat_ngram > 0:
                    gen_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram
                with torch.inference_mode():
                    gen = model.generate(**inputs, **gen_kwargs)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                n_in = inputs["input_ids"].shape[1]
                n_new = gen.shape[1] - n_in
                output = processor.batch_decode(
                    gen[:, n_in:], skip_special_tokens=True)[0]
                rec = dict(pdf=pdf_rel, page=pno, output=output,
                           gen_seconds=round(dt, 3), new_tokens=int(n_new),
                           input_tokens=int(n_in),
                           tok_per_s=round(n_new / max(dt, 1e-6), 2))
                out.write(json.dumps(rec) + "\n")
                out.flush()
                n_done += 1
                print(f"[{n_done}] {pdf_rel} p{pno}: {n_new} tok in {dt:.1f}s "
                      f"({rec['tok_per_s']} tok/s)", flush=True)
    print(f"DONE {n_done} pages -> {args.out}")


if __name__ == "__main__":
    main()
```

### 11.5 Gold-rule scorer — `score.py` (COMPLETE)

Covers the three dominant rule types (97% of the text_content rule mass). Scores are 0–100.

```python
#!/usr/bin/env python3
"""Score ParseBench parser outputs against gold rules.

Usage: python -I score.py --pb-dir DIR --split text --pred PRED.jsonl [--report OUT.json]

Covers the three dominant rule types (97% of text_content rules):
  missing_specific_word      -> fraction of gold words present
  missing_specific_sentence  -> fraction of gold sentences present
  order                      -> 1 if both present and in correct order else 0
Reports per-doc macro mean and per-rule micro mean.
"""
import argparse
import json
import os
import re
from collections import defaultdict


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def score_rule(rule, doc_text):
    t = rule["type"]
    r = json.loads(rule["rule"]) if isinstance(rule["rule"], str) else rule["rule"]
    dt = norm(doc_text)
    if t == "missing_specific_word":
        bag = r.get("bag_of_word", {})
        if not bag:
            return None
        hit = sum(1 for w in bag if norm(w) in dt)
        return hit / len(bag)
    if t == "missing_specific_sentence":
        bag = r.get("bag_of_sentence", {})
        if not bag:
            return None
        hit = sum(1 for s in bag if norm(s) in dt)
        return hit / len(bag)
    if t == "order":
        b, a = norm(r.get("before", "")), norm(r.get("after", ""))
        if not b or not a:
            return None
        ib, ia = dt.find(b), dt.find(a)
        if ib == -1 or ia == -1:
            return 0.0
        return 1.0 if ib < ia else 0.0
    return None  # unsupported type skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pb-dir", default="/tmp/parsebench")
    ap.add_argument("--split", default="text")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    rulefile = {"text": "text_content.jsonl", "table": "table.jsonl",
                "chart": "chart.jsonl", "layout": "layout.jsonl",
                "text_formatting": "text_formatting.jsonl"}[args.split]

    # gather predicted docs
    pages = defaultdict(list)
    with open(args.pred) as f:
        for line in f:
            d = json.loads(line)
            pages[d["pdf"]].append((d["page"], d["output"]))
    doc_text = {pdf: "\n\n".join(t for _, t in sorted(ps)) for pdf, ps in pages.items()}

    # gather rules for those docs
    rules_by_doc = defaultdict(list)
    skipped = 0
    with open(os.path.join(args.pb_dir, rulefile)) as f:
        for line in f:
            r = json.loads(line)
            if r["pdf"] in doc_text:
                if r["type"] in ("missing_specific_word", "missing_specific_sentence", "order"):
                    rules_by_doc[r["pdf"]].append(r)
                else:
                    skipped += 1

    per_doc = {}
    n_rules = 0
    s_rules = 0.0
    by_type = defaultdict(lambda: [0.0, 0])
    for pdf, rules in rules_by_doc.items():
        scores = [score_rule(r, doc_text[pdf]) for r in rules]
        scores = [s for s in scores if s is not None]
        if not scores:
            continue
        per_doc[pdf] = sum(scores) / len(scores)
        for r, s in zip(rules, scores):
            by_type[r["type"]][0] += s
            by_type[r["type"]][1] += 1
        s_rules += sum(scores)
        n_rules += len(scores)

    macro = sum(per_doc.values()) / max(len(per_doc), 1)
    micro = s_rules / max(n_rules, 1)
    report = dict(
        pred=args.pred, split=args.split,
        docs_scored=len(per_doc), rules_scored=n_rules, rules_skipped_unsupported=skipped,
        macro_doc_score=round(macro * 100, 2), micro_rule_score=round(micro * 100, 2),
        by_type={t: round(100 * v[0] / max(v[1], 1), 2) for t, v in by_type.items()},
        per_doc={k: round(v * 100, 2) for k, v in sorted(per_doc.items())},
    )
    print(json.dumps({k: v for k, v in report.items() if k != "per_doc"}, indent=1))
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
```

### 11.6 Validation — run these BEFORE trusting any eval number

Two harnesses. `toggle_check.py` proves the patch is numerically transparent (prefill untouched, greedy identical, teacher-forced agreement). `sparse_fires.py` proves the sparse path actually executes on the global layers.

**`toggle_check.py` (COMPLETE):**

```python
#!/usr/bin/env python3
"""In-process A/B: patch on/off on ONE loaded model (mask-safe monkeypatch version).

  [1] prefill logits in both modes (must be identical: prefill is eager fallback)
  [2] 48-token forced greedy continuation in both modes
  [3] teacher-forced per-position top-1 agreement + logit cos
Usage: python -I toggle_check.py --model-dir DIR
"""
import argparse
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForMultimodalLM
    import sparse_patch

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_dir, dtype=getattr(torch, args.dtype),
        device_map={"": args.device} if args.device.startswith("cuda") else None,
        attn_implementation="eager")
    model.eval()

    lore = ("The quick brown fox jumps over the lazy dog. "
            "In 1861, the city council approved the construction of a new railway bridge. "
            "Photosynthesis converts sunlight, water and carbon dioxide into oxygen and energy. ")
    ids = tok(lore * 90, return_tensors="pt").input_ids.to(model.device)
    print("context tokens:", ids.shape[1])

    state = sparse_patch.apply(model, topk=12, block=32, local=128, min_len=512)

    def prefill_logits():
        with torch.inference_mode():
            return model(input_ids=ids).logits[0, -1].float().cpu()

    def gen():
        with torch.inference_mode():
            return model.generate(ids, max_new_tokens=48, min_new_tokens=48,
                                  do_sample=False)[0, ids.shape[1]:].cpu()

    sparse_patch.unapply()
    lgA = prefill_logits()
    sparse_patch.apply(model, topk=12, block=32, local=128, min_len=512)
    lgB = prefill_logits()
    cos = torch.nn.functional.cosine_similarity(lgA, lgB, dim=0).item()
    print(f"[1] prefill logits off-vs-on cosine: {cos:.6f}  (expect 1.000000)")

    sparse_patch.unapply()
    gA = gen()
    sparse_patch.apply(model, topk=12, block=32, local=128, min_len=512)
    gB = gen()
    match = (gA == gB).float().mean().item()
    print(f"[2] greedy 48-token continuation token match: {match*100:.1f}%")
    print("    eager:", repr(tok.decode(gA[:24]))[:130])
    print("    quest:", repr(tok.decode(gB[:24]))[:130])

    full = torch.cat([ids.cpu(), gA.unsqueeze(0)], dim=1).to(model.device)
    sparse_patch.unapply()
    with torch.inference_mode():
        lgE = model(input_ids=full).logits[0, -48:].float().cpu()
    sparse_patch.apply(model, topk=12, block=32, local=128, min_len=512)
    with torch.inference_mode():
        lgQ = model(input_ids=full).logits[0, -48:].float().cpu()
    agree = (lgE.argmax(-1) == lgQ.argmax(-1)).float().mean().item()
    cosf = torch.nn.functional.cosine_similarity(lgE.flatten(), lgQ.flatten(), dim=0).item()
    print(f"[3] teacher-forced last-48: top-1 agreement {agree*100:.1f}%, logit cos {cosf:.4f}")
    print("[stats]", {k: v for k, v in state.items() if k != "keys_seen"},
          "keys_seen:", state["keys_seen"])


if __name__ == "__main__":
    main()
```

**`sparse_fires.py` (COMPLETE):**

```python
#!/usr/bin/env python3
"""One-shot proof that the quest sparse path actually FIRES at decode.

Single apply(), short context just over min_len, a few decode steps on CPU.
Prints QUEST_DEBUG lines from inside the dispatch plus final counters.
Usage: python -I sparse_fires.py --model-dir DIR
"""
import argparse
import os
import sys

os.environ["QUEST_DEBUG"] = "1"
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForMultimodalLM
    import sparse_patch

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_dir, dtype=torch.float32, device_map=None,
        attn_implementation="eager")
    model.eval()

    lore = ("The quick brown fox jumps over the lazy dog near the river bank. "
            "Invoice number 4821 totals $1,234.56 due on March 15 2026. ")
    ids = tok(lore * 12, return_tensors="pt").input_ids
    print("context tokens:", ids.shape[1])

    state = sparse_patch.apply(model, topk=12, block=32, local=128, min_len=512)
    with torch.inference_mode():
        gen = model.generate(ids, max_new_tokens=8, min_new_tokens=8,
                             do_sample=False)
    out = gen[0, ids.shape[1]:]
    print("decoded:", repr(tok.decode(out)))
    print("FINAL STATS:", {k: v for k, v in state.items()})
    layers = model.model.language_model.layers
    types = [(i, getattr(l.self_attn, "layer_type", "?")) for i, l in enumerate(layers)]
    full_idx = [i for i, t in types if t == "full_attention"]
    print("layer types:", [t for _, t in types])
    print("full_attention layers:", full_idx)
    print(f"expected sparse calls if firing: {len(full_idx)} global x 8 steps = {len(full_idx)*8}")


if __name__ == "__main__":
    main()
```

**Expected output** (what §9.3 reports): `[1] prefill cosine 0.999999`, `[2] token match 100.0%`, `[3] top-1 agreement 100.0%, logit cos 0.9999`, and `sparse_fires` prints `full_attention layers: [4, 9, 14, 19, 24, 29, 34]` with `sparse_calls = 49` (7 layers × 7 sparse steps; the first decode step uses dense as the mask lands).

### 11.7 Long-context / speed harnesses (COMPLETE)

**`stress_test.py`** — multi-image end-to-end (`generate()`), gives §9.5(a):

```python
#!/usr/bin/env python3
"""Long-context stress test: feed N page-images as ONE multi-image request and
measure prefill time + decode tok/s, eager vs quest_sparse.

Usage: python -I stress_test.py --model-dir DIR --pdf PATH --pages 4,8,16,32 \
         --mode {eager,quest} --max-new-tokens 256 --dpi 110
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import pymupdf
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--pdf", default="/tmp/parsebench/docs/text/text_dense__baoutou.pdf")
    ap.add_argument("--pages", default="4,8,16,32")
    ap.add_argument("--mode", default="eager", choices=["eager", "quest"])
    ap.add_argument("--sparse-topk", type=int, default=12)
    ap.add_argument("--sparse-block", type=int, default=32)
    ap.add_argument("--sparse-min-len", type=int, default=1024)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoProcessor, AutoModelForMultimodalLM

    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="eager")
    model.eval()
    vt = getattr(model, "vision_tower", None) or getattr(
        getattr(model, "model", None), "vision_tower", None)
    if vt is not None:
        vt.config._attn_implementation = "sdpa"
        print("[cfg] vision_tower -> sdpa (batched pages make eager O(L^2) OOM)")

    if args.mode == "quest":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import sparse_patch
        sparse_patch.apply(model, mode="quest", topk=args.sparse_topk,
                           block=args.sparse_block, min_len=args.sparse_min_len)

    import glob
    imgs = []
    if args.pdf == "MULTI":
        # pull first page of many distinct PDFs (ParseBench docs are 1-3 pages)
        paths = sorted(glob.glob("/tmp/parsebench/docs/text/*.pdf"))
        for p in paths:
            try:
                d = pymupdf.open(p)
                pix = d[0].get_pixmap(dpi=args.dpi)
                imgs.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
                d.close()
            except Exception:
                pass
            if len(imgs) >= 40:
                break
    else:
        d = pymupdf.open(args.pdf)
        for page in d:
            pix = page.get_pixmap(dpi=args.dpi)
            imgs.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
        d.close()
    n_avail = len(imgs)
    print(f"[cfg] {n_avail} page images available")

    results = []
    for n in [int(x) for x in args.pages.split(",")]:
        if n > n_avail:
            print(f"[skip] need {n} pages, have {n_avail}")
            continue
        sel = imgs[:n]
        content = []
        for _ in range(n):
            content.append({"type": "image"})
        content.append({"type": "text", "text":
                        "These are document pages. Write a one-paragraph summary of their combined content."})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, images=sel, return_tensors="pt").to(0, torch.bfloat16)
        n_in = inputs["input_ids"].shape[1]

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, repetition_penalty=1.1)
        torch.cuda.synchronize()
        dt_total = time.perf_counter() - t0

        # rough prefill/decode split via a 1-token warmup pass
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.inference_mode():
            model.generate(**inputs, max_new_tokens=1, do_sample=False)
        torch.cuda.synchronize()
        dt_prefill = time.perf_counter() - t1
        n_new = gen.shape[1] - n_in
        dt_decode = max(dt_total - dt_prefill, 1e-6)
        rec = dict(mode=args.mode, pages=n, input_tokens=int(n_in),
                   new_tokens=int(n_new), total_s=round(dt_total, 2),
                   prefill_s=round(dt_prefill, 2),
                   decode_tok_s=round((n_new - 1) / dt_decode, 2))
        results.append(rec)
        print(json.dumps(rec), flush=True)
        del gen, inputs
        torch.cuda.empty_cache()

    if args.out:
        with open(args.out, "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
    if args.mode == "quest":
        import sparse_patch
        print("[sparse stats]", sparse_patch.stats())


if __name__ == "__main__":
    main()
```

**`decode_bench.py`** — text-only PURE decode curve, gives §9.5(b). Note the `logits_to_keep=1` on the prefill (§8.5b) — without it the [1, L, 262144] logits OOM at L=1500.

```python
#!/usr/bin/env python3
"""Text-only PURE decode-speed benchmark vs cache length: eager vs quest sparse.

Prefills once (dense, both modes identical), then times a manual argmax decode
loop — so the number is decode-step latency only, no prefill mixed in.

Usage: python -I decode_bench.py --model-dir DIR --lengths 1500,3000,4500 --new-tokens 128
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LORE = ("The quarterly report details revenue of $4.2 million across seventeen "
        "regional offices, with the strongest growth in logistics (23.4%) and "
        "consulting (11.8%). Table 14 summarizes expenditure by department; "
        "see appendix C for the full reconciliation of accounts receivable. ")


def decode_loop(model, ids, n_new):
    with torch.inference_mode():
        # logits_to_keep=1: full prefill logits would be [1, L, 262144] (~786MB @1500) -> OOM
        out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        next_id = out.logits[:, -1].argmax(-1, keepdim=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_new):
            out = model(input_ids=next_id, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_id = out.logits[:, -1].argmax(-1, keepdim=True)
        torch.cuda.synchronize()
        return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--lengths", default="1500,3000,4500")
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--sparse-min-len", type=int, default=1024)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForMultimodalLM

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="eager")
    model.eval()
    import sparse_patch

    results = []
    for L in [int(x) for x in args.lengths.split(",")]:
        ids = tok(LORE * 400, return_tensors="pt").input_ids[:, :L].to(0)
        real_L = ids.shape[1]
        for mode in ("eager", "quest"):
            if mode == "quest":
                sparse_patch.apply(model, topk=12, block=32, local=128,
                                   min_len=args.sparse_min_len)
            else:
                sparse_patch.unapply()
            try:
                dt = decode_loop(model, ids, args.new_tokens)
                rec = dict(mode=mode, ctx=real_L, new_tokens=args.new_tokens,
                           decode_s=round(dt, 2),
                           decode_tok_s=round(args.new_tokens / dt, 2))
            except torch.OutOfMemoryError:
                rec = dict(mode=mode, ctx=real_L, error="OOM")
                torch.cuda.empty_cache()
            results.append(rec)
            print(json.dumps(rec), flush=True)
        del ids
        torch.cuda.empty_cache()
    sparse_patch.unapply()
    if args.out:
        with open(args.out, "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
```

**`compare_runs.py`** — per-doc A/B of two score reports:

```python
#!/usr/bin/env python3
"""Compare two ParseBench .score.json reports per-doc (from score.py --report).

Usage: python -I compare_runs.py A.score.json B.score.json
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    args = ap.parse_args()
    A = json.load(open(args.a))
    B = json.load(open(args.b))
    pa, pb = A["per_doc"], B["per_doc"]
    docs = sorted(set(pa) | set(pb))
    print(f"A = {args.a}  (macro {A['macro_doc_score']}, micro {A['micro_rule_score']})")
    print(f"B = {args.b}  (macro {B['macro_doc_score']}, micro {B['micro_rule_score']})")
    print("-" * 86)
    print("doc".ljust(56) + "A".rjust(8) + "B".rjust(8) + "B-A".rjust(9))
    print("-" * 86)
    rows = []
    for d in docs:
        va, vb = pa.get(d, 0.0), pb.get(d, 0.0)
        rows.append((d, va, vb, vb - va))
        print(d[-54:].ljust(56) + f"{va:7.2f}%{vb:7.2f}%{vb-va:+8.2f}")
    print("-" * 86)
    rows.sort(key=lambda r: r[3])
    print("biggest regressions:")
    for d, va, vb, dd in rows[:5]:
        print(f"  {d[-60:]}: {va:.1f} -> {vb:.1f} ({dd:+.1f})")
    print("biggest improvements:")
    for d, va, vb, dd in rows[-5:][::-1]:
        print(f"  {d[-60:]}: {va:.1f} -> {vb:.1f} ({dd:+.1f})")


if __name__ == "__main__":
    main()
```

**`autopsy.py`** — the §5.4/§9.2 weight census (CPU, mmap, no model-code imports):

```python
#!/usr/bin/env python3
"""Layer-level weight autopsy of google/gemma-4-E2B-it (single 10.2GB safetensors).

Usage: python -I autopsy.py <model_dir> <out_json>
CPU-only, mmap-backed, chunked stats. Untrusted-data safe: no imports from model dir.
"""
import json
import re
import sys
from collections import defaultdict

import torch
from safetensors import safe_open


def tensor_stats(handle, key, chunk_rows=8192):
    """Chunked stats for a tensor via get_slice (torch, bf16-safe)."""
    info = handle.get_slice(key)
    shape = info.get_shape()
    dtype = info.get_dtype()
    numel = 1
    for s in shape:
        numel *= s
    if len(shape) == 0:
        v = float(handle.get_tensor(key).to(torch.float64).ravel()[0])
        return dict(shape=shape, dtype=dtype, numel=1, mean=v, std=0.0,
                    absmax=abs(v), l2=abs(v), zeros=int(v == 0))
    n = 0
    s1 = torch.zeros((), dtype=torch.float64)
    s2 = torch.zeros((), dtype=torch.float64)
    absmax = 0.0
    zeros = 0
    if len(shape) == 1:
        arr = info[:].to(torch.float64)
        s1 += arr.sum()
        s2 += (arr * arr).sum()
        absmax = float(arr.abs().max())
        zeros += int((arr == 0).sum())
        n += arr.numel()
    else:
        rows = shape[0]
        for r0 in range(0, rows, chunk_rows):
            r1 = min(r0 + chunk_rows, rows)
            arr = info[r0:r1].to(torch.float64)
            s1 += arr.sum()
            s2 += (arr * arr).sum()
            m = float(arr.abs().max())
            if m > absmax:
                absmax = m
            zeros += int((arr == 0).sum())
            n += arr.numel()
    n = max(n, 1)
    mean = float(s1) / n
    var = max(float(s2) / n - mean * mean, 0.0)
    return dict(shape=shape, dtype=dtype, numel=numel, mean=mean, std=var ** 0.5,
                absmax=absmax, l2=float(s2) ** 0.5, zeros=zeros)


def main():
    model_dir, out_json = sys.argv[1], sys.argv[2]
    path = f"{model_dir}/model.safetensors"
    tensors = {}
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        for i, k in enumerate(keys):
            tensors[k] = tensor_stats(f, k)
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(keys)}] {k}", flush=True)

    # ---- aggregate structure ----
    layers = defaultdict(lambda: defaultdict(float))
    towers = defaultdict(float)
    tower_tensors = defaultdict(int)
    layer_re = re.compile(r"model\.language_model\.layers\.(\d+)\.")
    for k, st in tensors.items():
        m = layer_re.search(k)
        if m:
            li = int(m.group(1))
            sub = k[m.end():]
            grp = ("attn" if "self_attn" in sub else
                   "mlp" if sub.startswith("mlp") else
                   "norm" if "layernorm" in sub else
                   "ple" if "per_layer" in sub else "other")
            layers[li][grp] += st["numel"]
            layers[li]["total"] += st["numel"]
            layers[li][f"absmax:{grp}"] = max(layers[li].get(f"absmax:{grp}", 0), st["absmax"])
        else:
            top = k.split(".")[1] if k.startswith("model.") else k.split(".")[0]
            towers[top] += st["numel"]
            tower_tensors[top] += 1

    layer_table = []
    for li in sorted(layers):
        d = layers[li]
        layer_table.append(dict(
            layer=li,
            total=int(d["total"]),
            attn=int(d.get("attn", 0)),
            mlp=int(d.get("mlp", 0)),
            ple=int(d.get("ple", 0)),
            norm=int(d.get("norm", 0)),
            has_kv=d.get("attn", 0) > 0 and d["attn"] > 5_000_000,  # kv-less layers have smaller attn
            attn_absmax=round(d.get("absmax:attn", 0), 3),
            mlp_absmax=round(d.get("absmax:mlp", 0), 3),
        ))

    out = dict(
        num_tensors=len(tensors),
        total_params=sum(t["numel"] for t in tensors.values()),
        towers={k: dict(params=int(v), tensors=tower_tensors[k]) for k, v in sorted(towers.items())},
        layers=layer_table,
        tensors={k: dict(shape=v["shape"], dtype=v["dtype"], numel=v["numel"],
                         mean=round(v["mean"], 6), std=round(v["std"], 6),
                         absmax=round(v["absmax"], 4), zeros=v["zeros"])
                 for k, v in tensors.items()},
    )
    with open(out_json, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nWROTE {out_json}")
    print(f"tensors={out['num_tensors']} total_params={out['total_params']:,}")
    print("\nTOWERS:")
    for k, v in out["towers"].items():
        print(f"  {k:20s} {v['params']:>14,} params  ({v['tensors']} tensors)")
    print("\nPER-LAYER (first/last rows shown):")
    hdr = f"{'L':>3} {'total':>12} {'attn':>10} {'mlp':>11} {'ple':>8} {'attn|max':>9} {'mlp|max':>9}"
    print(hdr)
    for row in layer_table[:3] + layer_table[13:17] + layer_table[-3:]:
        print(f"{row['layer']:>3} {row['total']:>12,} {row['attn']:>10,} {row['mlp']:>11,} "
              f"{row['ple']:>8,} {row['attn_absmax']:>9} {row['mlp_absmax']:>9}")


if __name__ == "__main__":
    main()
```

### 11.8 The exact commands that produced the numbers in §9

```bash
PY=/tmp/g4venv/bin/python
M=/tmp/models/gemma-4-E2B-it
PB=/tmp/parsebench
W=/tmp/g4work   # scratch output dir

# --- 0. validate the patch first (do NOT skip) ---
$PY -I $W/toggle_check.py  --model-dir $M            # expect: cos 0.999999 / 100% / 100% / 0.9999
$PY -I $W/sparse_fires.py  --model-dir $M            # expect: full layers [4,9,14,19,24,29,34], sparse_calls=49

# --- 1. BASELINE (stock, 280 vision tok) -> macro 21.25 / micro 13.9 ---
$PY -I $W/eval_gemma4.py --model-dir $M --pb-dir $PB --split text \
    --n-docs 24 --max-new-tokens 2048 --rep-penalty 1.1 --vision-sdpa \
    --out $W/base_rp11.jsonl
$PY -I $W/score.py --pb-dir $PB --split text --pred $W/base_rp11.jsonl --report $W/base_rp11.score.json

# --- 2. + SR-SWA/Quest sparse -> macro 21.86 / micro 14.41 ---
$PY -I $W/eval_gemma4.py --model-dir $M --pb-dir $PB --split text \
    --n-docs 24 --max-new-tokens 2048 --rep-penalty 1.1 --vision-sdpa \
    --sparse-global quest --sparse-min-len 512 \
    --out $W/quest_rp11.jsonl
$PY -I $W/score.py --pb-dir $PB --split text --pred $W/quest_rp11.jsonl --report $W/quest_rp11.score.json

# --- 3. + vision budget 560 (S1, the big quality lever) -> macro 33.91 / micro 28.81 ---
$PY -I $W/eval_gemma4.py --model-dir $M --pb-dir $PB --split text \
    --n-docs 24 --max-new-tokens 2048 --rep-penalty 1.1 --vision-sdpa \
    --max-soft-tokens 560 \
    --out $W/s1_560.jsonl
$PY -I $W/score.py --pb-dir $PB --split text --pred $W/s1_560.jsonl --report $W/s1_560.score.json

# --- 4. per-doc diff (baseline vs quest, baseline vs S1) ---
$PY -I $W/compare_runs.py $W/base_rp11.score.json $W/quest_rp11.score.json
$PY -I $W/compare_runs.py $W/base_rp11.score.json $W/s1_560.score.json

# --- 5. speed: multi-image (§9.5a) and pure-decode curve (§9.5b) ---
$PY -I $W/stress_test.py  --model-dir $M --pdf MULTI --pages 4,8,16 --mode eager --out $W/stress.jsonl
$PY -I $W/stress_test.py  --model-dir $M --pdf MULTI --pages 4,8,16 --mode quest --sparse-min-len 1024 --out $W/stress.jsonl
$PY -I $W/decode_bench.py --model-dir $M --lengths 1500,1750 --new-tokens 128 --out $W/decode_bench.jsonl
```

All runs use greedy (`do_sample=False`) + `rep-penalty 1.1` + `--vision-sdpa`. Deterministic 24-doc subset via `pick_docs` (even-spaced over the sorted doc list), so any rerun lands on the same 24 docs.

### 11.9 The driver that survives a shared GPU box — `run_chain.sh` (COMPLETE)

Runs the eval chain sequentially, waiting for a GPU with ≥10,500 MiB free before each step (the model needs 10,498). This is how the numbers were produced on a box with a co-tenant using ~11.8 GB.

```bash
#!/bin/bash
# Sequential eval chain with vision-sdpa (kills the O(L^2) vision prefill spike).
# Waits for a GPU with >= NEED MiB free before each step.
set -u
PY=/tmp/g4venv/bin/python
M=/tmp/models/gemma-4-E2B-it
W=/tmp/g4work
PB=/tmp/parsebench
NEED=10500   # MiB free required (model 9.77GiB=10498MiB; vision-sdpa removes the big transient)

pick_gpu() {  # prints index of freest GPU
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -nr | head -1 | tr -d ' ' | cut -d, -f1
}
free_mib() {  # $1 = gpu idx -> free MiB
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F, -v i="$1" '$1+0==i {print $2+0}'
}
wait_gpu() {  # echoes a gpu idx with >= NEED free, waiting up to ~50 min
  local tries=0
  while [ $tries -lt 100 ]; do
    G=$(pick_gpu); F=$(free_mib "$G")
    if [ "$F" -ge "$NEED" ]; then echo "$G"; return 0; fi
    echo "[wait] freest GPU $G has ${F} MiB < ${NEED}; sleeping 30s" >&2
    sleep 30; tries=$((tries+1))
  done
  echo "$G"; return 1
}

echo "[chain] start $(date)"
G=$(wait_gpu); echo "[chain] quest eval on GPU $G"
CUDA_VISIBLE_DEVICES=$G $PY -I $W/eval_gemma4.py --model-dir $M --pb-dir $PB --split text \
  --n-docs 24 --max-new-tokens 2048 --rep-penalty 1.1 --vision-sdpa \
  --sparse-global quest --sparse-min-len 512 --out $W/quest_rp11.jsonl > $W/quest_rp11.log 2>&1
echo "[chain] quest eval exit $? $(date)"
$PY -I $W/score.py --pb-dir $PB --split text --pred $W/quest_rp11.jsonl \
  --report $W/quest_rp11.score.json > $W/quest_rp11.score.txt 2>&1
tail -12 $W/quest_rp11.score.txt

G=$(wait_gpu); echo "[chain] S1-560 eval on GPU $G"
CUDA_VISIBLE_DEVICES=$G $PY -I $W/eval_gemma4.py --model-dir $M --pb-dir $PB --split text \
  --n-docs 24 --max-new-tokens 2048 --rep-penalty 1.1 --vision-sdpa \
  --max-soft-tokens 560 --out $W/s1_560.jsonl > $W/s1_560.log 2>&1
echo "[chain] S1 eval exit $? $(date)"
$PY -I $W/score.py --pb-dir $PB --split text --pred $W/s1_560.jsonl \
  --report $W/s1_560.score.json > $W/s1_560.score.txt 2>&1
tail -12 $W/s1_560.score.txt

G=$(wait_gpu); echo "[chain] stress eager on GPU $G"
CUDA_VISIBLE_DEVICES=$G $PY -I $W/stress_test.py --model-dir $M --pdf MULTI \
  --pages 4,8,16 --mode eager --out $W/stress.jsonl > $W/stress_eager.log 2>&1
echo "[chain] stress eager exit $?"
G=$(wait_gpu); echo "[chain] stress quest on GPU $G"
CUDA_VISIBLE_DEVICES=$G $PY -I $W/stress_test.py --model-dir $M --pdf MULTI \
  --pages 4,8,16 --mode quest --sparse-min-len 1024 --out $W/stress.jsonl > $W/stress_quest.log 2>&1
echo "[chain] stress quest exit $?"
cat $W/stress.jsonl 2>/dev/null
echo "[chain] DONE $(date)"
```

### 11.10 Failure modes seen on this box, and the fix for each

| Symptom | Cause | Fix |
|---|---|---|
| prefill logits cos = 0.9754, decode degenerates to whitespace | custom `attn_implementation` string → no causal mask (§8.3) | monkeypatch `eager_attention_forward`, never a custom impl string |
| OOM on S1-560 eval (582 MB vision matrix) | vision tower eager over ~5,040 pre-pool patches | `--vision-sdpa` (change A) |
| OOM at text ctx ≥ 2,000 in **both** modes | prefill transient ~310–580 MB vs ~540 MB free on shared box | hardware limit; use `run_chain.sh` wait_gpu or a free GPU |
| OOM in `decode_bench` at L=1,500 | `[1, L, 262144]` prefill logits = 786 MB | `logits_to_keep=1` (§8.5b) |
| greedy decode loops on a long transcription | gemma-4 greedy repetition | `--rep-penalty 1.1` or `--no-repeat-ngram 4` |
| `sparse_calls = 0` in stats | `apply()` re-zeroes counters each call | it's a red herring; use `sparse_fires.py` for the true count |
