Yes—EmbeddingGemma 1 already has **task prompting**, so you can implement task-steered zero-shot routing today. It is not a switch that changes the network architecture, but a training-aligned input format: prepend the appropriate task instruction consistently to both the live state and your option/label prototypes. EmbeddingGemma 2 extends this approach to its own named prompt templates and adds multimodal inputs, but its 8,192-token input budget is fixed; you generally should not simply raise it at inference time. [huggingface](https://huggingface.co/google/embeddinggemma-300m)

## 1. Task steering in EmbeddingGemma 1

For the original `google/embeddinggemma-300m`, the model card explicitly recommends prepending task strings. Its documented query format is:

```text
task: {task description} | query: {content}
```

For a zero-shot classifier/decision router, use the documented classification task:

```text
task: classification | query: {state or candidate label text}
```

For example, if you route support events into a fixed action set, make your labels concrete descriptions of the *decision condition*, rather than bare category names:

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("google/embeddinggemma-300m")

def task_classification(text: str) -> str:
    return f"task: classification | query: {text}"

actions = {
    "reset_password": (
        "User cannot sign in because they forgot a password, "
        "lost access to credentials, or requests a password reset."
    ),
    "account_locked": (
        "User account is locked after failed authentication attempts "
        "and needs an unlock or identity-verification workflow."
    ),
    "billing_refund": (
        "User requests a refund, reports an unexpected charge, "
        "or disputes a payment."
    ),
    "human_escalation": (
        "The request contains ambiguity, a security concern, a legal issue, "
        "or lacks enough information for safe automated handling."
    ),
}

label_names = list(actions)
label_vectors = model.encode(
    [task_classification(actions[name]) for name in label_names],
    normalize_embeddings=True,
)

def route(state: str):
    state_vector = model.encode(
        task_classification(state),
        normalize_embeddings=True,
    )

    scores = state_vector @ label_vectors.T  # cosine similarity
    best_index = int(np.argmax(scores))

    return {
        "action": label_names[best_index],
        "confidence": float(scores[best_index]),
        "alternatives": sorted(
            zip(label_names, scores.tolist()),
            key=lambda x: x [huggingface](https://huggingface.co/google/embeddinggemma-300m),
            reverse=True,
        ),
    }

print(route(
    "I tried signing in several times while travelling. "
    "It now says my account has been locked."
))
```

EmbeddingGemma 1’s published prompt catalog also includes `search result`, `question answering`, `fact checking`, `clustering`, `sentence similarity`, and `code retrieval`. For your use case, `classification` is the appropriate starting point; do not use retrieval query and document prompts interchangeably when the actual objective is selecting among classes. [huggingface](https://huggingface.co/google/embeddinggemma-300m)

### Important design rule

Use the **same task family and formatting** for all embeddings that participate in a cosine comparison.

| Scenario | State/input encoding | Candidate encoding |
|---|---|---|
| Closed-set intent routing | `task: classification \| query: {state}` | `task: classification \| query: {rich label description}` |
| Find a relevant policy/runbook | `task: search result \| query: {incident}` | `title: {name} \| text: {policy/runbook}` |
| Match duplicate incidents | `task: sentence similarity \| query: {incident}` | `task: sentence similarity \| query: {prior incident}` |
| Retrieve code for a natural-language request | `task: code retrieval \| query: {request}` | Document-style code representation |

The first model’s prompt interface is therefore already a practical form of task steering. However, the model card does **not** establish that arbitrary long natural-language directives will reliably behave like an instruction-following LLM. Stick initially to the task phrases on which it was trained, then evaluate any custom task string on a held-out routing set. [huggingface](https://huggingface.co/google/embeddinggemma-300m)

### Make zero-shot routing safer

Cosine similarity always returns a “winner,” even if none of your actions is appropriate. A robust router needs abstention logic:

```python
def decide(state: str, accept_threshold=0.58, margin_threshold=0.06):
    result = route(state)
    ranked = result["alternatives"]
    best_name, best_score = ranked[0]
    runner_up_score = ranked [huggingface](https://huggingface.co/google/embeddinggemma-300m) if len(ranked) > 1 else -1.0

    if best_score < accept_threshold:
        return "human_escalation", "low_similarity"

    if best_score - runner_up_score < margin_threshold:
        return "human_escalation", "ambiguous_boundary"

    return best_name, "accepted"
```

Tune the absolute threshold and top-two margin on representative, labeled data. The actual values will vary substantially with label wording, class overlap, domain vocabulary, and whether you normalize embeddings.

Also include explicit competing categories such as “insufficient information,” “unsupported request,” and “security-sensitive issue.” This makes abstention a first-class semantic decision instead of hoping that an arbitrary business-action vector happens to be the least wrong answer.

## 2. Extending EmbeddingGemma 2 context

EmbeddingGemma 2 has a **fixed 8,192-token multimodal context window**. That budget is shared by text, images, video frames, and audio—not 8K text tokens plus separate media capacity. The text/code encoder is documented as having an 8,192-token window; media consumes fixed portions of the same window. [developers.googleblog](https://developers.googleblog.com/embeddinggemma-2-the-developer-guide/)

As documented, a single-modality input can hold approximately:

| Input type | Approximate maximum in one embedding call |
|---|---:|
| Text/code | 8,192 tokens |
| Images | 29 images |
| Video | 58 frames |
| Audio | 5.5 minutes |

The model’s default video processing is one frame per second, and audio should be 16 kHz mono. In a mixed item, text and media compete for the same context budget. [developers.googleblog](https://developers.googleblog.com/embeddinggemma-2-the-developer-guide/)

### What not to do

Do not assume that setting a larger `max_seq_length`, editing a tokenizer setting, or changing a configuration field converts the released model into a reliably longer-context encoder. Positional behavior and embedding quality outside the trained context regime are not guaranteed; the documented interface specifies 8,192 tokens. [developers.googleblog](https://developers.googleblog.com/embeddinggemma-2-the-developer-guide/)

### Practical ways to handle larger states

For decision engines, longer raw input is usually better handled as a **hierarchical routing pipeline**, not as one gigantic embedding.

1. **Chunk and embed**
   
   Split long logs, transcripts, source files, or documents into semantically coherent chunks. For logs, preserve event/session boundaries; for code, preserve function/class boundaries; for conversations, preserve turns or compact time windows.

2. **Embed chunks with provenance**
   
   Store each embedding with identifiers such as `case_id`, `timestamp range`, `service`, `source file`, and `chunk type`.

3. **Retrieve/select the most decision-relevant chunks**
   
   Compare a current-state query against the chunks, retain the top \(k\), and only then form a compact decision representation.

4. **Use a one-vector decision state**
   
   If you require a literal one-forward-pass *final decision*, precompute or externally select evidence, then place only the salient snippets plus the current state within the 8K budget. The final state can be compared once to pre-embedded action prototypes.

5. **Use temporal/windowed aggregation**
   
   For continuously growing logs and audio, embed fixed windows—such as 30-second audio segments or 200-event log windows—and pool or vote at the application layer. This is not a native longer context, but it scales indefinitely and gives you an audit trail.

For raw 50,000-token incident logs, an effective architecture is typically:

```text
Raw stream
→ chunk embeddings
→ relevance / anomaly shortlist
→ compact evidence packet under 8K
→ one EmbeddingGemma 2 decision embedding
→ cosine similarity against action vectors
→ threshold + margin + policy guardrail
```

That preserves the advantages of an embedding-based router while avoiding truncation of the evidence you need.

## 3. One-forward-pass multimodal decisions

The key distinction is this: EmbeddingGemma 2 can generate **one embedding from an interleaved state** containing text plus media. You then compare that state embedding with precomputed text action/intent embeddings using cosine similarity. It is one forward pass for the *state decision*, assuming your action vectors were precomputed earlier.

The model maps text, code, images, video, and audio into a shared 768-dimensional space, and Sentence Transformers supports interleaved text/media using `<|image|>`, `<|video|>`, and `<|audio|>` markers. It also supports task prompts for text such as `prompt_name="SearchQuery"` and `prompt_name="Document"`.  [developers.googleblog](https://developers.googleblog.com/embeddinggemma-2-the-developer-guide/)

### A realistic dispatch pattern

Pre-embed candidate actions in the appropriate text task space:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google/embeddinggemma-2")

action_texts = [
    "Route to urgent safety review when the scene or audio suggests fire, smoke, alarm, collision, injury, or immediate danger.",
    "Create a facilities work order for a non-urgent physical defect such as water leakage, damaged equipment, or blocked access.",
    "Open an IT incident when a screen shows a software failure, access issue, service error, or application outage.",
    "Request human review when the evidence is unclear, conflicting, or does not match an approved workflow.",
]

action_embeddings = model.encode(
    action_texts,
    prompt_name="Document",
    normalize_embeddings=True,
)
```

Then produce a **single joint state representation**:

```python
state = {
    "text": (
        "Employee report: <|image|> "
        "A loud repeating alarm was recorded: <|audio|>"
    ),
    "image": "warehouse_camera_frame.jpg",
    "audio": "warehouse_audio.wav",
}

state_embedding = model.encode(
    state,
    normalize_embeddings=True,
)

scores = model.similarity(state_embedding, action_embeddings)[0]
```

The exact prompt behavior for interleaved media should follow the model’s current model-card/API guidance. Google’s developer guide specifically shows media dictionaries without a prompt and demonstrates adding `<|image|>`, `<|video|>`, or `<|audio|>` within text to define placement in a combined input.  [developers.googleblog](https://developers.googleblog.com/embeddinggemma-2-the-developer-guide/)

## Concrete use cases

| Decision model | One joint state | Action vectors | Why vision/audio changes the decision |
|---|---|---|---|
| Warehouse safety triage | Operator text, CCTV frame, alarm audio | Urgent safety response; equipment maintenance; false alarm review | Text may say “machine is making noise,” while smoke/flame cues in the image and a siren pattern in audio can move the state toward urgent response |
| Facilities maintenance | Tenant description, photo of damage, 10-second sound clip | Plumbing dispatch; electrical dispatch; HVAC dispatch; routine work order | A photo can differentiate ceiling water damage from a cracked panel; audio can distinguish a rattling HVAC unit from a leaking pipe report |
| Retail loss-prevention queue | Short guard note, camera frame, store audio | Escalate to trained staff; customer-service intervention; technical camera issue; no action | The system can use visual evidence of a blocked exit or crowding and audio indicators such as a broken alarm, rather than relying only on vague notes |
| Contact-center escalation | Transcript text, original customer voice clip, optional screenshot | Billing workflow; account recovery; accessibility support; abuse/threat escalation; human specialist | Text transcription loses prosody. Audio may convey crying, panic, shouting, repeated alarm tones, or interrupted speech that should influence a cautious escalation policy |
| Field-service routing | Technician note, equipment photo, machine recording | Dispatch refrigeration technician; electrical technician; mechanical technician; stop-use/safety inspection | A compressor sound and a photographed fault code together provide a more specific state than either a text summary or an image alone |
| Insurance intake prioritization | Claim narrative, accident photos, short voice description | Emergency mitigation; routine appraisal; fraud-review queue; missing-information request | Photo evidence can identify apparent water intrusion or collision damage; the voice note may add urgency cues and facts absent from the typed form |
| Accessibility-aware desktop assistant | UI screenshot, user voice command, selected UI metadata | Explain screen; click/activate action; dictate reply; ask clarification; escalate sensitive action | The state combines what is visually on-screen with the user’s spoken intent, which is much stronger than OCR or ASR alone for deciding the next assistive action |
| Industrial quality gate | Image of assembly, microphone recording, batch/config text | Pass; hold for inspection; recalibrate tool; stop line | Some defects are visible, while others are acoustic—bearing chatter, vibration, or an abnormal motor tone—so a joint state better supports the decision |
| Local video review | Text query/context plus sampled video frames and audio window | Flag potential incident; retain clip; ignore as non-event; send for analyst review | A frame may show people running, while audio distinguishes a celebratory crowd from an alarm, impact, or distress event |
| Document intake | Email text, photographed/scanned page, caller’s audio explanation | Extract-and-file; request a clearer scan; route to compliance; route to support | Visual layout, handwriting, tables, stamps, and signatures may matter; the spoken explanation can disambiguate why the document was submitted |

### Example: maintenance dispatch

Suppose the state is:

```text
Text: “The server-room cooling unit is making a loud metallic noise.”

Image: photo of a wall-mounted HVAC unit, with no visible water leak.

Audio: 12-second clip with rhythmic metallic rattling.
```

Your candidate decisions might be:

```text
- Emergency electrical or fire hazard inspection
- HVAC mechanical service dispatch
- Plumbing/water-leak work order
- Request additional evidence
```

A text-only system may confuse “metallic noise” with electrical risk or a generic facilities request. A joint image-plus-audio-plus-text state has additional evidence that can align more closely with “HVAC mechanical service dispatch.” You should still place this behind a safety policy: high-risk outcomes must use conservative thresholds, have a human fallback, and never rely solely on an embedding score.

## Operational cautions

- Embedding similarity is a **retrieval/routing signal**, not formal logical reasoning, calibrated probability, or a guarantee that the chosen option is correct.
- Precompute the action vectors, version them with your prompt and model version, and re-embed them if you change the prompt template, embedding dimension, or model.
- Normalize all vectors if you interpret dot products as cosine similarity.
- Keep states and action vectors at the same MRL dimension. EmbeddingGemma 2 supports 768, 512, 256, and 128 dimensions; Google recommends 768/512 when multimodal recall matters, while 128 dimensions have a larger quality loss for image, video, and speech retrieval. [developers.googleblog](https://developers.googleblog.com/embeddinggemma-2-the-developer-guide/)
- Use 256d cautiously for cost-sensitive multimodal routing: Google reports about 95% of full-quality retrieval for image, video, and speech at that size, but validate the claim against your specific categories, sensors, compression, languages, and error costs. [developers.googleblog](https://developers.googleblog.com/embeddinggemma-2-the-developer-guide/)
- For high-stakes workflows—safety, health, finance, employment, law enforcement, access control, or emergency response—treat the embedder as one feature in a controlled decision system, with deterministic rules, uncertainty escalation, logging, and human review.
