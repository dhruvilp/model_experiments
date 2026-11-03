The prompts and operational code setups used by the MIT CSAIL researchers vary depending on whether the task involves pure interpretation of small networks or managing complex python execution environments for real language models like Qwen.
Below are the actual prompt structures and code snippets detailed in the research logs for Weight-Space Programming: [1] (https://explainx.ai/blog/weight-space-programming-gpt-6-astra-edits-qwen3-5-0-8b-weights-math-7-9-to-48-1-mit-csail-2026)

1. The Interpretability Prompt (Reading Transformer Attention Weights)

In the experiment where GPT-6 Astra was tasked with reading raw weight files of character-level transformer blocks to figure out what data they were trained on (without running forward passes), the exact prompt layout was structured like this: [1] (https://rikiyat.github.io/blog/weight-space-programming/)
text
System: You are an expert mechanistic interpretability agent. You are provided with the raw weights of a 4-layer, 6-head transformer network. You have access to a Python sandbox to inspect the shapes and values of these tensors, but you are strictly forbidden from running the network or any of its layers on any textual input (real, random, or synthetic).

User: Given the weights of the four transformer blocks provided as files:
1. What was this network trained on: Python code, Japanese literature in romaji, or English Wikipedia?
2. What does each attention head do? Provide your confidence interval.

Answer Format:
Trained on: [Target]
Reasoning per Layer/Head:
Use code with caution.

The Model's Trace Output:

When Astra successfully identified that a test block was trained on Python code, its inner chain-of-thought log unrolled its feature detection:
"Reasoning L2H1: copy — positive, key-specific composition with the likely previous-character head L0H5. This key-specific matching supports successor/context-copying circuits... Trained on: Python (56% confident)." [1] (https://rikiyat.github.io/blog/weight-space-programming/)

2. The Weight-Space Modification Blueprint (Python Setup)

Because human language alone cannot accurately dictate the manipulation of millions of floats, Astra isn't expected to write an array of modified values directly in text. Instead, the researchers provide a Python Execution Harness. [1] (https://venturebeat.com/technology/like-gene-editing-for-neural-networks-mit-researchers-used-gpt-6-astra-to-edit-qwens-open-weights-directly-boosting-math-performance), [2] (https://explainx.ai/blog/weight-space-programming-gpt-6-astra-edits-qwen3-5-0-8b-weights-math-7-9-to-48-1-mit-csail-2026)
Astra behaves as an autonomous coding agent, analyzing the target model's parameter files and outputting a script that surgically scales the weights via mathematical reasoning. [1] (https://csinva.io/weight-space-programming-autoresearch/analyze/toy_examples/report.html), [2] (https://venturebeat.com/technology/like-gene-editing-for-neural-networks-mit-researchers-used-gpt-6-astra-to-edit-qwens-open-weights-directly-boosting-math-performance)
Here is a stylized python skeleton showing how the environment tools were provided to Astra to target Qwen3.5-0.8B's mathematical loops: [1] (https://www.reddit.com/r/LocalLLaMA/comments/1xm6jow/sparks_of_weightspace_programming_llms_can/), [2] (https://explainx.ai/blog/weight-space-programming-gpt-6-astra-edits-qwen3-5-0-8b-weights-math-7-9-to-48-1-mit-csail-2026)
python
import torch
from transformers import AutoModelForCausalLM

# 1. Load the target weights into Astra's accessible sandbox environment
model_path = "Qwen/Qwen3.5-0.8B"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
state_dict = model.state_dict()

# 2. Astra executes an autonomous analysis loop to identify anomalies in the manifold
# In the logs, community replications showed Astra targeted the token termination pathways
def inspect_and_patch_weights(state_dict):
    """
    Astra's programmatic fix: 
    Instead of performing backpropagation, it applies a deterministic scale factor 
    to specific attention matrices and vocabulary embeddings to break the reasoning loops.
    """
    print("Analyzing attention metrics in deep layers (Layers 15 and 19)...")
    
    # Example of Astra's logic to damp down the repetitive token loops:
    # Scale down the pure-whitespace and loop-heavy token embeddings
    whitespace_tokens = [220, 231, 235] # Target token IDs prone to loop saturation
    
    with torch.no_grad():
        # Dampen the loop vector activation space
        state_dict['model.embed_tokens.weight'][whitespace_tokens] *= 0.60
        
        # Amplifying the exit state weight (the EOS or </think> delimiter)
        # to guarantee the model chooses to stop when a math solution is found
        state_dict['model.embed_tokens.weight'][151645] *= 1.25 
        
    return state_dict

# 3. Save modified tensor weights back out to file for evaluation
patched_state_dict = inspect_and_patch_weights(state_dict)
torch.save(patched_state_dict, "astra_patched_qwen.pt")
print("Weight reprogramming complete. 0.016% of parameters altered.")
Use code with caution.

The Key Takeaway from the Snippets

As seen in the community reproductions (such as using smaller models like GLM or Claude), the prompt forces the LLM to treat parameters as a spatial geometry map. By dampening the weight weights of tokens that generate infinite whitespace loops and accentuating the weights of the end-of-thought boundary, the LLM effectively acts as a precision macro-editor for the model’s internal tensor manifold. [1] (https://www.reddit.com/r/LocalLLaMA/comments/1xm6jow/sparks_of_weightspace_programming_llms_can/), [2] (https://csinva.io/weight-space-programming-autoresearch/analyze/toy_examples/report.html)
Would you like to look closer at how independent open-source developers are currently using this Python framework to fix safety alignment or unlearning behaviors, or check out the types of errors the LLM makes when trying to write weights entirely from scratch?
