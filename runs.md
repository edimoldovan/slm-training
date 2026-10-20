# Omarchy SLM: runs

## Environment

- Mac: Apple M5 Max, 128 GB, Darwin 27.0.0. Project `/Users/ed/dev/local-ai/slm-training`, venv `.venv`.
- Python 3.13.2 (Homebrew). The plan names python3.12, which is not installed on the Mac.
- mlx 0.32.4, mlx-lm 0.32.2, sentencepiece 0.2.2, transformers 5.20.2, safetensors 0.8.0, numpy 2.5.4, protobuf 7.36.2.
- `mx.default_device()` = `Device(gpu, 0)`.
- llama.cpp: Homebrew build 10080 (fd41bf65a): llama-cli, llama-completion, llama-server, llama-quantize. The package ships no `convert_hf_to_gguf.py`.
- AMD Omarchy box: AMD Radeon 610M (Granite Ridge iGPU).

## Step log

- Step 1 corpus: waiting for the owner's source list and manual URL.
- Step 2 environment: venv and named packages installed; GPU device confirmed. Converter not resolved yet.
- Tokenizer: `add_dummy_prefix=False` added to the plan's settings (with the default, `<|q|>` encodes as two ids). HF export sets `add_prefix_space: false` for the same reason.
- Pending GGUF checks for step 7, once the converter is available: token types of ids 3–5 in the GGUF must be USER_DEFINED (4); `tokenizer.ggml.add_space_prefix` must be false; `llama-tokenize -m <gguf> -p "<|q|>How do I update Omarchy?<|a|>" --ids` must equal `[1] + sp.encode(prompt)`.

## Corpus stats

(filled by step 1)

## Tokenizer

(filled by step 3)

## Runs

| run | config (layers, hidden) | steps | lr | best val loss | best step | tokens/s | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |

## Level 4 results

(filled by step 6)

## Sign-off

| Check | Threshold | Mac | AMD |
| --- | --- | --- | --- |
| Pipeline tests | all pass |  | n/a |
| Best val loss | < 2.0 |  | n/a |
| Command match, test | ≥ 0.90 |  |  |
| Well-formed, test | ≥ 0.98 |  |  |
| Mac vs AMD gap | ≤ 0.03 |  |  |
| Owner spot check | owner's call | n/a |  |
