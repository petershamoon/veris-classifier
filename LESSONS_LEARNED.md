# VERIS Classifier - Lessons Learned & Notes

## Assumptions
- VCDB incidents with `summary` fields produce better synthetic descriptions since GPT has context to expand from (9,518 of 10,037 have summaries)
- gpt-4o-mini is sufficient quality for generating synthetic incident descriptions - the descriptions are short (2-4 sentences) and the task is straightforward (JSON -> English)
- gpt-4o is better for the actual classification task (English -> VERIS JSON) since it requires understanding nuanced security concepts and mapping to 300+ enum values
- ZeroGPU on HF Spaces requires HF Pro ($9/month) but is worth it for free A10G burst inference — cold starts of 30-60s are acceptable for a portfolio project
- VCDB data has known bias toward healthcare (HIPAA mandatory disclosure) - this will be reflected in the training data

## OpenAI Rate Limit Lessons

### Per-Minute vs Per-Day Limits
- **gpt-4o-mini standard tier**: 200K TPM, 500 RPM, **10K RPD (requests per day)**
- At 30 concurrency we burned through the daily limit in ~6 batches (~1,500 requests)
- The overnight run got stuck: the process was alive but sleeping in retry loops for 24 hours (only 2s CPU time used!)
- **Fix**: Detect "per day" in 429 error messages and sleep much longer (10 min) vs per-minute limits (3-6s exponential backoff)
- **Also**: Increase max_retries from 3 to 10 to survive the long wait for daily limit reset

### Semaphore Placement
- Put `async with semaphore` **OUTSIDE** the retry loop so retries hold their slot instead of competing with new requests
- Wrong: semaphore inside retry = retries release and re-acquire, causing thundering herd
- Right: semaphore outside retry = retries keep their slot, smooth backoff

### Concurrency Tuning
- 30 concurrent requests: maxed TPM/RPM immediately, 20% loss to 429 errors
- 15 concurrent: better but still hitting per-minute limits frequently
- 10 concurrent: sweet spot for overnight runs with rate limit headroom
- **Key insight**: Lower concurrency with fewer errors is faster than high concurrency with many retries

## Fine-Tuning Lessons

### Fine-Tuning Teaches Behavior, Not Knowledge
- A common misconception: fine-tuning doesn't "teach" the model new facts
- It teaches input -> output patterns (behavior)
- This is why we added 311 VERIS Q&A training pairs - to teach the model the behavior of answering VERIS questions, not just classifying incidents
- The model already knows English and JSON from pre-training; we're teaching it a specific mapping

### QLoRA is Remarkably Efficient
- 4-bit quantization + LoRA adapters = fine-tune a 7B model on a single A10G GPU
- Only ~1% of parameters are trainable (the LoRA adapters)
- Training cost: ~$3-5 on HF AutoTrain for 10K examples, 3 epochs
- The adapters are small enough to share and merge easily

### Chat Template Matching Matters
- Each model family has its own chat template (Mistral, Llama, ChatML, etc.)
- Training data must match the model's expected chat template exactly
- `tokenizer.apply_chat_template()` auto-detects the correct format — no manual template specification needed
- Mismatched templates = the model sees garbled input and produces garbage output

### Model Selection for Structured Output
- 7B parameters is the sweet spot for JSON-structured classification with 300+ enum values
- 3B models may lack capacity for the full VERIS enumeration space
- 8B+ models (Llama 3.1) work but are slightly more expensive to train and serve
- Mistral-7B has strong structured output capability and is fully open (Apache 2.0)

### AutoTrain Gotchas
- `autotrain-advanced` pip package doesn't build on Python 3.14 (wheel build failures for Pillow, pydantic-core, etc.)
- The AutoTrain web UI can fail silently: OAuth timeouts to HF Hub cause form submissions to be received but training never starts
- Default settings in the UI are often wrong — always check LR, chat template, column mapping
- 409 errors on "Start Training" = output model repo already exists. Delete it or use a new name.
- 400 errors = invalid config. Check the Space container logs for the actual error, not the UI toast message.
- **Key lesson:** If AutoTrain isn't working, build your own training Space. A Docker container with `trl SFTTrainer` is ~100 lines and you control everything.

### HF Spaces Deployment Debugging
- **Python 3.13 broke Gradio**: `audioop` removed from stdlib → `pydub` import fails → Gradio can't start. Fix: add `pyaudioop` to requirements.txt (backport package). Alternative: use Docker SDK with Python 3.10.
- **Gradio + huggingface_hub version conflict**: `HfFolder` removed in newer `huggingface_hub` but Gradio's OAuth module still imports it. Fix: don't use Gradio for training at all.
- **trl missing `rich`**: `trl` uses `rich.console.Console` but doesn't declare it as a dependency. Fix: add `rich` to requirements.
- **`libgomp: Invalid value for OMP_NUM_THREADS`**: Warning in CUDA containers, harmless but noisy. Caused by HF Spaces setting empty env vars.
- **Spaces require port 7860 open**: Even a training-only container needs a health check endpoint. Python's `http.server` in a daemon thread is the lightest solution.
- **Minimize dependencies for reliability**: Every additional package is a potential version conflict. A training job needs torch + transformers + trl + peft + bitsandbytes. That's it. No web frameworks.
- **HfFolder shim for Gradio on Spaces**: The Spaces base image pre-installs `huggingface_hub >= 0.24` which removed `HfFolder`, while Gradio OAuth internals still reference it. Pinning `huggingface_hub` in requirements doesn't help because the base image version takes priority. Fix: add a compatibility shim at the **top** of `app.py` (before `import gradio`) that monkey-patches `huggingface_hub.HfFolder` with a minimal class wrapping `get_token()` and `login()`.
- **Theme/CSS placement**: Keep `theme` and `css` in the `gr.Blocks()` constructor. Passing them to `app.launch()` raises `TypeError`.

### SFTTrainer Dataset Pre-Processing (The `formatting_func` Trap)
- **Problem**: `SFTTrainer` with `formatting_func` + a dataset containing nested `messages` column (list of dicts) crashes with: `"Unable to create tensor... excessive nesting (inputs type 'list' where type 'int' is expected)"`
- **Root cause**: Even with `remove_unused_columns=False`, the default data collator tries to tensorize ALL columns in the dataset — including the nested `messages` field — before the `formatting_func` can process them into flat text.
- **Failed approach**: Using `formatting_func` parameter to convert messages → text at batch time. The collator runs first.
- **Working approach**: Pre-process the entire dataset with `.map()` BEFORE passing to SFTTrainer. Convert nested messages → flat text using `tokenizer.apply_chat_template()`, then use `dataset_text_field="text"` instead of `formatting_func`.
- **Key insight**: With SFTTrainer, always flatten your data before training. Don't rely on runtime formatting for nested structures.

## ZeroGPU Deployment Lessons

### ZeroGPU Requires HF Pro
- ZeroGPU is NOT available on the free HF tier — attempting to create a Space with `zero-a10g` hardware returns `403 Forbidden: Subscribe to PRO to use ZeroGPU`
- HF Pro costs $9/month but provides free A10G burst GPU for inference (no per-request costs)
- This is still much cheaper than running your own GPU instance or paying per-API-call to OpenAI

### LoRA Adapter Loading
- The fine-tuned model repo (`vibesecurityguy/veris-classifier-v2`) contains only 162 MB of LoRA adapter weights, NOT a full 7B model
- **Wrong**: `AutoModelForCausalLM.from_pretrained("vibesecurityguy/veris-classifier-v2")` — this tries to load adapter weights as a full model and crashes
- **Right**: Load the base model first, then apply the adapter:
  ```python
  model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
  model = PeftModel.from_pretrained(model, "vibesecurityguy/veris-classifier-v2")
  model = model.merge_and_unload()  # Merge for faster inference
  ```

### `@spaces.GPU` Decorator
- The `@spaces.GPU(duration=120)` decorator allocates a GPU only for the duration of the function call
- Must only be applied to functions that actually need GPU — applying it to the wrong functions wastes GPU quota
- The decorator is only available in the `spaces` package (pre-installed on HF Spaces) — guard with `if IS_SPACES:` for local compatibility

## Data Quality Notes
- First 836 rows were generated with gpt-4o (higher quality descriptions)
- Remaining ~9,200 rows generated with gpt-4o-mini (still good quality, spot-checked)
- Some VCDB records have empty actor fields (e.g., skimmer incidents) - the synthetic descriptions handle this gracefully by saying "unidentified perpetrator"
- VCDB records span schema versions 1.2 through 1.4 - some field structures vary slightly
- ~47 VCDB files are accidental duplicates (filename has " copy" or " 2" suffix) - these get deduplicated by incident_id in our pipeline
- Quality validation results: zero jargon leakage, 15 near-duplicates, 3 invalid asset values out of 10K+

## Architecture Decisions
- JSONL format for dataset storage (one JSON object per line) - easy to append, stream, and resume
- Classification target is a simplified/flattened version of full VERIS JSON - only includes the fields a model needs to predict
- Gradio chosen over Streamlit for HF Spaces because it has better API support and the examples feature is built-in
- Dual-mode inference (HF model + OpenAI fallback) for local development, but production Space is HF-only by design
- ZeroGPU for inference - requires HF Pro ($9/month) but no per-request GPU costs. Cold start is 30-60s but acceptable for a portfolio project
- QLoRA over full fine-tuning - 10x cheaper, similar quality for our task, easier to deploy
- JSON-first response + table toggle with filters/CSV export - keeps analyst and reporting workflows in one UI
- Runtime output validation - catches enum/schema drift immediately after inference
- Explicit session status + queue retry/backoff - reduces user confusion around OAuth quota routing and transient ZeroGPU queue failures

## Technical Notes
- **Gradio configuration**: Keep `theme` and `css` in `gr.Blocks()`; passing them to `launch()` raises `TypeError`. Always verify behavior against the installed Gradio version before UI refactors.
- **Python 3.14 compatibility**: `setuptools.backends._legacy:_Backend` doesn't exist - use `setuptools.build_meta` as build backend.
- **venv paths in launch configs**: Dev server configs need absolute paths to the venv Python binary, not just `python`.
- **HF Hub authentication**: Token username must match the repo owner. Our token was for `vibesecurityguy` but scripts initially had `pshamoon` as the username - resulted in 403 Forbidden.
- **Resume support pattern**: JSONL append with flush + dedup by incident_id. Saved the project when the overnight generation run failed.

## Completed (originally Future Improvements)
- [x] Fine-tune a 7B model on the dataset for zero-cost inference (Mistral-7B-Instruct-v0.3 with QLoRA → 162 MB adapter, 3,678 training steps)
- [x] Validate generated classifications against the actual VERIS JSON schema (scripts/07_validate_dataset.py)
- [x] Build an evaluation harness comparing fine-tuned model vs GPT-4o accuracy (scripts/06_evaluate.py with --compare flag)
- [x] Model published to HuggingFace Hub: `vibesecurityguy/veris-classifier-v2`
- [x] Deploy Gradio app to HF Spaces with ZeroGPU: [vibesecurityguy/veris-classifier](https://huggingface.co/spaces/vibesecurityguy/veris-classifier)
- [x] Model card with full training details pushed to HF Hub
- [x] Add output toggle (`JSON` / `Table`) with table filters and CSV export
- [x] Add runtime validation summary in classify flow
- [x] Add HF session status indicator and ZeroGPU queue retry/backoff

## Remaining Future Improvements
- [ ] Add confidence scoring to classifications
- [ ] Add batch classification mode (upload CSV of incidents)
- [ ] Scrape original reference URLs from VCDB records for real incident descriptions
- [ ] Add VERIS-to-MITRE ATT&CK mapping
- [ ] Support VERIS schema version selection (1.3 vs 1.4)
- [ ] Multi-language support
