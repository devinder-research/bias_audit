# Reproducibility

The reported audit is reproducible from the frozen code, configuration, dataset split, scorer revision, and recorded responses. The submission archive includes both original response logs, the analysis inputs, and the exported results. On the recorded environment, `python audit.py report` can regenerate the numerical tables and figures from those logs without new model inference. A fresh inference run repeats the protocol and may differ across runtimes or hardware. The published table values can be checked against the original cached outputs.

## Source and environment

- Repository: <https://github.com/devinder-research/bias_audit>
- Evaluated snapshot: [`abba5417c2d7abca8a6c2fc6c99d47aa2e053059`](https://github.com/devinder-research/bias_audit/commit/abba5417c2d7abca8a6c2fc6c99d47aa2e053059).
- Native Apple Silicon macOS, Python **3.10.11**, packages in [`requirements.txt`](requirements.txt), and Ollama **0.40.17** for fresh generation.
- [`data/frozen.json`](data/frozen.json) records the full model digests and hashes for frozen code, cases, and encoder files. [`config.json`](config.json) records decoding settings, sample and bootstrap seeds, and the expected request counts.

| Commit | Full SHA | Commit date and time in IST |
| --- | --- | --- |
| Regenerate report | `abba5417c2d7abca8a6c2fc6c99d47aa2e053059` | 2026-11-25 00:50:08 +05:30 |
| Complete manual error review | `709c2372504a5e1ca1e999e18d164936a3fa29df` | 2026-11-24 23:45:53 +05:30 |
| Update README | `17a7acbadb7e6720ee92165dba924ea46fb6e8d4` | 2026-11-23 23:56:38 +05:30 |
| Initial audit | `3097ebf2cece8fe2286d2891868641dd5ba6f079` | 2026-11-23 23:49:17 +05:30 |

## Checkpoints and data

The audit used pretrained weights and produced no trained checkpoints. The two Ollama builds total approximately 6.4 GB, so the submission points to the model registries. Verify the digest prefixes with `ollama list` before a fresh run.

| Component | Exact build or revision | Download |
| --- | --- | --- |
| Llama 3.2 3B Instruct | `llama3.2:3b`, `a80c4f17acd5`, 2.0 GB | [Ollama registry](https://ollama.com/library/llama3.2:3b) |
| Mistral 7B Instruct v0.3 | `mistral:7b-instruct`, `6577803aa9a0`, 4.4 GB | [Ollama registry](https://ollama.com/library/mistral:7b-instruct) |
| All-MiniLM-L6-v2 | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | [Pinned Hugging Face revision](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41) |

The repository includes the four BBQ source files, their shared metadata, and the frozen sample manifest, which total 21.4 MB. The source is publicly hosted at the [pinned BBQ revision](https://github.com/nyu-mll/BBQ/tree/bea11bd97d79217245b5871acd247b9d6eb24598). The dataset remains subject to its original license and attribution.

## Regenerate the submitted results from cached responses

Extract `Bias_Audit_Reproducibility.zip` to a new directory and work from its root. The two original response logs are in `results/models/{llama,mistral}/responses.jsonl`, with 6,708 records in each. The archive contains the original result tables, figures and metric caches for comparison. The local model server is unnecessary for this route.

```bash
shasum -a 256 -c SHA256SUMS
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python - <<'PY'
import json
from pathlib import Path
from huggingface_hub import snapshot_download

config = json.loads(Path('config.json').read_text())
snapshot_download(
    config['encoder'],
    revision=config['encoder_revision'],
    local_dir='data/encoder',
    allow_patterns=['*.json', '1_Pooling/*.json', 'model.safetensors', 'vocab.txt'],
)
PY
python -m unittest -v test_audit.py
python audit.py report
```

`report` validates request identities and verifies that the metrics belong to the supplied responses. To recompute all metrics from the raw logs, move the three `metrics.json` files to a separate backup directory in the extracted copy, then run `python audit.py report` again. The original CSV tables provide the reference values for comparison. PNG pixel content and SVG graphics can be compared visually, since SVG timestamps and generated element IDs may change when figures are regenerated. The Markdown review sections incorporate the accompanying annotations.

## Run the experiment again from scratch

Follow [Run a new audit](README.md#run-a-new-audit) to start with a fresh working directory. Install Ollama 0.40.17, start its local server, and pull the builds listed above with `ollama pull llama3.2:3b` and `ollama pull mistral:7b-instruct`. The documented setup checks the digests, and the full run issues 13,416 requests across development selection, the main test, paraphrases, and open-ended prompts. The deterministic always-unknown baseline uses no model calls.
