# Hardware smoke test

Development cases only; excluded from the full report.

- llama3.2:3b: 1.6 estimated inference hours; timings {"bbq": {"mean_seconds": 0.5360424895833754, "n": 12}, "own": {"mean_seconds": 1.7954801664999043, "n": 6}}
- mistral:7b-instruct: 4.1 estimated inference hours; timings {"bbq": {"mean_seconds": 1.3867564619166994, "n": 12}, "own": {"mean_seconds": 4.753738055499828, "n": 6}}

This extrapolation includes cold starts and short own outputs; long prompts, thermal throttling, and retries may change runtime.

Requests: 36. Unparsed BBQ outputs: 3.
Transport, both frozen models, all three LLM conditions, unloading, response caching, VADER and CPU Sentence-BERT verified.
