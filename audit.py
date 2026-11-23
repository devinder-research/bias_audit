#!/usr/bin/env python3
"""Local reproducible audit. Run setup once, then run (or run --smoke)."""
import argparse
import fcntl
import importlib.metadata
import json
import os
import platform
import sys
from types import SimpleNamespace
from collections import Counter
from pathlib import Path

# python.org's macOS build may not have its default certificate link installed.
if 'SSL_CERT_FILE' not in os.environ and Path('/etc/ssl/cert.pem').exists():
    os.environ['SSL_CERT_FILE'] = '/etc/ssl/cert.pem'
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')

import data
import prompts
from ollama import Cache, fingerprint, preflight, read_json, unload, write_json

ROOT = Path(__file__).resolve().parent
MODEL_TAGS = {'llama': 'llama3.2:3b', 'mistral': 'mistral:7b-instruct'}
CODE = ('audit.py', 'data.py', 'prompts.py', 'ollama.py', 'metrics.py', 'report.py', 'charts.py', 'requirements.txt')


def environment():
    if not __debug__:
        raise ValueError('Run without -O: protocol validation checks must remain enabled.')
    if platform.machine() != 'arm64' or sys.prefix == sys.base_prefix:
        raise ValueError('Use native arm64 Python inside the project virtual environment (see README).')


def setup(config):
    environment()
    if (ROOT / 'data' / 'frozen.json').exists():
        load(config, online=True)
        print('Frozen setup verified; no models or data changed.', flush=True)
        return
    models = preflight(config)
    print('Preparing official BBQ data and isolated splits...', flush=True)
    manifest = data.prepare(ROOT, config)
    data.verify_files(ROOT, manifest)
    print('Obtaining the pinned CPU scoring encoder (never Ollama weights)...', flush=True)
    from huggingface_hub import snapshot_download
    encoder = ROOT / 'data' / 'encoder'
    snapshot_download(config['encoder'], revision=config['encoder_revision'], local_dir=encoder,
                      allow_patterns=['*.json', '1_Pooling/*.json', 'model.safetensors', 'vocab.txt'],
                      max_workers=2)
    # Instantiate once to verify the frozen scorer is complete and usable offline.
    for model in config['models']:
        unload(config, model)
    from sentence_transformers import SentenceTransformer
    scorer = SentenceTransformer(str(encoder), device='cpu', local_files_only=True)
    assert scorer.encode(['A verified scoring setup.']).shape == (1, 384)
    scorer_hashes = {str(p.relative_to(encoder)): data.sha(p) for p in encoder.rglob('*')
                     if p.is_file() and '.cache' not in p.parts}
    packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()
                if d.metadata['Name'].lower() not in ('pip', 'setuptools')}
    frozen = {'config': config, 'models': models, 'cases_sha256': data.sha(ROOT / 'data' / 'cases.json'),
              'code_hashes': {name: data.sha(ROOT / name) for name in CODE},
              'encoder_files': scorer_hashes, 'packages': packages,
              'python': platform.python_version(), 'machine': platform.machine()}
    write_json(ROOT / 'data' / 'frozen.json', frozen)
    print('Setup frozen: 96 dev, 896 test, 320 matched robustness items, 4 demos, 180 own cases.', flush=True)


def load(config, online=False):
    environment()
    frozen = read_json(ROOT / 'data' / 'frozen.json')
    if frozen['config'] != config:
        raise ValueError('Config changed after setup. Restore the frozen protocol.')
    for name, digest in frozen['code_hashes'].items():
        if data.sha(ROOT / name) != digest:
            raise ValueError(f'Frozen implementation changed: {name}. Restore it before resuming.')
    if data.sha(ROOT / 'data' / 'cases.json') != frozen['cases_sha256']:
        raise ValueError('Frozen case manifest changed.')
    for name, digest in frozen['encoder_files'].items():
        if data.sha(ROOT / 'data' / 'encoder' / name) != digest:
            raise ValueError(f'Frozen encoder changed: {name}')
    installed = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    if any(installed.get(name) != version for name, version in frozen['packages'].items()):
        raise ValueError('Scoring environment differs from setup; restore pinned dependencies.')
    if platform.python_version() != frozen['python']:
        raise ValueError('Python version differs from frozen setup.')
    manifest = read_json(ROOT / 'data' / 'cases.json')
    data.validate(manifest, config)
    data.verify_files(ROOT, manifest)
    if online:
        preflight(config, frozen['models'])
    return manifest, frozen, fingerprint(frozen)


def request(case, model, condition, candidate, wording, stage, config, frozen, protocol, demos):
    options = dict(config['own_options'] if case['kind'] == 'own' else config['bbq_options'])
    if case['kind'] == 'own':
        options['seed'] = case['seed']
    return {'protocol': protocol, 'model': model, 'digest': frozen['models']['models'][model],
            'kind': case['kind'], 'split': case.get('split', 'own'), 'stage': stage,
            'condition': condition, 'candidate': candidate, 'wording': wording,
            'item_id': case['id'], 'options': options,
            'messages': prompts.messages(case, condition, candidate, wording, demos)}


def plan(stage, manifest, config, frozen, protocol, selected=None, smoke=False):
    if smoke:
        family = manifest['bbq'][0]['family']
        cases = [c for c in manifest['bbq'] if c['family'] == family]
        cases += manifest['own'][:2]
    else:
        cases = manifest['own'] if stage == 'own' else [c for c in manifest['bbq'] if
                 (c['split'] == 'dev' if stage == 'development' else c['split'] == 'test') and
                 (stage != 'robustness' or c['robust'])]
    arms = [('B1', '-'), ('M1', 'a'), ('M2', 'a')] if smoke else (
        [('B1', '-'), ('M1', 'a'), ('M1', 'b'), ('M2', 'a'), ('M2', 'b')] if stage == 'development' else
        [('B1', '-'), ('M1', selected['M1']), ('M2', selected['M2'])])
    wordings = (1, 2) if stage == 'robustness' and not smoke else (range(3) if stage == 'own' and not smoke else (0,))
    for model in config['models']:
        for condition, candidate in arms:
            for wording in wordings:
                for case in cases:
                    yield request(case, model, condition, candidate, wording, stage, config,
                                  frozen, protocol, manifest['demos']), case


def execute(jobs, cache, config, frozen, limit=None):
    jobs = list(jobs)
    pending = [(req, case) for req, case in jobs if cache.get(req) is None]
    print(f'{len(pending):,} pending / {len(jobs):,} requests in this stage.', flush=True)
    for (model, stage), count in Counter((r['model'], r['stage']) for r, _ in pending).items():
        print(f'  {model} / {stage}: {count:,} pending', flush=True)
    done, active = 0, None
    try:
        for req, case in pending:
            if limit is not None and done >= limit:
                return False
            if active != req['model']:
                if active:
                    unload(config, active)
                preflight(config, frozen['models'])
                active = req['model']
                # Also clear any model left loaded by a prior interrupted run.
                for tag in config['models']:
                    if tag != active:
                        unload(config, tag)
                print(f'{active} / {req["stage"]}: starting model block', flush=True)
            cache.run(config, req, case)
            done += 1
            if done == 1 or done % 10 == 0 or done == len(pending):
                print(f'{active} / {req["stage"]}: {done:,}/{len(pending):,} pending completed', flush=True)
    finally:
        if active:
            unload(config, active)
    return True


def model_alias(model):
    return next(alias for alias, tag in MODEL_TAGS.items() if tag == model)


def model_output(model, smoke=False):
    return ROOT / 'results' / ('smoke' if smoke else 'models') / model_alias(model)


def model_cache(model, frozen, protocol, smoke=False):
    output = model_output(model, smoke)
    manifest_path = output / 'run_manifest.json'
    run_manifest = {'protocol': protocol, 'smoke': smoke, 'model': model, 'frozen': frozen}
    if manifest_path.exists() and read_json(manifest_path) != run_manifest:
        raise ValueError(f'Run manifest differs for {model}; refusing to resume.')
    write_json(manifest_path, run_manifest)
    return Cache(output / 'responses.jsonl', protocol)


def model_jobs(stage, model, manifest, config, frozen, protocol, selection=None, smoke=False):
    return [(req, case) for req, case in plan(stage, manifest, config, frozen, protocol, selection, smoke)
            if req['model'] == model]


def combine(caches):
    records = {}
    for cache in caches.values():
        if records.keys() & cache.records.keys():
            raise ValueError('Duplicate requests across model folders.')
        records.update(cache.records)
    return SimpleNamespace(records=records)


def verify_complete(cache, manifest, config, frozen, protocol, selection, models=None):
    models = list(config['models']) if models is None else models
    expected = []
    for stage, count in config['expected_requests'].items():
        jobs = list(plan(stage, manifest, config, frozen, protocol, selection['selected']))
        if len(jobs) != count:
            raise ValueError(f'Unexpected {stage} plan count: {len(jobs)} != {count}')
        expected.extend(fingerprint(req) for req, _ in jobs if req['model'] in models)
    if len(set(expected)) != 6708 * len(models) or set(expected) != set(cache.records):
        raise ValueError(f'Incomplete or unexpected cache: {len(cache.records):,}/{len(expected):,} expected requests.')


def freeze_selection(manifest, config, frozen, protocol, caches, prepare=False):
    jobs = list(plan('development', manifest, config, frozen, protocol))
    if prepare:
        for model, cache in caches.items():
            execute([job for job in jobs if job[0]['model'] == model], cache, config, frozen)
    if any(caches[req['model']].get(req) is None for req, _ in jobs):
        raise ValueError('Shared development selection is not ready. First run: python3 audit.py prepare')
    from metrics import select_candidates
    rows = [caches[req['model']].get(req) for req, _ in jobs]
    selection = select_candidates([c for c in manifest['bbq'] if c['split'] == 'dev'], rows, config)
    selection.update(protocol=protocol, development_identity=fingerprint([r['checksum'] for r in rows]))
    path = ROOT / 'results' / 'selection.json'
    if path.exists() and read_json(path) != selection:
        raise ValueError('Development selection differs from the frozen decision.')
    write_json(path, selection)
    return selection


def smoke_report(cache, config, frozen, protocol, model):
    from sentence_transformers import SentenceTransformer
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import numpy as np
    encoder = SentenceTransformer(str(ROOT / 'data' / 'encoder'), device='cpu', local_files_only=True)
    embeddings = encoder.encode(['A clear and helpful response.'] * 2, normalize_embeddings=True)
    assert np.isclose(embeddings[0] @ embeddings[1], 1)
    assert SentimentIntensityAnalyzer().polarity_scores('excellent')['compound'] > 0
    rows = list(cache.records.values())
    stats, total = {}, 0
    for kind, count in [('bbq', 5088), ('own', 1620)]:
        times = [r['elapsed_seconds'] for r in rows if r['request']['kind'] == kind]
        stats[kind] = {'mean_seconds': sum(times) / len(times), 'n': len(times)}
        total += stats[kind]['mean_seconds'] * count
    output = model_output(model, smoke=True)
    write_json(output / 'timings.json', {'protocol': protocol, 'model': model, 'timings': stats,
                                       'estimated_hours': total / 3600})
    invalid = sum(r['parsed'] is None for r in rows if r['request']['kind'] == 'bbq')
    (output / 'report.md').write_text(f'# {model}: hardware smoke test\n\n'
        f'{len(rows)} development/own smoke requests; {invalid} unparsed BBQ answers.\n\n'
        f'Estimated full-model inference time: {total / 3600:.1f} hours. '
        'This extrapolation includes cold starts and short outputs; throttling and retries can change it.\n\n'
        'Verified local inference, all three LLM conditions, both task types, CPU scoring and caching. '
        'These responses are excluded from the full experiment.\n')
    print(f'Smoke report: {output / "report.md"}; estimated {total / 3600:.1f} hours for {model}', flush=True)


def render_models(manifest, config, frozen, protocol, caches, selection, models):
    import report
    for model in models:
        cache = caches[model]
        verify_complete(cache, manifest, config, frozen, protocol, selection, [model])
        model_config = dict(config, models={model: config['models'][model]})
        report.generate(ROOT, manifest, model_config, frozen, protocol, cache, selection, model_output(model))
    # A partial second model must never prevent a completed first-model report.
    if sum(len(c.records) for c in caches.values()) == 13416:
        merged = combine(caches)
        verify_complete(merged, manifest, config, frozen, protocol, selection)
        report.generate(ROOT, manifest, config, frozen, protocol, merged, selection)


def run(config, smoke=False, limit=None, model=None, prepare_only=False, report_only=False):
    manifest, frozen, protocol = load(config, online=not report_only)
    models = [model] if model else list(config['models'])
    caches = {m: model_cache(m, frozen, protocol, smoke) for m in (models if smoke else config['models'])}
    if smoke:
        complete = []
        for current in models:
            jobs = model_jobs('smoke', current, manifest, config, frozen, protocol, smoke=True)
            if execute(jobs, caches[current], config, frozen, limit):
                if set(caches[current].records) != {fingerprint(req) for req, _ in jobs}:
                    raise ValueError('Unexpected smoke request identities.')
                complete.append(current)
        # Never load the encoder while further model inference remains in this command.
        for current in complete:
            smoke_report(caches[current], config, frozen, protocol, current)
        return
    selection = freeze_selection(manifest, config, frozen, protocol, caches,
                                 prepare=prepare_only or (not model and not report_only))
    print(f'Frozen pooled-development selection: {selection["selected"]}', flush=True)
    if prepare_only:
        print('Preparation complete. Run --model llama and --model mistral independently.', flush=True)
        return
    if not report_only:
        pending = sum(6708 - len(caches[m].records) for m in models)
        print(f'{pending:,} pending evaluation requests across {len(models)} selected model(s).', flush=True)
        for current in models:
            jobs = [job for stage in ('test', 'robustness', 'own') for job in
                    model_jobs(stage, current, manifest, config, frozen, protocol, selection['selected'])]
            execute(jobs, caches[current], config, frozen)
    render_models(manifest, config, frozen, protocol, caches, selection, models)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('setup', 'prepare', 'run', 'report'))
    parser.add_argument('--model', choices=(*MODEL_TAGS, *MODEL_TAGS.values()),
                        help='Run/report only this model; shared preparation is required first.')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--max-requests', type=int, help='Smoke only: stop after N new requests per selected model.')
    args = parser.parse_args()
    if args.model and args.command not in ('run', 'report'):
        parser.error('--model applies only to run or report.')
    if (args.smoke or args.max_requests is not None) and args.command != 'run':
        parser.error('Smoke options require run.')
    if args.max_requests is not None and (not args.smoke or args.max_requests < 1):
        parser.error('--max-requests requires --smoke and a positive count.')
    os.chdir(ROOT)
    config = read_json(ROOT / 'config.json')
    with (ROOT / '.audit.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another setup/run/report process is already active.')
        if args.command == 'setup':
            setup(config)
        else:
            model = MODEL_TAGS.get(args.model, args.model)
            run(config, args.smoke, args.max_requests, model=model,
                prepare_only=args.command == 'prepare', report_only=args.command == 'report')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, OSError) as error:
        print(f'Audit stopped: {error}', file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print('\nInterrupted. Completed responses are saved; repeat the same command to resume.', file=sys.stderr)
        sys.exit(130)
