"""Official BBQ metadata join, conservative split isolation, and frozen own prompts."""
import csv
import hashlib
import json
import random
import re
import ssl
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import prompts
from ollama import fingerprint, read_json, write_json


def download(url, path):
    path = Path(path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            context = ssl.create_default_context(cafile="/etc/ssl/cert.pem" if Path("/etc/ssl/cert.pem").exists() else None)
            with urllib.request.urlopen(url, timeout=60, context=context) as response:
                content = response.read()
            temp = path.with_suffix('.download')
            temp.write_bytes(content)
            temp.replace(path)
            return
        except (OSError, TimeoutError):
            if attempt == 2:
                raise


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_cases(root, config):
    source = root / 'data' / 'source'
    names = [f'data/{c}.jsonl' for c in config['categories']] + ['supplemental/additional_metadata.csv']
    hashes = {}
    for name in names:
        path = source / Path(name).name
        download(f'https://raw.githubusercontent.com/nyu-mll/BBQ/{config["source_revision"]}/{name}', path)
        hashes[name] = sha(path)
    with (source / 'additional_metadata.csv').open(newline='') as handle:
        metadata = {}
        for row in csv.DictReader(handle):
            if row['category'] not in config['categories']:
                continue
            key = (row['category'], int(row['example_id']))
            if key in metadata and any(metadata[key][field] != row[field] for field in ('target_loc', 'question_index')):
                raise ValueError(f'Conflicting official target metadata: {key}')
            metadata[key] = row
    cases, excluded = [], []
    for category in config['categories']:
        for line in (source / f'{category}.jsonl').read_text().splitlines():
            item = json.loads(line)
            meta = metadata[(category, item['example_id'])]
            unknowns = [i for i in range(3) if item['answer_info'][f'ans{i}'][1] == 'unknown']
            assert len(unknowns) == 1
            unknown = unknowns[0]
            if meta['target_loc'] == 'NA':
                excluded.append({'id': f'{category}:{item["example_id"]}', 'reason': 'official target_loc is NA'})
                continue
            official_target = int(meta['target_loc'])
            # The official CSV has ALREADY reversed non-negative questions.
            # Recover the demographic target so the explicit polarity formula applies once.
            target = (official_target if item['question_polarity'] == 'neg' else
                      next(i for i in range(3) if i not in (official_target, unknown)))
            label = item['label']
            assert target in range(3) and target != unknown
            assert item['question_polarity'] in ('neg', 'nonneg')
            assert item['context_condition'] in ('ambig', 'disambig')
            assert label in range(3) and (label == unknown) == (item['context_condition'] == 'ambig')
            assert str(item['question_index']) == meta['question_index']
            cases.append({
                'kind': 'bbq', 'id': f'{category}:{item["example_id"]}', 'category': category,
                'example_id': item['example_id'], 'question_index': str(item['question_index']),
                # Entire authored question template stays in one split, including name/group variants.
                'group_key': f'{category}:q{item["question_index"]}',
                'family': f'{category}:f{item["example_id"] // 4}',
                'context': item['context'], 'question': item['question'],
                'answers': [item[f'ans{i}'] for i in range(3)],
                'label': label, 'unknown': unknown, 'target': target, 'official_target': official_target,
                'polarity': item['question_polarity'], 'context_condition': item['context_condition'],
            })
    return cases, hashes, excluded


def own_cases(config):
    cases = []
    for axis, spec in prompts.OWN_AXES.items():
        for template, wordings in enumerate(prompts.OWN_TEMPLATES):
            for group in spec['groups']:
                subject = spec['subject'].format(group=group).replace('an nurse', 'a nurse')
                role = f' who works as a {spec["role"]}' if spec['role'] else ''
                for seed in config['generation_seeds']:
                    cases.append({'kind': 'own', 'id': f'{axis}:{template}:{group}:{seed}',
                                  'axis': axis, 'template': template, 'group': group, 'seed': seed,
                                  'wordings': [w.format(subject=subject, role=role) for w in wordings]})
    return cases


def normalized_passage(text):
    return re.sub(r'\W+', ' ', text.casefold()).strip()


def validate(manifest, config):
    assert manifest['source_revision'] == config['source_revision']
    assert manifest['sample_seed'] == config['sample_seed']
    bbq = manifest['bbq']
    assert len(bbq) == 992 and len({c['id'] for c in bbq}) == 992
    assert Counter(c['split'] for c in bbq) == {'dev': 96, 'test': 896}
    by_split = {s: [c for c in bbq if c['split'] == s] for s in ('dev', 'test')}
    by_split['demo'] = manifest['demos']
    for left, right in [('dev', 'test'), ('dev', 'demo'), ('test', 'demo')]:
        assert not ({c['group_key'] for c in by_split[left]} & {c['group_key'] for c in by_split[right]})
        assert not ({normalized_passage(c['context']) for c in by_split[left]} &
                    {normalized_passage(c['context']) for c in by_split[right]})
    for split, total in [('dev', 96), ('test', 896), ('robust', 320)]:
        rows = [c for c in bbq if c['robust']] if split == 'robust' else by_split[split]
        counts = Counter((c['category'], c['context_condition'], c['polarity']) for c in rows)
        assert len(rows) == total and len(counts) == 16 and set(counts.values()) == {total // 16}
    demos = manifest['demos']
    assert len(demos) == 4 and {c['label'] for c in demos} == {0, 1, 2}
    assert sum(c['context_condition'] == 'ambig' for c in demos) == 2
    clear = [c for c in demos if c['context_condition'] == 'disambig']
    assert {aligned(c, c['label']) for c in clear} == {True, False}
    for case in bbq + demos:
        assert all(aligned(case, choice) == (choice == case['official_target']) for choice in range(3))
    family_counts = Counter(c['family'] for c in bbq)
    assert set(family_counts.values()) == {4}
    assert len(manifest['own']) == 180 and manifest['own'] == own_cases(config)


def aligned(case, choice):
    return choice != case['unknown'] and ((choice == case['target']) == (case['polarity'] == 'neg'))


def prepare(root, config):
    path = root / 'data' / 'cases.json'
    if path.exists():
        manifest = read_json(path)
        validate(manifest, config)
        return manifest
    cases, hashes, excluded = source_cases(root, config)
    rng = random.Random(config['sample_seed'])
    families = defaultdict(list)
    for case in cases:
        families[case['family']].append(case)
    # Only complete four-way matched families; never reconstruct or rewrite benchmark text.
    families = {key: rows for key, rows in families.items() if len(rows) == 4 and
                len({(c['context_condition'], c['polarity']) for c in rows}) == 4 and
                len({c['group_key'] for c in rows}) == 1}
    grouped = defaultdict(list)
    for key, rows in families.items():
        grouped[rows[0]['group_key']].append(key)
    # Reserve complete question templates for demonstrations before any evaluation split.
    demo_keys = set()
    demos = []
    for key in sorted(grouped):
        rows = [c for f in grouped[key] for c in families[f]]
        amb = [c for c in rows if c['context_condition'] == 'ambig']
        clear = [c for c in rows if c['context_condition'] == 'disambig']
        if not amb or not clear:
            continue
        a = amb[0]
        for b in clear:
            for c in clear:
                if aligned(b, b['label']) == aligned(c, c['label']):
                    continue
                if {a['label'], b['label'], c['label']} == {0, 1, 2}:
                    demos = [a, amb[1], b, c]
                    demo_keys = {key}
                    break
            if demos:
                break
        if demos:
            break
    if not demos:
        raise ValueError('Cannot find reserved balanced demonstrations.')
    selected = []
    for category in config['categories']:
        keys = sorted(k for k in grouped if k.startswith(category + ':') and k not in demo_keys)
        rng.shuffle(keys)
        dev_keys, keys = keys[:6], keys[6:]
        dev_families = [rng.choice(grouped[k]) for k in dev_keys]
        test_families = [f for k in keys for f in grouped[k]]
        rng.shuffle(test_families)
        if len(test_families) < 56:
            raise ValueError(f'Insufficient independent test families for {category}')
        chosen_test = test_families[:56]
        robust = set(rng.sample(chosen_test, 20))
        for split, chosen in [('dev', dev_families[:6]), ('test', chosen_test)]:
            for f in chosen:
                for case in families[f]:
                    selected.append(dict(case, split=split, robust=(f in robust)))
    manifest = {'source_revision': config['source_revision'], 'source_hashes': hashes,
                'source_exclusions': excluded, 'sample_seed': config['sample_seed'], 'split_rule': 'category/question_index template isolation; complete four-item families',
                'bbq': selected, 'demos': demos, 'own': own_cases(config),
                'prompts': prompts.snapshot()}
    validate(manifest, config)
    write_json(path, manifest)
    return manifest


def verify_files(root, manifest):
    for name, expected in manifest['source_hashes'].items():
        if sha(root / 'data' / 'source' / Path(name).name) != expected:
            raise ValueError(f'Official source changed: {name}')
    if manifest['prompts'] != json.loads(json.dumps(prompts.snapshot())):
        raise ValueError('Frozen prompts differ from prompts.py.')
