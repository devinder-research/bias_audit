"""Protocol checks. Synthetic artifacts stay in temporary directories."""
import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import audit
import data
import metrics
import prompts
import report
from ollama import Cache, canonical, fingerprint, read_json

ROOT = Path(__file__).resolve().parent
CONFIG = read_json(ROOT / 'config.json')
MANIFEST = read_json(ROOT / 'data' / 'cases.json')
FROZEN = {'models': {'models': {tag: prefix + 'synthetic' for tag, prefix in CONFIG['models'].items()},
                      'ollama_version': 'synthetic'}, 'machine': 'arm64', 'python': 'synthetic'}
PROTOCOL = 'synthetic-test-only'


class AuditTests(unittest.TestCase):
    def test_splits_and_exact_request_counts(self):
        data.validate(MANIFEST, CONFIG)
        dev = list(audit.plan('development', MANIFEST, CONFIG, FROZEN, PROTOCOL))
        selected = {'M1': 'a', 'M2': 'b'}
        expected = {'development': 960, 'test': 5376, 'robustness': 3840, 'own': 3240}
        keys = []
        for stage, count in expected.items():
            jobs = list(audit.plan(stage, MANIFEST, CONFIG, FROZEN, PROTOCOL, selected))
            self.assertEqual(len(jobs), count)
            keys += [fingerprint(req) for req, _ in jobs]
        self.assertEqual(len(set(keys)), 13416)
        self.assertEqual(len(dev), 960)
        main = [c['id'] for c in MANIFEST['bbq'] if c['split'] == 'test' and c['robust']]
        for wording in (1, 2):
            robust = [c['id'] for r, c in audit.plan('robustness', MANIFEST, CONFIG, FROZEN, PROTOCOL, selected)
                      if r['model'] == next(iter(CONFIG['models'])) and r['condition'] == 'B1' and r['wording'] == wording]
            self.assertEqual(main, robust)
        own = MANIFEST['own']
        for r, c in audit.plan('own', MANIFEST, CONFIG, FROZEN, PROTOCOL, selected):
            self.assertEqual(r['options']['temperature'], .7)
            self.assertEqual(r['options']['seed'], c['seed'])
        self.assertEqual(len(own), 180)

    def test_known_bbq_cases_and_invalid_outputs(self):
        cases = MANIFEST['demos']
        first = cases[0]
        self.assertEqual(first['id'], 'Age:0')
        self.assertTrue(data.aligned(first, 0))
        self.assertFalse(data.aligned(dict(first, polarity='nonneg'), 0))
        # Official target_loc is polarity-adjusted; never reverse it a second time.
        official, _, _ = data.source_cases(ROOT, CONFIG)
        for case in official:
            for choice in range(3):
                self.assertEqual(data.aligned(case, choice), choice == case['official_target'])
        age = {c['example_id']: c for c in official if c['category'] == 'Age'}
        self.assertFalse(data.aligned(age[1], age[1]['label']))
        self.assertFalse(data.aligned(age[3], age[3]['label']))
        self.assertEqual(age[0]['target'], age[2]['target'])
        self.assertNotEqual(age[0]['official_target'], age[2]['official_target'])
        test = [c for c in MANIFEST['bbq'] if c['split'] == 'test']
        result = metrics.bbq_summary(test, [c['unknown'] for c in test], CONFIG)
        self.assertEqual(result['accuracy']['value'], .5)
        self.assertEqual(result['ambig_bias']['value'], 0)
        self.assertIsNone(result['disambig_bias']['value'])
        invalid = metrics.bbq_summary(test, [None] * len(test), CONFIG)
        self.assertIsNone(invalid['accuracy']['value'])
        self.assertIsNone(invalid['ambig_bias']['value'])
        self.assertEqual(invalid['strict_accuracy']['value'], 0)
        self.assertEqual(invalid['unparsed'], 896)
        amb = [dict(first, polarity='neg'), dict(first, polarity='nonneg')]
        self.assertEqual(metrics.scores(metrics.bbq_stats(amb, [first['target']] * 2).sum(axis=0))['ambig_bias'], 0)

    def test_invalid_answers_cannot_create_success(self):
        cases = [c for c in MANIFEST['bbq'] if c['split'] == 'dev']
        baseline = [c['label'] if c['context_condition'] == 'disambig' or i % 8 < 4 else
                    (c['target'] if c['polarity'] == 'neg' else
                     next(x for x in range(3) if x not in (c['target'], c['unknown'])))
                    for i, c in enumerate(cases)]
        mitigation = [c['unknown'] if c['context_condition'] == 'ambig' else None for c in cases]
        result = metrics.bbq_comparison(cases, baseline, mitigation, CONFIG)
        self.assertGreater(result['ambig_reduction']['value'], 0)
        self.assertLess(result['accuracy_loss_pp']['value'], 0)
        self.assertGreater(result['strict_accuracy_loss_pp']['value'], 3)
        self.assertFalse(result['utility_met'])
        self.assertEqual(result['status'], 'utility target failed')

    def test_parsing_is_conservative(self):
        answers = ['The grandfather', 'The grandson', "Can't be determined"]
        for text in ('A', 'a.', '(A)', '[A]'):
            self.assertEqual(prompts.parse_choice(text, answers), 0)
        self.assertEqual(prompts.parse_choice('  THE GRANDFATHER. ', answers), 0)
        self.assertEqual(prompts.parse_choice('A) The grandfather', answers), 0)
        self.assertEqual(prompts.parse_choice('(A) The grandfather', answers), 0)
        for text in ('A or B', 'Answer: A', 'A. The grandson', 'A) The grandfather because he is older', '', 'C because not enough evidence'):
            self.assertIsNone(prompts.parse_choice(text, answers))
        self.assertIsNone(prompts.parse_choice('same', ['same', 'same', 'unknown']))

    def test_paired_bootstrap(self):
        cases = [c for c in MANIFEST['bbq'] if c['split'] == 'test']
        f1, w1 = metrics.family_weights(cases, 2000, 26008)
        f2, w2 = metrics.family_weights(cases, 2000, 26008)
        self.assertEqual(f1, f2)
        np.testing.assert_array_equal(w1, w2)
        np.testing.assert_array_equal(w1.sum(axis=1), np.full(2000, 224))
        choices = [c['label'] for c in cases]
        result = metrics.bbq_comparison(cases, choices, choices, CONFIG)
        for key in ('ambig_reduction', 'accuracy_loss_pp', 'strict_accuracy_loss_pp'):
            self.assertEqual(result[key]['value'], 0)
            self.assertEqual(result[key]['low'], 0)
            self.assertEqual(result[key]['high'], 0)
        self.assertEqual(result['status'], 'inconclusive')

    def test_cache_resume_and_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'cache.jsonl'
            cache = Cache(path, PROTOCOL)
            request = {'protocol': PROTOCOL, 'seed': 1}
            cache.add(request, 'A or B', None, .1)
            with path.open('ab') as handle:
                handle.write(b'{"partial":')
            resumed = Cache(path, PROTOCOL)
            self.assertEqual(len(resumed.records), 1)
            self.assertEqual(resumed.get(request)['raw'], 'A or B')
            self.assertIsNone(resumed.get(dict(request, seed=2)))
            with self.assertRaises(ValueError):
                Cache(path, 'different-protocol')
            with self.assertRaises(ValueError):
                resumed.add(request, 'A', 0, .1)
            with patch('ollama.http', side_effect=AssertionError('Unexpected inference')):
                self.assertEqual(resumed.run(CONFIG, request, {})['parsed'], None)

    def test_candidate_selection_and_fallback(self):
        cases = [c for c in MANIFEST['bbq'] if c['split'] == 'dev']
        jobs = list(audit.plan('development', MANIFEST, CONFIG, FROZEN, PROTOCOL))
        def records(invalid=False):
            return [{'request': req, 'parsed': None if invalid else c['label']} for req, c in jobs]
        selected = metrics.select_candidates(cases, records(), CONFIG)
        self.assertEqual(selected['selected'], {'M1': 'a', 'M2': 'a'})
        self.assertFalse(selected['development']['M1']['fallback'])
        fallback = metrics.select_candidates(cases, records(True), CONFIG)
        self.assertEqual(fallback['selected'], {'M1': 'a', 'M2': 'a'})
        self.assertTrue(fallback['development']['M1']['fallback'])

    def test_separate_model_commands_and_resume(self):
        with tempfile.TemporaryDirectory(prefix='bbq-model-scheduling-') as folder:
            root = Path(folder)
            calls = []
            def fake_run(cache, config, req, case):
                existing = cache.get(req)
                if existing:
                    return existing
                calls.append((req['model'], req['stage']))
                choice = case['label'] if case['kind'] == 'bbq' else None
                return cache.add(req, 'ABC'[choice] if choice is not None else 'Helpful work.', choice, .001)
            with patch('audit.ROOT', root), patch('audit.load', return_value=(MANIFEST, FROZEN, PROTOCOL)), \
                 patch('audit.preflight'), patch('audit.unload'), patch.object(Cache, 'run', fake_run), \
                 patch('report.generate') as render:
                first, second = list(CONFIG['models'])
                with self.assertRaisesRegex(ValueError, 'prepare'):
                    audit.run(CONFIG, model=first)
                self.assertEqual(calls, [])
                audit.run(CONFIG, prepare_only=True)
                self.assertEqual(len(calls), 960)
                self.assertTrue(all(stage == 'development' for _, stage in calls))
                calls.clear()
                audit.run(CONFIG, model=first)
                self.assertEqual(len(calls), 6228)
                self.assertEqual({model for model, _ in calls}, {first})
                self.assertEqual(len(audit.model_cache(second, FROZEN, PROTOCOL).records), 480)
                self.assertEqual(render.call_count, 1)
                self.assertEqual(list(render.call_args.args[2]['models']), [first])
                calls.clear()
                audit.run(CONFIG, model=first)
                self.assertEqual(calls, [])
                calls.clear()
                audit.run(CONFIG, model=second)
                self.assertEqual(len(calls), 6228)
                self.assertEqual({model for model, _ in calls}, {second})
                self.assertEqual(list(render.call_args.args[2]['models']), list(CONFIG['models']))
                for model in CONFIG['models']:
                    self.assertEqual(len(audit.model_cache(model, FROZEN, PROTOCOL).records), 6708)

    def test_complete_report_and_review_rerender_without_inference(self):
        # Build the complete request plan with labeled synthetic responses. Score using
        # the actual pinned VADER/CPU encoder, but never put these artifacts in results/.
        with tempfile.TemporaryDirectory(prefix='bbq-verification-') as folder:
            root = Path(folder)
            (root / 'results').mkdir()
            cache = Cache(root / 'results' / 'responses.jsonl', PROTOCOL)
            for req, case in audit.plan('development', MANIFEST, CONFIG, FROZEN, PROTOCOL):
                cache.add(req, 'ABC'[case['label']], case['label'], .001)
            jobs = list(audit.plan('development', MANIFEST, CONFIG, FROZEN, PROTOCOL))
            selection = metrics.select_candidates([c for c in MANIFEST['bbq'] if c['split'] == 'dev'],
                                                  [cache.get(req) for req, _ in jobs], CONFIG)
            for stage in ('test', 'robustness', 'own'):
                for req, case in audit.plan(stage, MANIFEST, CONFIG, FROZEN, PROTOCOL, selection['selected']):
                    if case['kind'] == 'bbq':
                        choice = (case['target'] if case['polarity'] == 'neg' else
                                  next(i for i in range(3) if i not in (case['target'], case['unknown'])))
                        raw = 'ABC'[choice]
                    else:
                        choice, raw = None, f'They completed the work carefully. Their reliable effort was helpful. Seed {case["seed"]}.'
                    cache.add(req, raw, choice, .001)
            audit.verify_complete(cache, MANIFEST, CONFIG, FROZEN, PROTOCOL, selection)
            with patch('ollama.http', side_effect=AssertionError('Report attempted HTTP')):
                report.generate(root, MANIFEST, CONFIG, FROZEN, PROTOCOL, cache, selection)
                self.assertTrue((root / 'results' / 'report.md').exists())
                review_path = root / 'results' / 'error_review.csv'
                with review_path.open(newline='') as handle:
                    reader = csv.DictReader(handle)
                    fieldnames = reader.fieldnames
                    review = list(reader)
                self.assertEqual(len(review), 8)
                review[0].update(reviewed_by='Synthetic verification reviewer', reviewed_at='test',
                                 available_evidence='Synthetic note only', error_pattern='Synthetic explanation')
                with review_path.open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(review)
                with patch('report.compute', side_effect=AssertionError('Rerender rescored unchanged outputs')):
                    report.generate(root, MANIFEST, CONFIG, FROZEN, PROTOCOL, cache, selection)
                text = (root / 'results' / 'report.md').read_text()
                self.assertIn('1 of 8 sampled cases reviewed', text)
                self.assertIn('Synthetic note only', text)
                self.assertIn('Manual review pending', text)
                result = read_json(root / 'results' / 'metrics.json')
                self.assertTrue(all(abs(r['tone']['value']) < 1e-9 for r in result['own']['gaps']))
                self.assertTrue(all(abs(r['consistency']['value']) < 1e-5 for r in result['own']['gaps']))
                self.assertEqual(len(result['own']['groups']), 162)
                self.assertEqual(len(result['own']['gaps']), 162)
                self.assertEqual(len(result['own']['gap_changes']), 108)
                self.assertEqual(len(result['bbq']), 160)
                self.assertEqual(len(result['comparisons']), 80)
                with patch('audit.ROOT', root), patch('audit.load', return_value=(MANIFEST, FROZEN, PROTOCOL)):
                    for model in CONFIG['models']:
                        model_cache = audit.model_cache(model, FROZEN, PROTOCOL)
                        matching = [r for r in cache.records.values() if r['request']['model'] == model]
                        model_cache.path.write_text(''.join(canonical(row) + '\n' for row in matching))
                    before = {model: audit.model_output(model).joinpath('responses.jsonl').read_bytes()
                              for model in CONFIG['models']}
                    audit.run(CONFIG)
                    for model in CONFIG['models']:
                        output = audit.model_output(model)
                        self.assertEqual(before[model], (output / 'responses.jsonl').read_bytes())
                        self.assertTrue((output / 'tables' / 'bbq.csv').exists())
                        self.assertTrue((output / 'figures' / 'bbq_overview.png').exists())
                        self.assertTrue((output / 'figures' / 'bbq_overview.svg').exists())
                        self.assertTrue((output / 'report.md').exists())
                    with patch('audit.load', return_value=(MANIFEST, FROZEN, PROTOCOL)) as offline:
                        audit.run(CONFIG, model=next(iter(CONFIG['models'])), report_only=True)
                        offline.assert_called_once_with(CONFIG, online=False)



if __name__ == '__main__':
    unittest.main()
