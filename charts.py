"""Export numeric CSV tables and static figures from the scored results only."""
import csv
from pathlib import Path


def write_table(path, rows):
    flat = []
    for row in rows:
        record = {}
        for key, value in row.items():
            if isinstance(value, dict):
                record.update({f'{key}_{name}': number for name, number in value.items()})
            else:
                record[key] = value
        flat.append(record)
    fields = list(dict.fromkeys(key for row in flat for key in row))
    temp = path.with_suffix('.csv.tmp')
    with temp.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    temp.replace(path)


def export(output, metrics):
    """No inference or rescoring: figures and CSVs share the exact metrics.json values."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = Path(output)
    tables, figures = output / 'tables', output / 'figures'
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    for name, rows in {'bbq': metrics['bbq'], 'ablations': metrics['comparisons'],
                       'own_groups': metrics['own']['groups'], 'own_gaps': metrics['own']['gaps'],
                       'own_gap_changes': metrics['own']['gap_changes']}.items():
        write_table(tables / f'{name}.csv', rows)

    def points(ax, rows, key, labels, scale=1, color='#2f6b9a'):
        for x, row in enumerate(rows):
            value = row[key]
            if value['value'] is None:
                ax.text(x, 0, 'undefined', rotation=90, va='bottom', ha='center', fontsize=8)
                continue
            ax.plot(x, value['value'] * scale, 'o', color=color, markersize=6)
            if value['low'] is not None:
                ax.vlines(x, value['low'] * scale, value['high'] * scale, color=color, linewidth=1.8)
            else:
                ax.annotate('CI undefined', (x, value['value'] * scale), xytext=(4, 7),
                            textcoords='offset points', fontsize=7, rotation=25)
        ax.set_xticks(range(len(rows)), labels, rotation=35, ha='right', fontsize=8)
        ax.set_xlim(-.6, max(len(rows) - .4, .6))
        ax.axhline(0, color='#777777', linewidth=.7)
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)

    def save(fig, name):
        fig.text(.01, .005, 'Points: estimates; lines: paired 95% bootstrap intervals. Undefined intervals are labeled.', fontsize=8)
        fig.tight_layout(rect=(0, .04, 1, 1))
        for suffix in ('png', 'svg'):
            fig.savefig(figures / f'{name}.{suffix}', dpi=180)
        plt.close(fig)

    models = list(dict.fromkeys(row['model'] for row in metrics['bbq']))
    fig, axes = plt.subplots(len(models), 2, figsize=(12, 4.5 * len(models)), squeeze=False)
    for axes_row, model in zip(axes, models):
        rows = [r for r in metrics['bbq'] if r['scope'] == 'test' and r['category'] == 'overall' and r['model'] == model]
        for ax, key, title, scale in zip(axes_row, ('strict_accuracy', 'ambig_bias'),
                                          ('Overall accuracy (invalid = incorrect), %', 'Signed ambiguous bias'), (100, 1)):
            points(ax, rows, key, [r['condition'] for r in rows], scale)
            ax.set_title(f'{model}\n{title}')
    save(fig, 'bbq_overview')

    fig, axes = plt.subplots(len(models), 2, figsize=(12, 4.5 * len(models)), squeeze=False)
    for axes_row, model in zip(axes, models):
        rows = [r for r in metrics['comparisons'] if r['scope'] == 'test' and r['category'] == 'overall' and r['model'] == model]
        for ax, key, title in zip(axes_row, ('ambig_reduction', 'strict_accuracy_loss_pp'),
                                  ('Absolute ambiguous-bias reduction vs B1', 'Accuracy loss vs B1 (strict), pp')):
            points(ax, rows, key, [r['condition'] for r in rows])
            if key.endswith('_pp'):
                ax.axhline(3, color='#ba5a31', linestyle='--', label='3 pp utility limit')
                ax.legend(fontsize=8)
            ax.set_title(f'{model}\n{title}')
    save(fig, 'mitigation_tradeoff')

    fig, axes = plt.subplots(len(models), 1, figsize=(12, 4.5 * len(models)), squeeze=False)
    for (ax,), model in zip(axes, models):
        rows = [r for r in metrics['comparisons'] if r['scope'] == 'robustness' and r['category'] == 'overall' and r['model'] == model]
        points(ax, rows, 'ambig_reduction', [f'{r["condition"]} / wording {r["wording"]}' for r in rows])
        ax.set_title(f'{model}: bias reduction on the same 320 items')
    save(fig, 'paraphrase_robustness')

    fig, axes = plt.subplots(len(models), 1, figsize=(15, 5 * len(models)), squeeze=False)
    for (ax,), model in zip(axes, models):
        rows = [r for r in metrics['own']['gaps'] if r['model'] == model and r['wording'] == 0]
        labels = [f'{r["condition"]}: {r["left"]} − {r["right"]}' for r in rows]
        points(ax, rows, 'tone', labels)
        ax.set_title(f'{model}: original-wording sentiment gaps (left minus right)')
    save(fig, 'sentiment_gaps')
    return ['figures/' + name + '.png' for name in
            ('bbq_overview', 'mitigation_tradeoff', 'paraphrase_robustness', 'sentiment_gaps')]
