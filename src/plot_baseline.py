"""Create the checkpoint cost figure from saved paired-bootstrap summaries."""
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'work/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

SUMMARY = ROOT / 'reports/class_weight/comparison/summary.json'


def build_figure():
    """Plot four held-out decision rules; never load raw data or fit a model."""
    summary = json.loads(SUMMARY.read_text())
    assert summary['status'] == 'SUCCESS' and summary['sealed_test_accessed'] is False
    rows = [
        ('unweighted_default_mean_cost', 'Unweighted logistic regression\nThreshold 0.5', '#2864AD'),
        ('balanced_default_mean_cost', 'Balanced class weights\nThreshold 0.5', '#D66B35'),
        ('unweighted_tuned_mean_cost', 'Unweighted logistic regression\nInner-CV cost threshold', '#2864AD'),
        ('balanced_tuned_mean_cost', 'Balanced class weights\nInner-CV cost threshold', '#D66B35'),
    ]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.labelcolor': '#263442', 'text.color': '#172B3A'})
    fig = plt.figure(figsize=(12, 7.5), facecolor='white')
    ax = fig.add_axes([.285, .34, .405, .45])
    fig.text(.045, .935, 'Credit card fraud: cost at a 50:1 loss ratio',
             fontsize=20, weight='bold')
    fig.text(.045, .888, 'Nested 5-fold CV on development data | 220,530 transactions | Final test set not used',
             fontsize=10.5, color='#536473')
    ax.axhspan(1.5, 3.5, color='#F3F6F9', zorder=0)
    ax.axhline(1.5, color='#D5DEE5', linewidth=.8)
    transform = blended_transform_factory(ax.transAxes, ax.transData)
    for y, (key, label, color) in zip([3, 2, 1, 0], rows):
        result = summary['metrics'][key]
        value = result['point_estimate'] * 1000
        low, high = [x * 1000 for x in result['conditional_95_percentile_interval']]
        assert low <= value <= high
        ax.errorbar(value, y, xerr=[[value - low], [high - value]],
                    fmt='o', color=color, markersize=8, capsize=5,
                    elinewidth=2, markeredgecolor='white', markeredgewidth=.8, zorder=3)
        ax.text(1.075, y, f'{value:.2f}  [{low:.2f}, {high:.2f}]',
                transform=transform, va='center', fontsize=11)
    ax.text(1.075, 1.09, 'Mean [95% interval]', transform=ax.transAxes,
            fontsize=10, weight='bold')
    ax.set_yticks([3, 2, 1, 0], [row[1] for row in rows])
    ax.tick_params(axis='y', length=0, pad=12, labelsize=10)
    ax.set_ylim(-.5, 3.5)
    ax.set_xlim(0, 48)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_xlabel('Relative cost units per 1,000 transactions\nLower is better', labelpad=10, fontsize=10)
    ax.grid(axis='x', color='#D7E0E7', linewidth=.7)
    ax.set_axisbelow(True)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#A3B1BC')
    handles = [Line2D([0], [0], marker='o', linestyle='', color=color, markersize=7, label=label)
               for color, label in [('#2864AD', 'Unweighted'), ('#D66B35', 'Balanced weights')]]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.285, .85),
               ncol=2, frameon=False, fontsize=10, borderaxespad=0)
    delta = summary['metrics']['delta_tuned_mean_cost']
    low, high = [x * 1000 for x in delta['conditional_95_percentile_interval']]
    value = delta['point_estimate'] * 1000
    fig.text(.045, .16,
             f'After threshold tuning, balanced minus unweighted: {value:+.2f} units / 1,000\n'
             f'95% paired interval [{low:+.2f}, {high:+.2f}]; no clear cost difference in this conditional analysis.',
             fontsize=11, linespacing=1.6)
    fig.text(.045, .045,
             'Cost = (FP + 50 x FN) / N. Points are outer-fold means; bars are conditional 95% percentile intervals.\n'
             '5,000 paired bootstrap resamples within fold and class, seed 42. Models and thresholds are fixed;\n'
             'intervals omit refitting variability, cross-fold training dependence, and prevalence shifts.',
             fontsize=9, color='#536473', linespacing=1.5)
    return fig


def save_figure(path=None):
    path = Path(path) if path is not None else ROOT / 'baseline.png'
    fig = build_figure()
    fig.savefig(path, dpi=250, facecolor='white')
    return fig, path


if __name__ == '__main__':
    figure, destination = save_figure()
    plt.close(figure)
    print(f'Saved {destination}')
