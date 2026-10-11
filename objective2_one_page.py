from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Save PNG images without requiring a plot window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / 'objective2_figures'
FIGURES.mkdir(exist_ok=True)

# Exactly 8 pre-match explanatory variables per model.
match_X = ['home_prior_played', 'away_prior_played', 'home_prior_ppg',
           'away_prior_ppg', 'home_prior_gf_avg', 'away_prior_gf_avg',
           'home_prior_ga_avg', 'away_prior_ga_avg']
team_X = ['is_home', 'team_prior_played', 'team_prior_ppg',
          'opponent_prior_ppg', 'team_prior_gf_avg', 'team_prior_win_rate',
          'team_prior_clean_rate', 'team_last_points']
tasks = [('2.1', 'objective2_1_104_matches.csv', 'goal_difference', match_X, 104),
         ('2.2', 'objective2_2_208_team_matches.csv', 'goals_scored', team_X, 208)]


def save_figure(fig, filename):
    filepath = FIGURES / filename
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Saved chart:', filepath.name)


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)


results = []
for number, filename, target, predictors, expected_rows in tasks:
    path = ROOT / filename
    if not path.exists():
        raise FileNotFoundError(f'Missing {filename}; put it beside this Python script.')
    df = pd.read_csv(path).sort_values('match_id')
    assert len(df) == expected_rows and df.match_id.nunique() == 104
    assert len(predictors) == 8 and df[predictors + [target]].notna().all().all()
    if number == '2.2':
        assert df.groupby('match_id').size().eq(2).all()

    # Chronological 75/25 split. Both team rows of a game are kept together.
    train, test = df[df.match_id <= 78], df[df.match_id > 78]
    model = LinearRegression().fit(train[predictors], train[target])
    actual = test[target].to_numpy()
    predicted = model.predict(test[predictors])
    metrics = {'model': number, 'total_rows': len(df), 'train_rows': len(train),
               'test_rows': len(test), 'MAE': mean_absolute_error(actual, predicted),
               'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
               'R2': r2_score(actual, predicted)}
    results.append(metrics)
    print(f'\nMODEL {number}: {target} | {len(predictors)} pre-match predictors')
    # Format numeric values only: a text field such as "model" cannot be rounded.
    print(' | '.join(f'{k}: {v:.4f}' if isinstance(v, (float, np.floating))
                     else f'{k}: {v}' for k, v in metrics.items()))
    print('Coefficients:', dict(zip(predictors, model.coef_.round(3))))

    # Figure 1 and Figure 2: actual vs predicted test-set values.
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(actual, predicted, s=55, alpha=0.7, edgecolors='white',
               linewidths=0.6, label='Held-out observations')
    lo, hi = min(actual.min(), predicted.min()) - 0.5, max(actual.max(), predicted.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], '--', label='Perfect predictions')
    ax.set(xlim=(lo, hi), ylim=(lo, hi),
           xlabel=f'Actual {target.replace("_", " ")}',
           ylabel=f'Predicted {target.replace("_", " ")}',
           title=f'Model {number} | Actual vs predicted')
    ax.set_aspect('equal', adjustable='box')
    ax.text(0.03, 0.97, f'Test n = {len(test)}    MAE = {metrics["MAE"]:.3f}    R² = {metrics["R2"]:.3f}',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.9})
    ax.legend(loc='lower right', frameon=False)
    style(ax)
    fig.tight_layout()
    index = 1 if number == '2.1' else 2
    save_figure(fig, f'figure{index}_model_{number.replace(".", "_")}_actual_vs_predicted.png')

# Figure 3: held-out MAE and RMSE for both regression tasks.
fig, ax = plt.subplots(figsize=(7.5, 5.0))
x = np.arange(2)
for offset, metric in [(-0.17, 'MAE'), (0.17, 'RMSE')]:
    bars = ax.bar(x + offset, [r[metric] for r in results], width=0.34, label=metric)
    ax.bar_label(bars, fmt='%.3f', padding=3)
ax.set(xticks=x, xticklabels=['2.1: Goal difference', '2.2: Team goals'],
       ylabel='Test prediction error (goals)', title='Model comparison | Prediction error')
ax.set_ylim(0, max(r['RMSE'] for r in results) * 1.3)
ax.legend(frameon=False)
style(ax)
fig.tight_layout()
save_figure(fig, 'figure3_test_error_comparison.png')

# Figure 4: held-out R², including negative values where applicable.
fig, ax = plt.subplots(figsize=(7.5, 5.0))
scores = [r['R2'] for r in results]
bars = ax.bar(['2.1: Goal difference', '2.2: Team goals'], scores, width=0.5)
ax.axhline(0, ls='--', color='gray', lw=1)
for bar, score in zip(bars, scores):
    ax.annotate(f'{score:+.3f}', (bar.get_x() + bar.get_width() / 2, score),
                xytext=(0, 8 if score >= 0 else -18), textcoords='offset points', ha='center')
ax.set(ylabel='Test R² (higher is better)', title='Model comparison | Predictive performance')
ax.set_ylim(min(-0.25, min(scores) - 0.08), max(0.38, max(scores) + 0.08))
style(ax)
fig.tight_layout()
save_figure(fig, 'figure4_r_squared_comparison.png')

pd.DataFrame(results).round(4).to_csv(ROOT / 'objective2_model_results.csv', index=False)
print('\nDone: evaluation CSV and four 300-DPI PNG charts generated.')
# All predictors use earlier-match historical performance, never same-match stats.



#  py objective2_one_page.py