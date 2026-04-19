"""
generate_figures.py
===================
CSCE 5222 – Group 18
Generates all figures for the project report from the saved results JSON.
All plots are created from scratch (no screenshots or copied figures).
"""

import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns

RES_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(FIG_DIR, exist_ok=True)

with open(os.path.join(RES_DIR, 'all_results.json')) as f:
    results = json.load(f)

clf_results  = results['classification']
fore_results = results['forecasting']
anom_results = results['anomaly_detection']

COLORS = {
    'raw':  '#4878CF',
    'c22':  '#6ACC65',
    'shap': '#D65F5F',
    'emb':  '#B47CC7',
}
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.dpi': 200,
})

# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 – Classification accuracy bar chart (all 5 UCR datasets)
# ─────────────────────────────────────────────────────────────────────────────
def fig_classification_accuracy():
    datasets = [r['dataset'] for r in clf_results]
    raw_acc  = [r['raw_acc']  for r in clf_results]
    c22_acc  = [r['c22_acc']  for r in clf_results]
    shap_acc = [r['shap_acc'] for r in clf_results]

    x = np.arange(len(datasets))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 4))
    b1 = ax.bar(x - w,   raw_acc,  w, label='Raw Features',      color=COLORS['raw'],  edgecolor='white', linewidth=0.5)
    b2 = ax.bar(x,       c22_acc,  w, label='Catch22',           color=COLORS['c22'],  edgecolor='white', linewidth=0.5)
    b3 = ax.bar(x + w,   shap_acc, w, label='Catch22 + SHAP',    color=COLORS['shap'], edgecolor='white', linewidth=0.5)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=7.5, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title('Classification Accuracy: Raw vs. Catch22 vs. Catch22+SHAP (UCR Datasets)')
    ax.set_ylim(0, 1.12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig1_classification_accuracy.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 – Classification F1-score bar chart
# ─────────────────────────────────────────────────────────────────────────────
def fig_classification_f1():
    datasets = [r['dataset'] for r in clf_results]
    raw_f1   = [r['raw_f1']  for r in clf_results]
    c22_f1   = [r['c22_f1']  for r in clf_results]
    shap_f1  = [r['shap_f1'] for r in clf_results]

    x = np.arange(len(datasets))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w, raw_f1,  w, label='Raw Features',   color=COLORS['raw'],  edgecolor='white')
    ax.bar(x,     c22_f1,  w, label='Catch22',         color=COLORS['c22'],  edgecolor='white')
    ax.bar(x + w, shap_f1, w, label='Catch22 + SHAP',  color=COLORS['shap'], edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha='right')
    ax.set_ylabel('Weighted F1-Score')
    ax.set_title('Weighted F1-Score: Raw vs. Catch22 vs. Catch22+SHAP (UCR Datasets)')
    ax.set_ylim(0, 1.10)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig2_classification_f1.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 – SHAP feature importance (GunPoint dataset, most interesting)
# ─────────────────────────────────────────────────────────────────────────────
def fig_shap_importance():
    # Use GunPoint (first dataset)
    r = clf_results[0]
    feat_names   = r['feat_names']
    shap_imp     = np.array(r['shap_importance'])
    order        = np.argsort(shap_imp)[::-1]
    sorted_names = [feat_names[i] for i in order]
    sorted_vals  = shap_imp[order]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(range(len(sorted_names)), sorted_vals[::-1],
                   color=plt.cm.RdYlGn(np.linspace(0.2, 0.85, len(sorted_names))),
                   edgecolor='white', linewidth=0.4)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names[::-1], fontsize=9)
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title('SHAP Feature Importance – GunPoint Dataset (Catch22 Features)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=0.4)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig3_shap_importance.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 – Forecasting MSE/MAE comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────
def fig_forecasting_metrics():
    methods = ['Raw\n(lag-1)', 'Lag+\nRolling', 'Lag+Roll\n+Embed', 'Lag+Roll\n+SHAP']
    mse_vals = [fore_results['raw_mse'], fore_results['lr_mse'],
                fore_results['emb_mse'], fore_results['shap_mse']]
    mae_vals = [fore_results['raw_mae'], fore_results['lr_mae'],
                fore_results['emb_mae'], fore_results['shap_mae']]

    x = np.arange(len(methods))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    colors = [COLORS['raw'], COLORS['c22'], COLORS['emb'], COLORS['shap']]
    for ax, vals, metric in zip(axes, [mse_vals, mae_vals], ['MSE', 'MAE']):
        bars = ax.bar(x, vals, color=colors, edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=9)
        ax.set_ylabel(metric)
        ax.set_title(f'ETTh1 Forecasting – {metric} by Method')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.4)

    plt.suptitle('ETTh1 Oil Temperature Forecasting: Method Comparison', fontsize=11, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig4_forecasting_metrics.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 – Forecasting: Actual vs. Predicted (Lag+Rolling method, first 200 pts)
# ─────────────────────────────────────────────────────────────────────────────
def fig_forecast_vs_actual():
    y_te      = np.array(fore_results['y_te'])[:200]
    y_pred_lr = np.array(fore_results['y_pred_lr'])[:200]
    y_pred_emb= np.array(fore_results['y_pred_emb'])[:200]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_te,       label='Ground Truth',       color='#333333', linewidth=1.5, zorder=3)
    ax.plot(y_pred_lr,  label='Lag+Rolling (LGB)',  color=COLORS['c22'],  linewidth=1.2, linestyle='--', alpha=0.85)
    ax.plot(y_pred_emb, label='Lag+Roll+Embed',     color=COLORS['emb'],  linewidth=1.2, linestyle=':',  alpha=0.85)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Oil Temperature (°C)')
    ax.set_title('ETTh1 Forecasting: Ground Truth vs. Predicted (First 200 Test Points)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig5_forecast_vs_actual.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 6 – Anomaly detection: PPG signal with anomaly overlay
# ─────────────────────────────────────────────────────────────────────────────
def fig_anomaly_signal():
    bvp    = np.array(anom_results['bvp'])
    labels = np.array(anom_results['labels'])

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(bvp, color='#2c7bb6', linewidth=0.8, label='BVP Signal')

    # Shade anomaly regions
    in_anom = False
    start   = 0
    for i, l in enumerate(labels):
        if l == 1 and not in_anom:
            start   = i
            in_anom = True
        elif l == 0 and in_anom:
            ax.axvspan(start, i, color='#d7191c', alpha=0.25)
            in_anom = False
    if in_anom:
        ax.axvspan(start, len(labels), color='#d7191c', alpha=0.25)

    normal_patch = mpatches.Patch(color='#2c7bb6', label='Normal BVP')
    anom_patch   = mpatches.Patch(color='#d7191c', alpha=0.4, label='Anomaly Region')
    ax.legend(handles=[normal_patch, anom_patch], fontsize=9)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('BVP Amplitude')
    ax.set_title('WESAD-like PPG Signal with Annotated Anomaly Regions')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig6_anomaly_signal.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 7 – Anomaly detection F1 and AUC comparison
# ─────────────────────────────────────────────────────────────────────────────
def fig_anomaly_metrics():
    methods  = ['Raw Windows', 'Catch22', 'Catch22+SHAP']
    f1_vals  = [anom_results['raw_f1'],  anom_results['c22_f1'],  anom_results['shap_f1']]
    auc_vals = [anom_results['raw_auc'], anom_results['c22_auc'], anom_results['shap_auc']]

    x = np.arange(len(methods))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    colors = [COLORS['raw'], COLORS['c22'], COLORS['shap']]

    for ax, vals, metric in zip(axes, [f1_vals, auc_vals], ['F1-Score', 'ROC-AUC']):
        bars = ax.bar(x, vals, color=colors, edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=9)
        ax.set_ylabel(metric)
        ax.set_title(f'Anomaly Detection – {metric}')
        ax.set_ylim(0, 1.15)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.4)

    plt.suptitle('WESAD-like PPG Anomaly Detection: Method Comparison', fontsize=11, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig7_anomaly_metrics.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 8 – Catch22 feature correlation heatmap (GunPoint)
# ─────────────────────────────────────────────────────────────────────────────
def fig_feature_correlation():
    # Re-extract Catch22 features for GunPoint
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from feature_engineering_pipeline import extract_catch22_matrix, _parse_ts_file
    import glob

    ts_files = glob.glob(os.path.join(DATA_DIR, 'GunPoint', '*.ts'))
    train_f  = [f for f in ts_files if 'TRAIN' in f.upper()]
    if not train_f:
        print("  Skipping correlation heatmap (no GunPoint data)")
        return

    X_tr, _ = _parse_ts_file(train_f[0])
    X_c22, feat_names = extract_catch22_matrix(X_tr)
    df_feats = pd.DataFrame(X_c22, columns=feat_names)
    corr = df_feats.corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, linewidths=0.3, ax=ax,
                cbar_kws={'shrink': 0.8})
    ax.set_title('Catch22 Feature Correlation Matrix – GunPoint Dataset')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0,  labelsize=8)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig8_feature_correlation.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 9 – PCA projection of Catch22 features (SyntheticControl, 6 classes)
# ─────────────────────────────────────────────────────────────────────────────
def fig_pca_projection():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from feature_engineering_pipeline import extract_catch22_matrix, _parse_ts_file
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import glob

    ts_files = glob.glob(os.path.join(DATA_DIR, 'SyntheticControl', '*.ts'))
    train_f  = [f for f in ts_files if 'TRAIN' in f.upper()]
    if not train_f:
        print("  Skipping PCA projection (no SyntheticControl data)")
        return

    X_tr, y_tr = _parse_ts_file(train_f[0])
    X_c22, _   = extract_catch22_matrix(X_tr)
    X_scaled   = StandardScaler().fit_transform(X_c22)
    pca        = PCA(n_components=2, random_state=42)
    X_2d       = pca.fit_transform(X_scaled)

    class_names = ['Normal', 'Cyclic', 'Increasing', 'Decreasing', 'Upward Shift', 'Downward Shift']
    palette = sns.color_palette('tab10', n_colors=len(np.unique(y_tr)))

    fig, ax = plt.subplots(figsize=(7, 5))
    for cls_idx in np.unique(y_tr):
        mask = y_tr == cls_idx
        label = class_names[cls_idx] if cls_idx < len(class_names) else f'Class {cls_idx}'
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   color=palette[cls_idx], label=label, s=40, alpha=0.75, edgecolors='white', linewidth=0.3)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)')
    ax.set_title('PCA Projection of Catch22 Features – SyntheticControl Dataset')
    ax.legend(fontsize=8, loc='best', framealpha=0.7)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig9_pca_projection.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 10 – Forecasting SHAP feature importance (ETT)
# ─────────────────────────────────────────────────────────────────────────────
def fig_forecasting_shap():
    feat_names = fore_results['feat_names']
    shap_imp   = np.array(fore_results['shap_importance'])
    order      = np.argsort(shap_imp)[::-1][:15]  # top 15
    sorted_names = [feat_names[i] for i in order]
    sorted_vals  = shap_imp[order]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(sorted_names)))[::-1]
    ax.barh(range(len(sorted_names)), sorted_vals[::-1], color=colors[::-1],
            edgecolor='white', linewidth=0.4)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names[::-1], fontsize=9)
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title('Top 15 SHAP Feature Importances – ETTh1 Forecasting')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=0.4)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig10_forecasting_shap.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 11 – Pipeline methodology flowchart (replaces incomplete slide 10)
# ─────────────────────────────────────────────────────────────────────────────
def fig_pipeline_flowchart():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    box_style = dict(boxstyle='round,pad=0.4', facecolor='#dce9f5', edgecolor='#2c7bb6', linewidth=1.5)
    arrow_props = dict(arrowstyle='->', color='#2c7bb6', lw=1.5)
    decision_style = dict(boxstyle='round,pad=0.4', facecolor='#fff2cc', edgecolor='#d6a800', linewidth=1.5)

    steps = [
        (5, 9.2, 'Raw Time-Series Data\n(UCR / ETTh1 / WESAD-like)', box_style),
        (5, 7.8, 'Preprocessing\n(Normalization, Missing Value Handling)', box_style),
        (2, 6.2, 'Feature Generation\nCatch22 + Lag + Rolling Stats', box_style),
        (5, 6.2, 'Self-Supervised\nEmbeddings (MLP)', box_style),
        (8, 6.2, 'Raw Features\n(Baseline)', box_style),
        (5, 4.6, 'Combined Feature Matrix', box_style),
        (5, 3.2, 'SHAP-Guided\nFeature Pruning', decision_style),
        (2, 1.8, 'Classification\n(LightGBM)', box_style),
        (5, 1.8, 'Forecasting\n(LightGBM Regressor)', box_style),
        (8, 1.8, 'Anomaly Detection\n(Isolation Forest)', box_style),
        (5, 0.4, 'Evaluation & Analysis\n(Acc / F1 / MSE / MAE / AUC)', box_style),
    ]

    for x, y, text, style in steps:
        ax.text(x, y, text, ha='center', va='center', fontsize=8.5,
                bbox=style, zorder=3)

    arrows = [
        ((5, 8.9), (5, 8.1)),
        ((5, 7.5), (2, 6.6)),
        ((5, 7.5), (5, 6.6)),
        ((5, 7.5), (8, 6.6)),
        ((2, 5.8), (5, 5.0)),
        ((5, 5.8), (5, 5.0)),
        ((8, 5.8), (5, 5.0)),
        ((5, 4.3), (5, 3.6)),
        ((5, 2.9), (2, 2.1)),
        ((5, 2.9), (5, 2.1)),
        ((5, 2.9), (8, 2.1)),
        ((2, 1.5), (5, 0.7)),
        ((5, 1.5), (5, 0.7)),
        ((8, 1.5), (5, 0.7)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=arrow_props, zorder=2)

    ax.set_title('Feature Engineering Pipeline – Group 18', fontsize=12, pad=10)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig11_pipeline_flowchart.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 12 – Summary table heatmap (all tasks, all methods)
# ─────────────────────────────────────────────────────────────────────────────
def fig_summary_heatmap():
    # Build a compact summary table
    data = {
        'Task': ['Classification\n(avg Acc)', 'Classification\n(avg F1)',
                 'Forecasting\n(MSE)', 'Forecasting\n(MAE)',
                 'Anomaly Det.\n(F1)', 'Anomaly Det.\n(AUC)'],
        'Raw Baseline': [
            np.mean([r['raw_acc'] for r in clf_results]),
            np.mean([r['raw_f1']  for r in clf_results]),
            fore_results['raw_mse'], fore_results['raw_mae'],
            anom_results['raw_f1'],  anom_results['raw_auc'],
        ],
        'Catch22 / Lag+Roll': [
            np.mean([r['c22_acc'] for r in clf_results]),
            np.mean([r['c22_f1']  for r in clf_results]),
            fore_results['lr_mse'], fore_results['lr_mae'],
            anom_results['c22_f1'], anom_results['c22_auc'],
        ],
        'Catch22+SHAP / Lag+Roll+SHAP': [
            np.mean([r['shap_acc'] for r in clf_results]),
            np.mean([r['shap_f1']  for r in clf_results]),
            fore_results['shap_mse'], fore_results['shap_mae'],
            anom_results['shap_f1'],  anom_results['shap_auc'],
        ],
    }
    df = pd.DataFrame(data).set_index('Task')

    # For MSE/MAE lower is better; invert for heatmap coloring
    df_display = df.copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(df_display, annot=True, fmt='.4f', cmap='YlGn',
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Metric Value'})
    ax.set_title('Summary of Results Across All Tasks and Methods', fontsize=11)
    ax.tick_params(axis='x', rotation=15, labelsize=9)
    ax.tick_params(axis='y', rotation=0,  labelsize=9)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig12_summary_heatmap.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating all figures...")
    fig_classification_accuracy()
    fig_classification_f1()
    fig_shap_importance()
    fig_forecasting_metrics()
    fig_forecast_vs_actual()
    fig_anomaly_signal()
    fig_anomaly_metrics()
    fig_feature_correlation()
    fig_pca_projection()
    fig_forecasting_shap()
    fig_pipeline_flowchart()
    fig_summary_heatmap()
    print("\nAll figures generated successfully.")
