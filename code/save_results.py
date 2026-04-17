# save_results.py — serialize all experiment results to JSON
import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')

def save_all_results(clf_results, fore_results, anom_results):
    """Save all experimental results to a JSON file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {
        'classification': clf_results,
        'forecasting': fore_results,
        'anomaly_detection': anom_results,
    }
    out_path = os.path.join(RESULTS_DIR, 'all_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {out_path}")
    return out_path
