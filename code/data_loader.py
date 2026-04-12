# data_loader.py  — fixed: added missing imports and DATA_DIR definition
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

def load_ucr_dataset(name):
    """Load a UCR dataset by name from the local data directory."""
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    files = [f for f in os.listdir(path) if f.endswith('.ts')]
    return files

def parse_ts_file(filepath):
    """Parse a .ts file from the UCR archive format."""
    X, y = [], []
    with open(filepath, 'r') as f:
        in_data = False
        for line in f:
            line = line.strip()
            if line.lower() == '@data':
                in_data = True
                continue
            if in_data and line and not line.startswith('#'):
                # UCR .ts files use colon to separate data from label
                # e.g.: -0.64,-0.63,...,-0.64:2
                if ':' in line and ',' in line:
                    data_part, label = line.rsplit(':', 1)  # split on LAST colon
                    vals = [float(v) for v in data_part.split(',')]
                elif ',' in line:
                    parts = line.split(',')
                    label = parts[-1].strip()
                    vals = [float(v) for v in parts[:-1]]
                else:
                    continue
                X.append(vals)
                y.append(label.strip())
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    min_len = min(len(r) for r in X)
    return np.array([r[:min_len] for r in X]), y_enc
