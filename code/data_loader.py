# data_loader.py  — first attempt at loading UCR datasets
# TODO: forgot to import os, DATA_DIR not defined yet

def load_ucr_dataset(name):
    path = DATA_DIR + "/" + name   # NameError: DATA_DIR not defined
    files = os.listdir(path)       # NameError: os not imported
    return files

def parse_ts_file(filepath):
    X, y = [], []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split(',')
            X.append(parts[:-1])
            y.append(parts[-1])
    return X, y
