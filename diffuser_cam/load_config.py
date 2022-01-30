import json

def load_config(config_fname):
    with open(config_fname) as f:
        config = json.load(f)
    return config