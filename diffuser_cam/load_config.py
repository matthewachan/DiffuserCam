import json

# TODO(mchan): Create a top-level class to hold state (i.e. configs). This should help to standardize the function signatures for all reconstruction algorithms.
def load_config(config_fname):
    with open(config_fname) as f:
        config = json.load(f)
    return config