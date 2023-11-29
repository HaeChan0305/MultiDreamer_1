# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import json
import os

from zoedepth.utils.easydict import EasyDict as edict

import pathlib
import platform

ROOT = pathlib.Path(__file__).parent.parent.resolve()

HOME_DIR = os.path.expanduser("~")

COMMON_CONFIG = {
    "save_dir": os.path.expanduser("~/shortcuts/monodepth3_checkpoints"),
    "project": "ZoeDepth",
    "tags": '',
    "notes": "",
    "gpu": None,
    "root": ".",
    "uid": None,
    "print_losses": False
}

COMMON_TRAINING_CONFIG = {
    "dataset": "nyu",
    "distributed": True,
    "workers": 16,
    "clip_grad": 0.1,
    "use_shared_dict": False,
    "shared_dict": None,
    "use_amp": False,

    "aug": True,
    "random_crop": False,
    "random_translate": False,
    "translate_prob": 0.2,
    "max_translation": 100,

    "validate_every": 0.25,
    "log_images_every": 0.1,
    "prefetch": False,
}

def infer_type(x):  # hacky way to infer type from string args
    if not isinstance(x, str):
        return x

    try:
        x = int(x)
        return x
    except ValueError:
        pass

    try:
        x = float(x)
        return x
    except ValueError:
        pass

    return x


def flatten(config, except_keys=('bin_conf')):
    def recurse(inp):
        if isinstance(inp, dict):
            for key, value in inp.items():
                if key in except_keys:
                    yield (key, value)
                if isinstance(value, dict):
                    yield from recurse(value)
                else:
                    yield (key, value)

    return dict(list(recurse(config)))



def parse_list(config, key, dtype=int):
    """Parse a list of values for the key if the value is a string. The values are separated by a comma. 
    Modifies the config in place.
    """
    if key in config:
        if isinstance(config[key], str):
            config[key] = list(map(dtype, config[key].split(',')))
        assert isinstance(config[key], list) and all([isinstance(e, dtype) for e in config[key]]
                                                     ), f"{key} should be a list of values dtype {dtype}. Given {config[key]} of type {type(config[key])} with values of type {[type(e) for e in config[key]]}."


def get_model_config(model_name, model_version=None):
    """Find and parse the .json config file for the model.

    Args:
        model_name (str): name of the model. The config file should be named config_{model_name}[_{model_version}].json under the models/{model_name} directory.
        model_version (str, optional): Specific config version. If specified config_{model_name}_{model_version}.json is searched for and used. Otherwise config_{model_name}.json is used. Defaults to None.

    Returns:
        easydict: the config dictionary for the model.
    """
    config_fname = f"config_{model_name}_{model_version}.json" if model_version is not None else f"config_{model_name}.json"
    config_file = os.path.join(ROOT, "models", model_name, config_fname)
    if not os.path.exists(config_file):
        return None

    with open(config_file, "r") as f:
        config = edict(json.load(f))

    # handle dictionary inheritance
    # only training config is supported for inheritance
    if "inherit" in config.train and config.train.inherit is not None:
        inherit_config = get_model_config(config.train["inherit"]).train
        for key, value in inherit_config.items():
            if key not in config.train:
                config.train[key] = value
    return edict(config)


def update_model_config(config, mode, model_name, model_version=None, strict=False):
    model_config = get_model_config(model_name, model_version)
    if model_config is not None:
        config = {**config, **
                  flatten({**model_config.model, **model_config[mode]})}
    elif strict:
        raise ValueError(f"Config file for model {model_name} not found.")
    return config

KEYS_TYPE_BOOL = ["use_amp", "distributed", "use_shared_dict", "same_lr", "aug", "three_phase",
                  "prefetch", "cycle_momentum"]  # Casting is not necessary as their int casted values in config are 0 or 1


def get_config():
    model_name = "zoedepth"
    mode = "infer"

    """Main entry point to get the config for the model.

    Args:
        model_name (str): name of the desired model.
        mode (str, optional): "train" or "infer". Defaults to 'train'.
        dataset (str, optional): If specified, the corresponding dataset configuration is loaded as well. Defaults to None.
    
    Keyword Args: key-value pairs of arguments to overwrite the default config.

    The order of precedence for overwriting the config is (Higher precedence first):
        # 1. overwrite_kwargs
        # 2. "config_version": Config file version if specified in overwrite_kwargs. The corresponding config loaded is config_{model_name}_{config_version}.json
        # 3. "version_name": Default Model version specific config specified in overwrite_kwargs. The corresponding config loaded is config_{model_name}_{version_name}.json
        # 4. common_config: Default config for all models specified in COMMON_CONFIG

    Returns:
        easydict: The config dictionary for the model.
    """
    config = flatten({**COMMON_CONFIG, **COMMON_TRAINING_CONFIG})
    config = update_model_config(config, mode, model_name)

    # Model specific post processing of config
    parse_list(config, "n_attractors")

    config['model'] = model_name
    typed_config = {k: infer_type(v) for k, v in config.items()}
    # add hostname to config
    config['hostname'] = platform.node()
    return edict(typed_config)
