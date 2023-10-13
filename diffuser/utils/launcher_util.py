import inspect
import os
from copy import deepcopy

import numpy as np
from ml_logger import RUN, instr
from params_proto.neo_proto import ParamsProto

assert instr  # single-entry for the instrumentation thunk factory
RUN.project = "diffuser"
RUN.script_root = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
)
RUN.prefix = "logs/{exp_name}/{job_name}"


def discover_latest_checkpoint_path(checkpoint_dir):
    try:
        file_names = os.listdir(checkpoint_dir)
    except FileNotFoundError:
        return None
    steps = []
    for fname in file_names:
        if fname.startswith("state_") and fname.endswith(".pt"):
            steps.append(int(fname.split("_")[1].split(".")[0]))
    if len(steps) == 0:
        return None
    else:
        return os.path.join(checkpoint_dir, f"state_{max(steps)}.pt")


def build_config_from_dict(specs, Config=None):
    if Config is None:

        class Config(ParamsProto):
            pass

    for k, v in specs.items():
        setattr(Config, k, v)
    return Config


def check_exp_spec_format(specs):
    """
    Check that all keys are strings that don't contain '.'
    """
    for k, v in specs.items():
        if not isinstance(k, str):
            return False
        if "." in k:
            return False
        if isinstance(v, dict):
            sub_ok = check_exp_spec_format(v)
            if not sub_ok:
                return False
    return True


def flatten_dict(dic):
    """
    Assumes a potentially nested dictionary where all keys
    are strings that do not contain a '.'

    Returns a flat dict with keys having format:
    {'key.sub_key.sub_sub_key': ..., etc.}
    """
    new_dic = {}
    for k, v in dic.items():
        if isinstance(v, dict):
            sub_dict = flatten_dict(v)
            for sub_k, v in sub_dict.items():
                new_dic[".".join([k, sub_k])] = v
        else:
            new_dic[k] = v

    return new_dic


def add_variable_to_constant_specs(constants, flat_variables):
    new_dict = deepcopy(constants)
    for k, v in flat_variables.items():
        cur_sub_dict = new_dict
        split_k = k.split(".")
        for sub_key in split_k[:-1]:
            cur_sub_dict = cur_sub_dict[sub_key]
        cur_sub_dict[split_k[-1]] = v
    return new_dict


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class VariantDict(AttrDict):
    def __init__(self, d, hidden_keys):
        super(VariantDict, self).__init__(d)
        self._hidden_keys = hidden_keys

    def dump(self):
        return {k: v for k, v in self.items() if k not in self._hidden_keys}


class VariantGenerator(object):
    """
    Usage:

    vg = VariantGenerator()
    vg.add("param1", [1, 2, 3])
    vg.add("param2", ['x', 'y'])
    vg.variants() => # all combinations of [1,2,3] x ['x','y']

    Supports noncyclic dependency among parameters:
    vg = VariantGenerator()
    vg.add("param1", [1, 2, 3])
    vg.add("param2", lambda param1: [param1+1, param1+2])
    vg.variants() => # ..
    """

    def __init__(self):
        self._variants = []
        self._populate_variants()
        self._hidden_keys = []
        for k, vs, cfg in self._variants:
            if cfg.get("hide", False):
                self._hidden_keys.append(k)

    def add(self, key, vals, **kwargs):
        self._variants.append((key, vals, kwargs))

    def _populate_variants(self):
        methods = inspect.getmembers(
            self.__class__,
            predicate=lambda x: inspect.isfunction(x) or inspect.ismethod(x),
        )
        methods = [
            x[1].__get__(self, self.__class__)
            for x in methods
            if getattr(x[1], "__is_variant", False)
        ]
        for m in methods:
            self.add(m.__name__, m, **getattr(m, "__variant_config", dict()))

    def variants(self, randomized=False):
        ret = list(self.ivariants())
        if randomized:
            np.random.shuffle(ret)
        return list(map(self.variant_dict, ret))

    def variant_dict(self, variant):
        return VariantDict(variant, self._hidden_keys)

    def to_name_suffix(self, variant):
        suffix = []
        for k, vs, cfg in self._variants:
            if not cfg.get("hide", False):
                suffix.append(k + "_" + str(variant[k]))
        return "_".join(suffix)

    def ivariants(self):
        dependencies = list()
        for key, vals, _ in self._variants:
            if hasattr(vals, "__call__"):
                args = inspect.getargspec(vals).args
                if hasattr(vals, "im_self") or hasattr(vals, "__self__"):
                    # remove the first 'self' parameter
                    args = args[1:]
                dependencies.append((key, set(args)))
            else:
                dependencies.append((key, set()))
        sorted_keys = []
        # topo sort all nodes
        while len(sorted_keys) < len(self._variants):
            # get all nodes with zero in-degree
            free_nodes = [k for k, v in dependencies if len(v) == 0]
            if len(free_nodes) == 0:
                error_msg = "Invalid parameter dependency: \n"
                for k, v in dependencies:
                    if len(v) > 0:
                        error_msg += k + " depends on " + " & ".join(v) + "\n"
                raise ValueError(error_msg)
            dependencies = [(k, v) for k, v in dependencies if k not in free_nodes]
            # remove the free nodes from the remaining dependencies
            for _, v in dependencies:
                v.difference_update(free_nodes)
            sorted_keys += free_nodes
        return self._ivariants_sorted(sorted_keys)

    def _ivariants_sorted(self, sorted_keys):
        if len(sorted_keys) == 0:
            yield dict()
        else:
            first_keys = sorted_keys[:-1]
            first_variants = self._ivariants_sorted(first_keys)
            last_key = sorted_keys[-1]
            last_vals = [v for k, v, _ in self._variants if k == last_key][0]
            if hasattr(last_vals, "__call__"):
                last_val_keys = inspect.getargspec(last_vals).args
                if hasattr(last_vals, "im_self") or hasattr(last_vals, "__self__"):
                    last_val_keys = last_val_keys[1:]
            else:
                last_val_keys = None
            for variant in first_variants:
                if hasattr(last_vals, "__call__"):
                    last_variants = last_vals(**{k: variant[k] for k in last_val_keys})
                    for last_choice in last_variants:
                        yield AttrDict(variant, **{last_key: last_choice})
                else:
                    for last_choice in last_vals:
                        yield AttrDict(variant, **{last_key: last_choice})


def build_nested_variant_generator(exp_spec):
    assert check_exp_spec_format(exp_spec)
    # from rllab.misc.instrument import VariantGenerator

    variables = exp_spec["variables"]
    constants = exp_spec["constants"]

    # check if we're effectively just running a single experiment
    if variables is None:

        def vg_fn():
            dict_to_yield = constants
            dict_to_yield.update(exp_spec["meta_data"])
            yield dict_to_yield

        return vg_fn

    variables = flatten_dict(variables)
    vg = VariantGenerator()
    for k, v in variables.items():
        vg.add(k, v)

    def vg_fn():
        for flat_variables in vg.variants():
            dict_to_yield = add_variable_to_constant_specs(constants, flat_variables)
            dict_to_yield.update(exp_spec["meta_data"])
            del dict_to_yield["_hidden_keys"]
            yield dict_to_yield

    return vg_fn
