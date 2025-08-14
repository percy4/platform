import sys
import os
import yaml
import importlib
import inspect
from copy import deepcopy
from collections import OrderedDict


class Node:
    def __init__(self, opt):
        self.opt = opt

        self.name = opt["name"]
        self.kind = opt["kind"]
        self.envs = opt["envs"] if "envs" in opt else {}
        self.module = opt["module"] if "module" in opt else None
        self.accepted_into_platform = (
            opt["accepted_into_platform"] if "accepted_into_platform" in opt else False
        )
        self.outputs = opt["outputs"] if "outputs" in opt else {}

        self.is_header = True
        self.func = None
        self.input_args = []
        self.args_len = 0
        self.input_is_ready = []
        self.input_source = []
        self.output_target = {}
        self.is_over = False

    def include_module(self, root):
        if self.kind == "batch-disperse" or self.kind == "batch-gather":
            self.input_is_ready = [False]
            self.input_args = ["input"]
            self.args_len = 1
            self.input_source = [(None, None)]
            return
        sys.path.append(root)
        module = importlib.import_module(f"modules.{self.module}.entrypoint")
        self.func = module.main
        arg_spec = inspect.getfullargspec(self.func)
        args = arg_spec.args
        defaults = arg_spec.defaults
        args_len = len(args)
        if defaults is not None:
            defaults_len = len(defaults)
            self.input_is_ready = [False] * (args_len - defaults_len) + [
                True
            ] * defaults_len
        else:
            self.input_is_ready = [False] * args_len
        self.input_args = args
        self.args_len = args_len
        self.input_source = [(None, None)] * args_len

    def add_source(self, source_node_name, o_index, t_index):
        self.is_header = False
        self.input_is_ready[t_index] = False
        self.input_source[t_index] = (source_node_name, o_index)

    def add_target(self, target_node_name, o_index, t_index):
        if o_index in self.output_target:
            self.output_target[o_index].append((target_node_name, t_index))
        else:
            self.output_target[o_index] = [(target_node_name, t_index)]

    def is_ready(self):
        if self.is_over:
            return False
        return sum(self.input_is_ready) == self.args_len

    def source_ready(self, t_index):
        self.input_is_ready[t_index] = True

    def run(self, workspace):
        print(f"[INFO] Node {self.name} running...")
        input_dict = {}
        for inx, source_opt in enumerate(self.input_source):
            source_node_name, s_index = source_opt[0], source_opt[1]
            if source_node_name is not None:
                arg_name = self.input_args[inx]
                input_dict[arg_name] = deepcopy(workspace[source_node_name][s_index])
        for key, value in self.envs.items():
            os.environ[key] = value
        if self.kind == "normal":
            ret = self.func(**input_dict)
        elif self.kind == "in-batch":
            ret = []
            for arg_key, arg_value in input_dict.items():
                batch_args = {}
                for item in arg_value:
                    batch_args[arg_key] = item
                    ret_batch = self.func(**batch_args)
                    ret.append(ret_batch)
        elif self.kind == "batch-disperse":
            ret = input_dict["input"]
        elif self.kind == "batch-gather":
            ret = input_dict["input"]
        else:
            raise NotImplementedError(
                f"Node [{self.name}] kind [{self.kind}] not support!"
            )
        print(f"[INFO] Node {self.name} finished.")
        self.is_over = True
        return ret


def ordered_yaml():
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representor(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representor)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def yaml_load(filename):
    if os.path.isfile(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return yaml.load(f, Loader=ordered_yaml()[0])
    else:
        raise FileNotFoundError(f"{filename} file not found!")
