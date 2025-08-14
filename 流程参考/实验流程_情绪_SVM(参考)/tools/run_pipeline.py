import sys
from os import path

from .util import yaml_load, Node


def run(root, args):
    if not path.exists(args.config):
        raise FileNotFoundError(f'{args.config} 配置文件不存在.')

    if args.log != '':
        f = open(args.log, 'w')
        print(f'[INFO] 指定日志输出文件{args.log}')
        sys.stdout = f
        sys.stderr = f

    config = yaml_load(args.config)
    dag = build_dag(root, config)

    workspace = {}
    for node in dag.values():
        if node.is_ready():
            solve(workspace, dag, node)


def build_dag(root, config):
    workflow = config['workflow']
    print(f"[INFO] Experiment name: {workflow['name']}")

    nodes_opt = workflow['nodes']
    nodes_dict = {}
    for node_opt in nodes_opt:
        node = Node(node_opt)
        node.include_module(root)
        nodes_dict[node.name] = node
    for node_name, node in nodes_dict.items():
        node_outputs = node.outputs
        for output_opt in node_outputs.items():
            output_index = output_opt[0]
            o_index = int(output_index)
            for target_opt in output_opt[1]:
                target_node_name, target_index = target_opt.split(':')
                t_index = int(target_index)
                target_node = nodes_dict[target_node_name]
                target_node.add_source(node_name, o_index, t_index)
                node.add_target(target_node_name, o_index, t_index)

    print('[INFO] Build DAG success.')
    return nodes_dict


def solve(workspace, dag, node):
    workspace[node.name] = {}
    node_space = workspace[node.name]
    node_output = node.run(workspace)

    node_wait_queue = set()
    for o_index, o_list in node.output_target.items():
        node_space[o_index] = node_output[o_index]
        for target_opt in o_list:
            target_node_name, target_index = target_opt[0], target_opt[1]
            target_node = dag[target_node_name]
            target_node.source_ready(target_index)
            node_wait_queue.add(target_node_name)

    for target_node_name in node_wait_queue:
        target_node = dag[target_node_name]
        if target_node.is_ready():
            solve(workspace, dag, target_node)
