import argparse
from os import path

from tools import run
from tools import pack


def parser_options():
    parser = argparse.ArgumentParser()
    sub_parser = parser.add_subparsers(dest='command')

    run_parser = sub_parser.add_parser('run', help='基于yaml配置执行实验')
    run_parser.add_argument('-config', type=str, default=path.join(root, 'experiment.yaml'),
                            help='配置文件路径')
    run_parser.add_argument('-log', type=str, default='',
                            help='执行日志文件路径，未配置则输出到控制台')

    pack_parser = sub_parser.add_parser('pack', help='项目打包')
    pack_parser.add_argument('-path', type=str, default=root,
                             help='项目路径')
    pack_parser.add_argument('-config', type=str, default=path.join(root, 'experiment.yaml'),
                             help='配置文件路径')

    args = parser.parse_args()

    if args.command == 'run':
        run(root, args)
    elif args.command == 'pack':
        pack(root, args)


if __name__ == '__main__':
    root = path.abspath(path.join(__file__, path.pardir))
    parser_options()
