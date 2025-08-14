import zipfile
import os
from os import path

from .util import yaml_load

required_files = ['entrypoint.py', 'module.yaml']


def pack(root, args):
    project_path = args.path
    if not path.isdir(project_path):
        raise NotADirectoryError('项目路径不存在')
    config = yaml_load(args.config)
    modules_list = get_modules_list(config)
    zip_path = path.join(root, path.basename(project_path) + '.zip')
    if path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(str(zip_path), 'w') as zf:
        zf.write(args.config, arcname=path.basename(args.config))
        modules_root = path.join(project_path, 'modules')
        zf.write(modules_root, arcname='modules')
        for module_name in modules_list:
            module_path = path.join(modules_root, module_name)
            if check_dir(module_path):
                tar_name = path.join('modules', module_name)
                zf.write(module_path, arcname=tar_name)
                for r_file in required_files:
                    arc_name = path.join(tar_name, r_file)
                    zf.write(path.join(module_path, r_file), arcname=arc_name)
            else:
                zf.close()
                os.remove(zip_path)
                raise FileNotFoundError(f'[Error] 模块 "{module_name}" 需要包含文件 '
                                        f'"entrypoint.py" 和 "module.yaml".')
        print('打包完成，包含文件:')
        print_tree(zf)


def get_modules_list(config):
    nodes_opt = config['workflow']['nodes']
    modules_set = set()
    for node_opt in nodes_opt:
        module_name = node_opt['module'] if 'module' in node_opt else None
        if module_name is not None:
            modules_set.add(module_name)
    return list(modules_set)


def check_dir(dir_path):
    return all(path.isfile(path.join(dir_path, file)) for file in required_files)


def print_tree(zipinfo, pre='', indent=0, visited=None):
    if visited is None:
        visited = set()
    for info in zipinfo.infolist():
        if info.filename.startswith(pre) and info.filename not in visited:
            visited.add(info.filename)
            relative_path = info.filename[len(pre):].lstrip(os.sep)
            if path.isdir(info.filename):
                print(' ' * indent + f'{relative_path}/')
                print_tree(zipinfo, pre=info.filename, indent=indent + 2, visited=visited)
            else:
                print(' ' * indent + relative_path)
