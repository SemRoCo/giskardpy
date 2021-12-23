import sys
from collections import defaultdict

import rospy

from giskardpy.tree.garden import TreeManager


def search_for(lines, function_name):
    data = [(x.split('giskardpy/src/giskardpy/tree/behaviors/')[1][:-3], lines[i - 1].split(' ')[2]) for i, x in enumerate(lines) if
            x.startswith('File') and function_name in lines[i + 6] and 'behavior' in x]
    result = defaultdict(dict)
    for file_name, time in data:
        result[file_name][function_name] = float(time)
    return result


def extract_data_from_profile(path):
    data = defaultdict(lambda: defaultdict(lambda: 'n/a'))
    with open(path, 'r') as f:
        profile = f.read()
    lines = profile.split('\n')
    keywords = ['__init__', 'setup', 'initialise', 'update']
    for function_name in keywords:
        new_data = search_for(lines, function_name)
        for file_name, function_data in new_data.items():
            data[file_name].update(function_data)
    for file_name, function_data in data.items():
        for function_name in keywords:
            data[file_name][function_name]
    return data


sys.argv.append('../test/test_keep_position3_3_no_collision.txt')
# sys.argv.append('../test/test_keep_position3_10.txt')

rospy.init_node('tests')
tree = TreeManager.from_param_server()
tree.render(extract_data_from_profile(sys.argv[1]))

# pydot.graph_from_dot_file(sys.argv[1])


pass
