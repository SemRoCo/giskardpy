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
    data = defaultdict(dict)
    with open(path, 'r') as f:
        profile = f.read()
    lines = profile.split('\n')
    setups = search_for(lines, 'setup')
    for file_name, function_data in setups.items():
        data[file_name].update(function_data)
    updates = search_for(lines, 'update')
    for file_name, function_data in updates.items():
        data[file_name].update(function_data)
    initialise = search_for(lines, 'initialise')
    for file_name, function_data in initialise.items():
        data[file_name].update(function_data)
    return data


sys.argv.append('../test/test_keep_position3_10.txt')

rospy.init_node('tests')
tree = TreeManager.from_param_server()
tree.render(extract_data_from_profile(sys.argv[1]))

# pydot.graph_from_dot_file(sys.argv[1])


pass
