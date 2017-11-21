from giskardpy.robot import Robot

def hacky_urdf_parser_fix(urdf_path):
    fixed_urdf = ''
    delete = False
    black_list = ['transmission']
    black_open = ['<{}'.format(x) for x in black_list]
    black_close = ['</{}'.format(x) for x in black_list]
    with open(urdf_path, 'r') as urdf:
        for line in urdf.readlines():
            if len([x for x in black_open if x in line]) > 0:
                delete = True
            if len([x for x in black_close if x in line]) > 0:
                delete = False
                continue
            if not delete:
                fixed_urdf += line
    return fixed_urdf

class PR2(Robot):
    def __init__(self, urdf_path='pr2.urdf'):
        super(PR2, self).__init__()
        urdf = hacky_urdf_parser_fix(urdf_path)
        self.load_from_urdf_string(urdf, 'base_link', ['l_gripper_tool_frame', 'r_gripper_tool_frame'])
        for joint_name in self.weight_input.get_float_names():
            self.set_joint_weight(joint_name, 1)

