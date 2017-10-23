from giskardpy.robot import Robot


class PR2(Robot):
    def __init__(self, urdf_path='pr2.urdf'):
        super(PR2, self).__init__()
        urdf = self.hacky_urdf_parser_fix(urdf_path)
        self.load_from_urdf_string(urdf, 'base_link', ['l_gripper_tool_frame', 'r_gripper_tool_frame'])
        for joint_name in self.joint_constraints:
            self.set_joint_weight(joint_name, 0.01)

    def hacky_urdf_parser_fix(self, urdf_path):
        fixed_urdf = ''
        delete = False
        black_list = ['transmission']
        with open(urdf_path, 'r') as urdf:
            for line in urdf.readlines():
                if len([x for x in black_list if x in line]) > 0:
                    if not delete:
                        delete = True
                    else:
                        delete = False
                        continue
                if not delete:
                    fixed_urdf += line
        return fixed_urdf
