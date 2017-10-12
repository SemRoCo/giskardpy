from urdf_parser_py.urdf import URDF, Robot, Joint

import sympy as sp


def urdf_string_to_robot_constraints(urdf_string, root_link, tip_links):
    return urdf_object_to_robot_constraints(URDF.from_xml_string(urdf_string), root_link, tip_links)


def urdf_path_to_robot_constraints(path_to_urdf, root_link, tip_links):
    return urdf_object_to_robot_constraints(URDF.from_xml_file(path_to_urdf), root_link, tip_links)


def urdf_object_to_robot_constraints(urdf, root_link, tip_links):
    robot_constraints = {}
    robot_constraints["input_vars"] = tree_joints_from_urdf_object(urdf, root_link, tip_links)

    # TODO: get controllable constraints, i.e. velocity limits
    # TODO: get hard constraints, i.e. position limits
    return robot_constraints


def chain_joints_from_urdf_object(urdf, root_link, tip_link):
    """
    Returns a dict with joint names as keys and sympy symbols
    as values for all 1-dof movable robot joints in URDF between
    ROOT_LINK and TIP_LINK.

    :param urdf: URDF.Robot, obtained from URDF parser.
    :param root_link: str, denoting the root of the kin. chain
    :param tip_link: str, denoting the tip of the kin. chain
    :return: dict{str, sympy.Symbol}, with symbols for all joints in chain
    """
    input_vars = {}
    for joint_name in urdf.get_chain(root_link, tip_link, True, False, False):
        if urdf.joint_map[joint_name].type in ['revolute', 'continuous', 'prismatic']:
            input_vars[joint_name] = sp.Symbol(joint_name)
    return input_vars


def tree_joints_from_urdf_object(urdf, root_link, tip_links):
    """
    Returns a dict with joint names as keys and sympy symbols
    as values for all 1-dof movable robot joints in URDF between
    ROOT_LINK and TIP_LINKS.

    :param urdf: URDF.Robot, obtained from URDF parser.
    :param root_link: str, denoting the root of the kin. tree
    :param tip_links: str, denoting the tips of the kin. tree
    :return: dict{str, sympy.Symbol}, with symbols for all joints in tree
    """
    input_vars = {}
    for tip_link in tip_links:
        input_vars.update(chain_joints_from_urdf_object(urdf, root_link, tip_link))
    return input_vars