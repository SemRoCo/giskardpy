import numpy as np

import PyKDL as kdl
import urdf_parser_py.urdf as up

from giskardpy.model.utils import hacky_urdf_parser_fix


def euler_to_quat(r, p, y):
    sr, sp, sy = np.sin(r/2.0), np.sin(p/2.0), np.sin(y/2.0)
    cr, cp, cy = np.cos(r/2.0), np.cos(p/2.0), np.cos(y/2.0)
    return [sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy,
            cr*cp*cy + sr*sp*sy]

def urdf_pose_to_kdl_frame(pose):
    pos = [0., 0., 0.]
    rot = [0., 0., 0.]
    if pose is not None:
        if pose.position is not None:
            pos = pose.position
        if pose.rotation is not None:
            rot = pose.rotation
    return kdl.Frame(kdl.Rotation.Quaternion(*euler_to_quat(*rot)),
                     kdl.Vector(*pos))

def urdf_joint_to_kdl_joint(jnt):
    origin_frame = urdf_pose_to_kdl_frame(jnt.origin)
    if jnt.joint_type == 'fixed':
        return kdl.Joint(jnt.name, kdl.Joint)
    axis = kdl.Vector(*jnt.axis)
    if jnt.joint_type == 'revolute':
        return kdl.Joint(jnt.name, origin_frame.p,
                         origin_frame.M * axis, kdl.Joint.RotAxis)
    if jnt.joint_type == 'continuous':
        return kdl.Joint(jnt.name, origin_frame.p,
                         origin_frame.M * axis, kdl.Joint.RotAxis)
    if jnt.joint_type == 'prismatic':
        return kdl.Joint(jnt.name, origin_frame.p,
                         origin_frame.M * axis, kdl.Joint.TransAxis)
    print("Unknown joint type: %s." % jnt.joint_type)
    return kdl.Joint(jnt.name, kdl.Joint)

def urdf_inertial_to_kdl_rbi(i):
    origin = urdf_pose_to_kdl_frame(i.origin)
    rbi = kdl.RigidBodyInertia(i.mass, origin.p,
                               kdl.RotationalInertia(i.inertia.ixx,
                                                     i.inertia.iyy,
                                                     i.inertia.izz,
                                                     i.inertia.ixy,
                                                     i.inertia.ixz,
                                                     i.inertia.iyz))
    return origin.M * rbi

def kdl_tree_from_urdf_model(urdf):
    root = urdf.get_root()
    tree = kdl.Tree(root)
    def add_children_to_tree(parent):
        if parent in urdf.child_map:
            for joint, child_name in urdf.child_map[parent]:
                child = urdf.link_map[child_name]
                if child.inertial is not None:
                    kdl_inert = urdf_inertial_to_kdl_rbi(child.inertial)
                else:
                    kdl_inert = kdl.RigidBodyInertia()
                kdl_jnt = urdf_joint_to_kdl_joint(urdf.joint_map[joint])
                kdl_origin = urdf_pose_to_kdl_frame(urdf.joint_map[joint].origin)
                kdl_sgm = kdl.Segment(child_name, kdl_jnt,
                                      kdl_origin, kdl_inert)
                tree.addSegment(kdl_sgm, parent)
                add_children_to_tree(child_name)
    add_children_to_tree(root)
    return tree


class KDL(object):
    class KDLRobot(object):
        def __init__(self, chain):
            self.chain = chain
            self.fksolver = kdl.ChainFkSolverPos_recursive(self.chain)
            self.jac_solver = kdl.ChainJntToJacSolver(self.chain)
            self.jacobian = kdl.Jacobian(self.chain.getNrOfJoints())
            self.joints = self.get_joints()

        def get_joints(self):
            joints = []
            for i in range(self.chain.getNrOfSegments()):
                joint = self.chain.getSegment(i).getJoint()
                if joint.getType() != 8:
                    joints.append(str(joint.getName()))
            return joints

        def fk(self, js_dict):
            js = [js_dict[j] for j in self.joints]
            f = kdl.Frame()
            joint_array = kdl.JntArray(len(js))
            for i in range(len(js)):
                joint_array[i] = js[i]
            self.fksolver.JntToCart(joint_array, f)
            return f

        def fk_np(self, js_dict):
            f = self.fk(js_dict)
            r = [[f.M[0, 0], f.M[0, 1], f.M[0, 2], f.p[0]],
                 [f.M[1, 0], f.M[1, 1], f.M[1, 2], f.p[1]],
                 [f.M[2, 0], f.M[2, 1], f.M[2, 2], f.p[2]],
                 [0, 0, 0, 1], ]
            return np.array(r)

        def fk_np_inv(self, js_dict):
            f = self.fk(js_dict).Inverse()
            r = [[f.M[0, 0], f.M[0, 1], f.M[0, 2], f.p[0]],
                 [f.M[1, 0], f.M[1, 1], f.M[1, 2], f.p[1]],
                 [f.M[2, 0], f.M[2, 1], f.M[2, 2], f.p[2]],
                 [0, 0, 0, 1], ]
            return np.array(r)

    def __init__(self, urdf):
        if urdf.endswith(u'.urdfs'):
            with open(urdf, u'r') as file:
                urdf = file.read()
        r = up.URDF.from_xml_string(hacky_urdf_parser_fix(urdf))
        self.tree = kdl_tree_from_urdf_model(r)
        self.robots = {}

    def get_robot(self, root, tip):
        root = str(root)
        tip = str(tip)
        if (root, tip) not in self.robots:
            self.chain = self.tree.getChain(root, tip)
            self.robots[root, tip] = self.KDLRobot(self.chain)
        return self.robots[root, tip]