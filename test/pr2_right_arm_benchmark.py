import symengine as se
from symengine import cos, sin, acos, sqrt
from symengine import Matrix as M
from symengine import Symbol as S


def right_arm_fk():
    torso_lift_joint = S('torso_lift_joint')
    r_shoulder_pan_joint = S('r_shoulder_pan_joint')
    r_shoulder_lift_joint = S('r_shoulder_lift_joint')
    r_upper_arm_roll_joint = S('r_upper_arm_roll_joint')
    r_elbow_flex_joint = S('r_elbow_flex_joint')
    r_forearm_roll_joint = S('r_forearm_roll_joint')
    r_wrist_flex_joint = S('r_wrist_flex_joint')
    r_wrist_roll_joint = S('r_wrist_roll_joint')

    world_to_base_link = se.eye(4)
    base_link_to_torso_lift_link = M([[1.0, 0, 0, -0.05],
                                      [0, 1.0, 0, 0],
                                      [0, 0, 1.0, 0.739675 + torso_lift_joint],
                                      [0, 0, 0, 1]])
    r_shoulder_pan_link = M([[cos(r_shoulder_pan_joint), -sin(r_shoulder_pan_joint), 0, 0],
                             [sin(r_shoulder_pan_joint), cos(r_shoulder_pan_joint), 0, -0.188],
                             [0, 0, ((1 - cos(r_shoulder_pan_joint)) + cos(r_shoulder_pan_joint)), 0],
                             [0, 0, 0, 1]])
    r_shoulder_lift_link = M([[cos(r_shoulder_lift_joint), 0, sin(r_shoulder_lift_joint), 0.1],
                              [0, ((1 - cos(r_shoulder_lift_joint)) + cos(r_shoulder_lift_joint)), 0, 0],
                              [-sin(r_shoulder_lift_joint), 0, cos(r_shoulder_lift_joint), 0],
                              [0, 0, 0, 1]])
    r_upper_arm_roll_link = M([[(1 - cos(r_upper_arm_roll_joint) + cos(r_upper_arm_roll_joint)), 0, 0, 0],
                               [0, cos(r_upper_arm_roll_joint), -sin(r_upper_arm_roll_joint), 0],
                               [0, sin(r_upper_arm_roll_joint), cos(r_upper_arm_roll_joint), 0],
                               [0, 0, 0, 1]])
    r_upper_arm_link = se.eye(4)
    r_elbow_flex_link = M([[cos(r_elbow_flex_joint), 0, sin(r_elbow_flex_joint), 0.4],
                           [0, ((1 - cos(r_elbow_flex_joint)) + cos(r_elbow_flex_joint)), 0, 0],
                           [-sin(r_elbow_flex_joint), 0, cos(r_elbow_flex_joint), 0],
                           [0, 0, 0, 1]])
    r_forearm_roll_link = M([[((1 - cos(r_forearm_roll_joint)) + cos(r_forearm_roll_joint)), 0, 0, 0],
                             [0, cos(r_forearm_roll_joint), -sin(r_forearm_roll_joint), 0],
                             [0, sin(r_forearm_roll_joint), cos(r_forearm_roll_joint), 0],
                             [0, 0, 0, 1]])
    r_forearm_link = se.eye(4)
    r_wrist_flex_link = M([[cos(r_wrist_flex_joint), 0, sin(r_wrist_flex_joint), 0.321],
                           [0, ((1 - cos(r_wrist_flex_joint)) + cos(r_wrist_flex_joint)), 0, 0],
                           [-sin(r_wrist_flex_joint), 0, cos(r_wrist_flex_joint), 0],
                           [0, 0, 0, 1]])
    r_wrist_roll_link = M([[((1 - cos(r_wrist_roll_joint)) + cos(r_wrist_roll_joint)), 0, 0, 0],
                           [0, cos(r_wrist_roll_joint), -sin(r_wrist_roll_joint), 0],
                           [0, sin(r_wrist_roll_joint), cos(r_wrist_roll_joint), 0],
                           [0, 0, 0, 1]])
    r_gripper_palm_link = se.eye(4)
    r_gripper_tool_frame = M([[1.0, 0, 0, 0.18],
                              [0, 1.0, 0, 0],
                              [0, 0, 1.0, 0],
                              [0, 0, 0, 1]])

    return world_to_base_link * base_link_to_torso_lift_link * r_shoulder_pan_link * r_shoulder_lift_link * \
           r_upper_arm_roll_link * r_upper_arm_link * r_elbow_flex_link * r_forearm_roll_link * r_forearm_roll_link * \
           r_forearm_link * r_wrist_flex_link * r_wrist_roll_link * r_gripper_palm_link * r_gripper_tool_frame


def axis_angle_from_matrix(rotation_matrix):
    rm = rotation_matrix
    angle = (sum(rm[i, i] for i in range(3)) - 1) / 2
    angle = acos(angle)
    x = (rm[2, 1] - rm[1, 2])
    y = (rm[0, 2] - rm[2, 0])
    z = (rm[1, 0] - rm[0, 1])
    n = sqrt(x * x + y * y + z * z)

    axis = M([x / n, y / n, z / n])
    return axis, angle


def jacobian(fk, js):
    current_pose_evaluated = fk.subs(js)
    x = fk[0, 3]
    y = fk[1, 3]
    z = fk[2, 3]

    current_rotation = fk[:3, :3]
    current_rotation_evaluated = current_pose_evaluated[:3, :3]

    # multiply with a slight rotation around z axis to prevent devision by zero
    epsilon = M([[0.999999995, -9.99999998333333e-05, 0],
                 [9.99999998333333e-05, 0.999999995, 0],
                 [0, 0, 1.0]])
    axis, angle = axis_angle_from_matrix((current_rotation.T * (current_rotation_evaluated )).T)
    c_aa = current_rotation[:3, :3] * (axis * angle)

    rx = c_aa[0]
    ry = c_aa[1]
    rz = c_aa[2]

    m = M([x, y, z, rx, ry, rz])

    slow_jacobian = m.jacobian(M(js.keys()))
    return slow_jacobian

def make_fast(slow_jacobian):
    free_symbols = list(slow_jacobian.free_symbols)

    fast_jacobian = se.Lambdify(free_symbols, slow_jacobian, real=True, cse=False, backend='llvm')

    fast_jacobian_cse = se.Lambdify(free_symbols, slow_jacobian, real=True, cse=True, backend='llvm')

    subs = [js[x] for x in free_symbols]
    slow_jacobian.subs(js)
    fast_jacobian(subs)
    fast_jacobian_cse(subs)
    se.cse(slow_jacobian)


if __name__ == '__main__':
    fk = right_arm_fk()
    # some random joint state
    js = {S('torso_lift_joint'): 0.17070830166937445,
          S('r_elbow_flex_joint'): -1.808187398260726,
          S('r_shoulder_lift_joint'): 0.9460061920661217,
          S('r_upper_arm_roll_joint'): -2.57273953335457,
          S('r_shoulder_pan_joint'): 0.5554896480472835,
          S('r_forearm_roll_joint'): 4.311660523426189,
          S('r_wrist_roll_joint'): 2.465788739206398,
          S('r_wrist_flex_joint'): -1.8156365372418775}
    slow_jacobian = jacobian(fk, js)
    for i in range(10):
        make_fast(slow_jacobian)
