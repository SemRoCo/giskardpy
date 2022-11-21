import errno
import os
import pickle
from typing import List, Tuple, Union, Callable, Iterable, Optional

import casadi as ca
import numpy as np
import rospy
from casadi import sign, cos, sin, sqrt, atan2, acos
from numpy import pi

from giskardpy.my_types import expr_symbol, expr_matrix, any_expr
from giskardpy.utils import logging
from giskardpy.utils.tfwrapper import msg_to_homogeneous_matrix


def var(variables_names: str) -> List[expr_symbol]:
    """
    :param variables_names: e.g. 'a b c'
    :return: e.g. [Symbol('a'), Symbol('b'), Symbol('c')]
    """
    symbols = []
    for v in variables_names.split(' '):
        symbols.append(Symbol(v))
    return symbols


def diag(args: Union[List[expr_symbol], expr_matrix]) -> expr_matrix:
    return ca.diag(Matrix(args))


def Symbol(data: str) -> ca.SX:
    if isinstance(data, str):
        return ca.SX.sym(data)
    return ca.SX(data)


def jacobian(expressions: expr_matrix, symbols: Union[List[ca.SX], expr_matrix], order: float = 1) -> expr_matrix:
    if isinstance(expressions, list):
        expressions = Matrix(expressions)
    if order == 1:
        return ca.jacobian(expressions, Matrix(symbols))
    elif order == 2:
        j = jacobian(expressions, symbols, order=1)
        for i, symbol in enumerate(symbols):
            j[:, i] = jacobian(j[:, i], symbol)
        return j
    else:
        raise NotImplementedError('jacobian only supports order 1 and 2')


def equivalent(expression1: any_expr, expression2: any_expr) -> bool:
    return ca.is_equal(ca.simplify(expression1), ca.simplify(expression2), 1)


def free_symbols(expression: any_expr) -> List[ca.SX]:
    return ca.symvar(expression)


def is_matrix(expression: any_expr) -> bool:
    return hasattr(expression, 'shape') and expression.shape[0] * expression.shape[1] > 1


def is_homo_vector(expression: any_expr):
    return is_matrix(expression) and expression.shape == (4, 1) and expression[-1] == 0


def is_symbol(expression: any_expr) -> bool:
    return expression.shape[0] * expression.shape[1] == 1


def create_symbols(names: List[str]) -> List[ca.SX]:
    return [Symbol(x) for x in names]


def compile_and_execute(f: Callable, params: Union[List[Union[float, np.ndarray]], np.ndarray]) \
        -> Union[float, np.ndarray]:
    input = []
    symbol_params = []
    symbol_params2 = []

    for i, param in enumerate(params):
        if isinstance(param, list):
            param = np.array(param)
        if isinstance(param, np.ndarray):
            symbol_param = ca.SX.sym('m', *param.shape)
            if len(param.shape) == 2:
                l = param.shape[0] * param.shape[1]
            else:
                l = param.shape[0]

            input.append(param.reshape((l, 1)))
            symbol_params.append(symbol_param)
            asdf = symbol_param.T.reshape((l, 1))
            symbol_params2.extend(asdf[k] for k in range(l))
        else:
            input.append(np.array([param], ndmin=2))
            symbol_param = ca.SX.sym('s')
            symbol_params.append(symbol_param)
            symbol_params2.append(symbol_param)
    fast_f = speed_up(f(*symbol_params), symbol_params2)
    input = np.concatenate(input).T[0]
    result = fast_f.call2(input)
    if result.shape[0] * result.shape[1] == 1:
        return result[0][0]
    elif result.shape[1] == 1:
        return result.T[0]
    elif result.shape[0] == 1:
        return result[0]
    else:
        return result


def Matrix(data: Union[Iterable[expr_symbol],
                       Iterable[Iterable[expr_symbol]],
                       np.ndarray,
                       expr_matrix]) -> expr_matrix:
    try:
        return ca.SX(data)
    except NotImplementedError:
        if hasattr(data, 'shape'):
            m = ca.SX(*data.shape)
            y = data.shape[1]
        else:
            x = len(data)
            if isinstance(data[0], list) or isinstance(data[0], tuple):
                y = len(data[0])
            else:
                y = 1
            m = ca.SX(x, y)
        for i in range(m.shape[0]):
            if y > 1:
                for j in range(m.shape[1]):
                    try:
                        m[i, j] = data[i][j]
                    except:
                        m[i, j] = data[i, j]
            else:
                m[i] = data[i]
        return m


def ros_msg_to_matrix(msg: rospy.Message) -> expr_matrix:
    return Matrix(msg_to_homogeneous_matrix(msg))


def matrix_to_list(m: expr_matrix) -> List[expr_symbol]:
    try:
        len(m)
        return m
    except:
        return [m[i] for i in range(m.shape[0])]


def zeros(x: int, y: int) -> expr_matrix:
    return ca.SX.zeros(x, y)


def ones(x: int, y: int) -> expr_matrix:
    return ca.SX.ones(x, y)


def abs(x: expr_symbol) -> expr_symbol:
    return ca.fabs(x)


def max(x: expr_symbol, y: expr_symbol) -> expr_symbol:
    return ca.fmax(x, y)


def min(x: expr_symbol, y: expr_symbol) -> expr_symbol:
    return ca.fmin(x, y)


def limit(x: expr_symbol, lower_limit: expr_symbol, upper_limit: expr_symbol) -> expr_symbol:
    return max(lower_limit, min(upper_limit, x))


def if_else(condition: expr_symbol, if_result: expr_symbol, else_result: expr_symbol) -> expr_symbol:
    return ca.if_else(condition, if_result, else_result)


def logic_and(*args: List[expr_symbol]) -> expr_symbol:
    assert len(args) >= 2, 'and must be called with at least 2 arguments'
    if len(args) == 2:
        return ca.logic_and(args[0], args[1])
    else:
        return ca.logic_and(args[0], logic_and(*args[1:]))


def logic_or(*args: List[expr_symbol]) -> expr_symbol:
    assert len(args) >= 2, 'and must be called with at least 2 arguments'
    if len(args) == 2:
        return ca.logic_or(args[0], args[1])
    else:
        return ca.logic_or(args[0], logic_and(*args[1:]))


def if_greater(a: expr_symbol, b: expr_symbol, if_result: expr_symbol, else_result: expr_symbol) -> expr_symbol:
    return if_else(ca.gt(a, b), if_result, else_result)


def if_less(a: expr_symbol, b: expr_symbol, if_result: expr_symbol, else_result: expr_symbol) -> expr_symbol:
    return if_else(ca.lt(a, b), if_result, else_result)


def if_greater_zero(condition: expr_symbol, if_result: expr_symbol, else_result: expr_symbol) -> expr_symbol:
    """
    :return: if_result if condition > 0 else else_result
    """
    _condition = sign(condition)  # 1 or -1
    _if = max(0, _condition) * if_result  # 0 or if_result
    _else = -min(0, _condition) * else_result  # 0 or else_result
    return _if + _else + (1 - abs(_condition)) * else_result  # if_result or else_result


def if_greater_eq_zero(condition: expr_symbol, if_result: expr_symbol, else_result: expr_symbol) -> expr_symbol:
    """
    :return: if_result if condition >= 0 else else_result
    """
    return if_greater_eq(condition, 0, if_result, else_result)


def if_greater_eq(a: expr_symbol, b: expr_symbol, if_result: expr_symbol, else_result: expr_symbol) -> expr_symbol:
    """
    :return: if_result if a >= b else else_result
    """
    return if_else(ca.ge(a, b), if_result, else_result)


def if_less_eq(a: expr_symbol, b: expr_symbol, if_result: expr_symbol, else_result: expr_symbol) -> expr_symbol:
    """
    :return: if_result if a <= b else else_result
    """
    return if_greater_eq(b, a, if_result, else_result)


def if_eq_zero(condition:expr_symbol, if_result: expr_symbol, else_result: expr_symbol) -> expr_symbol:
    """
    :return: if_result if condition == 0 else else_result
    """
    return if_else(condition, else_result, if_result)


def if_eq(a: expr_symbol, b: expr_symbol, if_result: expr_symbol, else_result: expr_symbol) -> expr_symbol:
    return if_else(ca.eq(a, b), if_result, else_result)


def if_eq_cases(a: expr_symbol, b_result_cases: Union[Iterable[Tuple[expr_symbol, expr_symbol]], expr_matrix],
                else_result: expr_symbol) -> expr_symbol:
    result = else_result
    if isinstance(b_result_cases, list):
        b_result_cases = Matrix(b_result_cases)
    for i in range(b_result_cases.shape[0]):
        b = b_result_cases[i, 0]
        b_result = b_result_cases[i, 1]
        result = if_eq(a, b, b_result, result)
    return result


def if_less_eq_cases(a: expr_symbol, b_result_cases: Iterable[Tuple[expr_symbol, expr_symbol]],
                     else_result: expr_symbol) -> expr_symbol:
    """
    This only works if b_result_cases is sorted in an ascending order.
    """
    result = else_result
    if isinstance(b_result_cases, list):
        b_result_cases = Matrix(b_result_cases)
    for i in reversed(range(b_result_cases.shape[0] - 1)):
        b = b_result_cases[i, 0]
        b_result = b_result_cases[i, 1]
        result = if_less_eq(a, b, b_result, result)
    return result


def safe_compiled_function(f, file_name):
    if not os.path.exists(os.path.dirname(file_name)):
        try:
            os.makedirs(os.path.dirname(file_name))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(file_name, 'w') as file:
        pickle.dump(f, file)
        logging.loginfo('saved {}'.format(file_name))


def load_compiled_function(file_name):
    if os.path.isfile(file_name):
        try:
            with open(file_name, 'r') as file:
                fast_f = pickle.load(file)
                return fast_f
        except EOFError as e:
            os.remove(file_name)
            logging.logerr('{} deleted because it was corrupted'.format(file_name))


class CompiledFunction:
    def __init__(self, str_params: Iterable[str], fast_f: ca.Function, shape: Tuple[int, int]):
        self.str_params = str_params
        self.fast_f = fast_f
        self.shape = shape
        self.buf, self.f_eval = fast_f.buffer()
        self.out = np.zeros(self.shape, order='F')
        self.buf.set_res(0, memoryview(self.out))

    def __call__(self, **kwargs) -> np.ndarray:
        filtered_args = [kwargs[k] for k in self.str_params]
        return self.call2(filtered_args)

    def call2(self, filtered_args: Iterable[float]) -> np.ndarray:
        """
        :param filtered_args: parameter values in the same order as in self.str_params
        """

        filtered_args = np.array(filtered_args, dtype=float)
        self.buf.set_arg(0, memoryview(filtered_args))
        self.f_eval()
        return self.out


def speed_up(function, parameters, backend='clang') -> CompiledFunction:
    str_params = [str(x) for x in parameters]
    try:
        f = ca.Function('f', [Matrix(parameters)], [ca.densify(function)])
    except:
        # try:
        f = ca.Function('f', [Matrix(parameters)], ca.densify(function))
        # except:
        #     f = ca.Function('f', parameters, [ca.densify(function)])
    return CompiledFunction(str_params, f, function.shape)


def cross(u: expr_matrix, v: expr_matrix) -> expr_matrix:
    """
    :param u: 1d matrix
    :param v: 1d matrix
    :return: 1d Matrix. If u and v have length 4, it ignores the last entry and adds a zero to the result.
    """
    if is_homo_vector(u):
        u = u[:-1]
    if is_homo_vector(v):
        v = v[:-1]
    result = ca.cross(u, v)
    return Matrix([result[0],
                   result[1],
                   result[2],
                   0])


def vector3(x: expr_symbol, y: expr_symbol, z: expr_symbol) -> expr_matrix:
    return Matrix([x, y, z, 0])


def point3(x: expr_symbol, y: expr_symbol, z: expr_symbol) -> expr_matrix:
    return Matrix([x, y, z, 1])


def norm(v: expr_matrix) -> expr_symbol:
    return ca.norm_2(v)


def scale(v: expr_matrix, a: expr_symbol) -> expr_matrix:
    return save_division(v, norm(v)) * a


def dot(*matrices: expr_matrix) -> expr_matrix:
    return ca.mtimes(matrices)


def translation3(x: expr_symbol, y: expr_symbol, z: expr_symbol) -> expr_matrix:
    """
    :return: 4x4 Matrix
        [[1,0,0,x],
         [0,1,0,y],
         [0,0,1,z],
         [0,0,0,1]]
    """
    r = eye(4)
    r[0, 3] = x
    r[1, 3] = y
    r[2, 3] = z
    return r


def rotation_matrix_from_vectors(x: Optional[expr_matrix] = None,
                                 y: Optional[expr_matrix] = None,
                                 z: Optional[expr_matrix] = None) -> expr_matrix:
    if x is not None:
        x = scale(x, 1)
    if y is not None:
        y = scale(y, 1)
    if z is not None:
        z = scale(z, 1)
    if x is not None and y is not None and z is None:
        z = cross(x, y)
        z = scale(z, 1)
    elif x is not None and y is None and z is not None:
        y = cross(z, x)
        y = scale(y, 1)
    elif x is None and y is not None and z is not None:
        x = cross(y, z)
        x = scale(x, 1)
    else:
        raise AttributeError(f'only one vector can be None')
    R = Matrix([[x[0], y[0], z[0], 0],
                [x[1], y[1], z[1], 0],
                [x[2], y[2], z[2], 0],
                [0, 0, 0, 1]])
    return R


def rotation_matrix_from_rpy(roll: expr_symbol, pitch: expr_symbol, yaw: expr_symbol) -> expr_matrix:
    """
    Conversion of roll, pitch, yaw to 4x4 rotation matrix according to:
    https://github.com/orocos/orocos_kinematics_dynamics/blob/master/orocos_kdl/src/frames.cpp#L167
    :return: 4x4 Matrix
    """
    # TODO don't split this into 3 matrices

    rx = Matrix([[1, 0, 0, 0],
                 [0, ca.cos(roll), -ca.sin(roll), 0],
                 [0, ca.sin(roll), ca.cos(roll), 0],
                 [0, 0, 0, 1]])
    ry = Matrix([[ca.cos(pitch), 0, ca.sin(pitch), 0],
                 [0, 1, 0, 0],
                 [-ca.sin(pitch), 0, ca.cos(pitch), 0],
                 [0, 0, 0, 1]])
    rz = Matrix([[ca.cos(yaw), -ca.sin(yaw), 0, 0],
                 [ca.sin(yaw), ca.cos(yaw), 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])
    return dot(rz, ry, rx)


def rotation_matrix_from_axis_angle(axis: Union[Tuple[expr_symbol, expr_symbol, expr_symbol], expr_matrix],
                                    angle: expr_symbol) -> expr_matrix:
    """
    Conversion of unit axis and angle to 4x4 rotation matrix according to:
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/index.htm
    :param axis: 3x1 Matrix
    :return: 4x4 Matrix
    """
    ct = ca.cos(angle)
    st = ca.sin(angle)
    vt = 1 - ct
    m_vt_0 = vt * axis[0]
    m_vt_1 = vt * axis[1]
    m_vt_2 = vt * axis[2]
    m_st_0 = axis[0] * st
    m_st_1 = axis[1] * st
    m_st_2 = axis[2] * st
    m_vt_0_1 = m_vt_0 * axis[1]
    m_vt_0_2 = m_vt_0 * axis[2]
    m_vt_1_2 = m_vt_1 * axis[2]
    return Matrix([[ct + m_vt_0 * axis[0], -m_st_2 + m_vt_0_1, m_st_1 + m_vt_0_2, 0],
                   [m_st_2 + m_vt_0_1, ct + m_vt_1 * axis[1], -m_st_0 + m_vt_1_2, 0],
                   [-m_st_1 + m_vt_0_2, m_st_0 + m_vt_1_2, ct + m_vt_2 * axis[2], 0],
                   [0, 0, 0, 1]])


def rotation_matrix_from_quaternion(x: expr_symbol, y: expr_symbol, z: expr_symbol, w: expr_symbol) -> expr_matrix:
    """
    Unit quaternion to 4x4 rotation matrix according to:
    https://github.com/orocos/orocos_kinematics_dynamics/blob/master/orocos_kdl/src/frames.cpp#L167
    :return: 4x4 Matrix
    """
    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w
    return Matrix([[w2 + x2 - y2 - z2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y, 0],
                   [2 * x * y + 2 * w * z, w2 - x2 + y2 - z2, 2 * y * z - 2 * w * x, 0],
                   [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, w2 - x2 - y2 + z2, 0],
                   [0, 0, 0, 1]])


def frame_axis_angle(x: expr_symbol, y: expr_symbol, z: expr_symbol, axis: expr_matrix, angle: expr_symbol) -> expr_matrix:
    """
    :param axis: 3x1 Matrix
    :return: 4x4 Matrix
    """
    return dot(translation3(x, y, z), rotation_matrix_from_axis_angle(axis, angle))


def frame_rpy(x: expr_symbol, y: expr_symbol, z: expr_symbol, roll: expr_symbol, pitch: expr_symbol, yaw: expr_symbol) \
        -> expr_matrix:
    """
    :return: 4x4 Matrix
    """
    return dot(translation3(x, y, z), rotation_matrix_from_rpy(roll, pitch, yaw))


def frame_quaternion(x: expr_symbol, y: expr_symbol, z: expr_symbol, qx: expr_symbol, qy: expr_symbol, qz: expr_symbol,
                     qw: expr_symbol) -> expr_matrix:
    """
    :return: 4x4 Matrix
    """
    return dot(translation3(x, y, z), rotation_matrix_from_quaternion(qx, qy, qz, qw))


def frame_from_x_y_rot(x: expr_symbol, y: expr_symbol, rot: expr_symbol) -> expr_matrix:
    parent_P_link = translation3(x, y, 0)
    link_R_child = rotation_matrix_from_axis_angle(vector3(0, 0, 1),
                                                   rot)
    return dot(parent_P_link, link_R_child)


def eye(size: int) -> expr_symbol:
    return ca.SX.eye(size)


def kron(m1: expr_matrix, m2: expr_matrix) -> expr_matrix:
    return ca.kron(m1, m2)


def inverse_frame(frame: expr_matrix) -> expr_matrix:
    """
    :param frame: 4x4 Matrix
    :return: 4x4 Matrix
    """
    inv = eye(4)
    inv[:3, :3] = frame[:3, :3].T
    inv[:3, 3] = dot(-inv[:3, :3], frame[:3, 3])
    return inv


def position_of(frame: expr_matrix) -> expr_matrix:
    """
    :param frame: 4x4 Matrix
    :return: 4x1 Matrix; the translation part of a frame in form of a point
    """
    return frame[:4, 3:]


def translation_of(frame: expr_matrix) -> expr_matrix:
    """
    :param frame: 4x4 Matrix
    :return: 4x4 Matrix; sets the rotation part of a frame to identity
    """
    r = eye(4)
    r[0, 3] = frame[0, 3]
    r[1, 3] = frame[1, 3]
    r[2, 3] = frame[2, 3]
    return r


def rotation_of(frame: expr_matrix) ->expr_matrix:
    """
    :param frame: 4x4 Matrix
    :return: 4x4 Matrix; sets the translation part of a frame to 0
    """
    r = eye(4)
    for i in range(3):
        for j in range(3):
            r[i, j] = frame[i, j]
    return r


def trace(matrix: expr_matrix) -> expr_symbol:
    s = 0
    for i in range(matrix.shape[0]):
        s += matrix[i, i]
    return s


def rotation_distance(a_R_b: expr_matrix, a_R_c: expr_matrix) -> expr_symbol:
    """
    :param a_R_b: 4x4 or 3x3 Matrix
    :param a_R_c: 4x4 or 3x3 Matrix
    :return: angle of axis angle representation of b_R_c
    """
    difference = dot(a_R_b.T, a_R_c)
    # return axis_angle_from_matrix(difference)[1]
    angle = (trace(difference[:3, :3]) - 1) / 2
    angle = min(angle, 1)
    angle = max(angle, -1)
    return acos(angle)


def vstack(list_of_matrices: List[expr_matrix]) -> expr_matrix:
    return ca.vertcat(*list_of_matrices)


def hstack(list_of_matrices: List[expr_matrix]) -> expr_matrix:
    return ca.horzcat(*list_of_matrices)


def axis_angle_from_matrix(rotation_matrix: expr_matrix) -> Tuple[expr_matrix, expr_symbol]:
    """
    MAKE SURE MATRIX IS NORMALIZED
    :param rotation_matrix: 4x4 or 3x3 Matrix
    :return: 3x1 Matrix, angle
    """
    q = quaternion_from_matrix(rotation_matrix)
    return axis_angle_from_quaternion(q[0], q[1], q[2], q[3])


def axis_angle_from_quaternion(x: expr_symbol, y: expr_symbol, z: expr_symbol, w: expr_symbol) \
        -> Tuple[expr_matrix, expr_symbol]:
    """
    :return: 4x1 Matrix
    """
    l = norm(Matrix([x, y, z, w]))
    x, y, z, w = x / l, y / l, z / l, w / l
    w2 = sqrt(1 - w ** 2)
    m = if_eq_zero(w2, 1, w2)  # avoid /0
    angle = if_eq_zero(w2, 0, (2 * acos(limit(w, -1, 1))))
    x = if_eq_zero(w2, 0, x / m)
    y = if_eq_zero(w2, 0, y / m)
    z = if_eq_zero(w2, 1, z / m)
    return Matrix([x, y, z]), angle


def normalize_axis_angle(axis: expr_matrix, angle: expr_symbol) -> Tuple[expr_matrix, expr_symbol]:
    axis = if_less(angle, 0, -axis, axis)
    angle = abs(angle)
    return axis, angle


def normalize_rotation_matrix(R: expr_symbol) -> expr_symbol:
    """Scales each of the axes to the length of one."""
    scale_v = 1.0
    R[:3, 0] = scale(R[:3, 0], scale_v)
    R[:3, 1] = scale(R[:3, 1], scale_v)
    R[:3, 2] = scale(R[:3, 2], scale_v)
    return R


def quaternion_from_axis_angle(axis: expr_matrix, angle: expr_symbol) -> expr_matrix:
    """
    :param axis: 3x1 Matrix
    :return: 4x1 Matrix
    """
    half_angle = angle / 2
    return Matrix([axis[0] * sin(half_angle),
                   axis[1] * sin(half_angle),
                   axis[2] * sin(half_angle),
                   cos(half_angle)])


def axis_angle_from_rpy(roll: expr_symbol, pitch: expr_symbol, yaw: expr_symbol) -> Tuple[expr_matrix, expr_symbol]:
    """
    :return: 3x1 Matrix, angle
    """
    q = quaternion_from_rpy(roll, pitch, yaw)
    return axis_angle_from_quaternion(q[0], q[1], q[2], q[3])


_EPS = np.finfo(float).eps * 4.0


def rpy_from_matrix(rotation_matrix: expr_matrix) -> Tuple[expr_symbol, expr_symbol, expr_symbol]:
    """
    :param rotation_matrix: 4x4 Matrix
    :return: roll, pitch, yaw
    """
    i = 0
    j = 1
    k = 2

    cy = sqrt(rotation_matrix[i, i] * rotation_matrix[i, i] + rotation_matrix[j, i] * rotation_matrix[j, i])
    if0 = cy - _EPS
    ax = if_greater_zero(if0,
                         atan2(rotation_matrix[k, j], rotation_matrix[k, k]),
                         atan2(-rotation_matrix[j, k], rotation_matrix[j, j]))
    ay = if_greater_zero(if0,
                         atan2(-rotation_matrix[k, i], cy),
                         atan2(-rotation_matrix[k, i], cy))
    az = if_greater_zero(if0,
                         atan2(rotation_matrix[j, i], rotation_matrix[i, i]),
                         0)
    return ax, ay, az


def quaternion_from_rpy(roll: expr_symbol, pitch: expr_symbol, yaw: expr_symbol) -> expr_matrix:
    """
    :return: 4x1 Matrix
    """
    roll_half = roll / 2.0
    pitch_half = pitch / 2.0
    yaw_half = yaw / 2.0

    c_roll = cos(roll_half)
    s_roll = sin(roll_half)
    c_pitch = cos(pitch_half)
    s_pitch = sin(pitch_half)
    c_yaw = cos(yaw_half)
    s_yaw = sin(yaw_half)

    cc = c_roll * c_yaw
    cs = c_roll * s_yaw
    sc = s_roll * c_yaw
    ss = s_roll * s_yaw

    x = c_pitch * sc - s_pitch * cs
    y = c_pitch * ss + s_pitch * cc
    z = c_pitch * cs - s_pitch * sc
    w = c_pitch * cc + s_pitch * ss

    return Matrix([x, y, z, w])


def quaternion_from_matrix(matrix: expr_matrix) -> expr_matrix:
    """
    :param matrix: 4x4 or 3x3 Matrix
    :return: 4x1 Matrix
    """
    q = Matrix([0, 0, 0, 0])
    if isinstance(matrix, np.ndarray):
        M = Matrix(matrix.tolist())
    else:
        M = Matrix(matrix)
    t = trace(M)

    if0 = t - M[3, 3]

    if1 = M[1, 1] - M[0, 0]

    m_i_i = if_greater_zero(if1, M[1, 1], M[0, 0])
    m_i_j = if_greater_zero(if1, M[1, 2], M[0, 1])
    m_i_k = if_greater_zero(if1, M[1, 0], M[0, 2])

    m_j_i = if_greater_zero(if1, M[2, 1], M[1, 0])
    m_j_j = if_greater_zero(if1, M[2, 2], M[1, 1])
    m_j_k = if_greater_zero(if1, M[2, 0], M[1, 2])

    m_k_i = if_greater_zero(if1, M[0, 1], M[2, 0])
    m_k_j = if_greater_zero(if1, M[0, 2], M[2, 1])
    m_k_k = if_greater_zero(if1, M[0, 0], M[2, 2])

    if2 = M[2, 2] - m_i_i

    m_i_i = if_greater_zero(if2, M[2, 2], m_i_i)
    m_i_j = if_greater_zero(if2, M[2, 0], m_i_j)
    m_i_k = if_greater_zero(if2, M[2, 1], m_i_k)

    m_j_i = if_greater_zero(if2, M[0, 2], m_j_i)
    m_j_j = if_greater_zero(if2, M[0, 0], m_j_j)
    m_j_k = if_greater_zero(if2, M[0, 1], m_j_k)

    m_k_i = if_greater_zero(if2, M[1, 2], m_k_i)
    m_k_j = if_greater_zero(if2, M[1, 0], m_k_j)
    m_k_k = if_greater_zero(if2, M[1, 1], m_k_k)

    t = if_greater_zero(if0, t, m_i_i - (m_j_j + m_k_k) + M[3, 3])
    q[0] = if_greater_zero(if0, M[2, 1] - M[1, 2],
                           if_greater_zero(if2, m_i_j + m_j_i,
                                           if_greater_zero(if1, m_k_i + m_i_k, t)))
    q[1] = if_greater_zero(if0, M[0, 2] - M[2, 0],
                           if_greater_zero(if2, m_k_i + m_i_k,
                                           if_greater_zero(if1, t, m_i_j + m_j_i)))
    q[2] = if_greater_zero(if0, M[1, 0] - M[0, 1],
                           if_greater_zero(if2, t, if_greater_zero(if1, m_i_j + m_j_i,
                                                                   m_k_i + m_i_k)))
    q[3] = if_greater_zero(if0, t, m_k_j - m_j_k)

    q *= 0.5 / sqrt(t * M[3, 3])
    return q


def quaternion_multiply(q1: expr_matrix, q2: expr_matrix) -> expr_matrix:
    """
    :param q1: 4x1 Matrix
    :param q2: 4x1 Matrix
    :return: 4x1 Matrix
    """
    x0 = q2[0]
    y0 = q2[1]
    z0 = q2[2]
    w0 = q2[3]
    x1 = q1[0]
    y1 = q1[1]
    z1 = q1[2]
    w1 = q1[3]
    return Matrix([x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                   -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                   x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                   -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0])


def quaternion_conjugate(quaternion: expr_matrix) -> expr_matrix:
    """
    :param quaternion: 4x1 Matrix
    :return: 4x1 Matrix
    """
    return Matrix([-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]])


def quaternion_diff(q0: expr_matrix, q1: expr_matrix) -> expr_matrix:
    """
    :param q0: 4x1 Matrix
    :param q1: 4x1 Matrix
    :return: 4x1 Matrix p, such that q1*p=q2
    """
    return quaternion_multiply(quaternion_conjugate(q0), q1)


def cosine_distance(v0: expr_matrix, v1: expr_matrix) -> expr_symbol:
    """
    cosine distance ranging from 0 to 2
    :param v0: nx1 Matrix
    :param v1: nx1 Matrix
    """
    return 1 - ((dot(v0.T, v1))[0] / (norm(v0) * norm(v1)))


def euclidean_distance(v1: expr_matrix, v2: expr_matrix) -> expr_symbol:
    """
    :param v1: nx1 Matrix
    :param v2: nx1 Matrix
    """
    return norm(v1 - v2)


def fmod(a: expr_symbol, b: expr_symbol) -> expr_symbol:
    return ca.fmod(a, b)


def normalize_angle_positive(angle: expr_symbol) -> expr_symbol:
    """
    Normalizes the angle to be 0 to 2*pi
    It takes and returns radians.
    """
    return fmod(fmod(angle, 2.0 * pi) + 2.0 * pi, 2.0 * pi)


def normalize_angle(angle: expr_symbol) -> expr_symbol:
    """
    Normalizes the angle to be -pi to +pi
    It takes and returns radians.
    """
    a = normalize_angle_positive(angle)
    return if_greater(a, pi, a - 2.0 * pi, a)


def shortest_angular_distance(from_angle: expr_symbol, to_angle: expr_symbol) -> expr_matrix:
    """
    Given 2 angles, this returns the shortest angular
    difference.  The inputs and outputs are of course radians.

    The result would always be -pi <= result <= pi. Adding the result
    to "from" will always get you an equivalent angle to "to".
    """
    return normalize_angle(to_angle - from_angle)


def quaternion_slerp(q1: expr_matrix, q2: expr_matrix, t: expr_symbol) -> expr_matrix:
    """
    spherical linear interpolation that takes into account that q == -q
    :param q1: 4x1 Matrix
    :param q2: 4x1 Matrix
    :param t: float, 0-1
    :return: 4x1 Matrix; Return spherical linear interpolation between two quaternions.
    """
    cos_half_theta = dot(q1.T, q2)

    if0 = -cos_half_theta
    q2 = if_greater_zero(if0, -q2, q2)
    cos_half_theta = if_greater_zero(if0, -cos_half_theta, cos_half_theta)

    if1 = abs(cos_half_theta) - 1.0

    # enforce acos(x) with -1 < x < 1
    cos_half_theta = min(1, cos_half_theta)
    cos_half_theta = max(-1, cos_half_theta)

    half_theta = acos(cos_half_theta)

    sin_half_theta = sqrt(1.0 - cos_half_theta * cos_half_theta)
    if2 = 0.001 - abs(sin_half_theta)

    ratio_a = save_division(sin((1.0 - t) * half_theta), sin_half_theta)
    ratio_b = save_division(sin(t * half_theta), sin_half_theta)
    return if_greater_eq_zero(if1,
                              Matrix(q1),
                              if_greater_zero(if2,
                                              0.5 * q1 + 0.5 * q2,
                                              ratio_a * q1 + ratio_b * q2))


def scale_quaternion(q: expr_matrix, angle: expr_symbol) -> expr_matrix:
    axis, _ = axis_angle_from_quaternion(q[0], q[1], q[2], q[3])
    return quaternion_from_axis_angle(axis, angle)


def quaternion_angle(q: expr_matrix) -> expr_symbol:
    return axis_angle_from_quaternion(q[0], q[1], q[2], q[3])[1]


def slerp(v1: expr_matrix, v2: expr_matrix, t: float) -> expr_matrix:
    """
    spherical linear interpolation
    :param v1: any vector
    :param v2: vector of same length as v1
    :param t: value between 0 and 1. 0 is v1 and 1 is v2
    """
    angle = save_acos(dot(v1.T, v2)[0])
    angle2 = if_eq(angle, 0, 1, angle)
    return if_eq(angle, 0,
                 v1,
                 (sin((1 - t) * angle2) / sin(angle2)) * v1 + (sin(t * angle2) / sin(angle2)) * v2)


def to_numpy(matrix: expr_matrix) -> np.ndarray:
    return np.array(matrix.tolist()).astype(float).reshape(matrix.shape)


def save_division(nominator: any_expr, denominator: expr_symbol, if_nan: expr_symbol = 0) -> any_expr:
    save_denominator = if_eq_zero(denominator, 1, denominator)
    return nominator * if_eq_zero(denominator, if_nan, 1. / save_denominator)


def save_acos(angle: expr_symbol) -> expr_symbol:
    angle = limit(angle, -1, 1)
    return acos(angle)


def entrywise_product(matrix1: expr_matrix, matrix2: expr_matrix) -> expr_matrix:
    assert matrix1.shape == matrix2.shape
    result = zeros(*matrix1.shape)
    for i in range(matrix1.shape[0]):
        for j in range(matrix1.shape[1]):
            result[i, j] = matrix1[i, j] * matrix2[i, j]
    return result


def floor(x: expr_symbol) -> expr_symbol:
    return ca.floor(x)


def ceil(x: expr_symbol) -> expr_symbol:
    return ca.ceil(x)


def round_up(x: expr_symbol, decimal_places: expr_symbol) -> expr_symbol:
    f = 10 ** (decimal_places)
    return ceil(x * f) / f


def round_down(x: expr_symbol, decimal_places: expr_symbol) -> expr_symbol:
    f = 10 ** (decimal_places)
    return floor(x * f) / f


def sum(matrix: expr_matrix) -> expr_symbol:
    """
    the equivalent to np.sum(matrix)
    """
    return ca.sum1(ca.sum2(matrix))


def sum_row(matrix: expr_matrix) -> expr_matrix:
    """
    the equivalent to np.sum(matrix, axis=0)
    """
    return ca.sum1(matrix)


def sum_column(matrix: expr_matrix) -> expr_matrix:
    """
    the equivalent to np.sum(matrix, axis=1)
    """
    return ca.sum2(matrix)


def distance_point_to_line_segment(point: expr_matrix, line_start: expr_matrix, line_end: expr_matrix) \
        -> Tuple[expr_symbol, expr_matrix]:
    """
    :param point: current position of an object (i. e.) gripper tip
    :param line_start: start of the approached line
    :param line_end: end of the approached line
    :return: distance to line, the nearest point on the line
    """
    line_vec = line_end - line_start
    pnt_vec = point - line_start
    line_len = norm(line_vec)
    line_unitvec = line_vec / line_len
    pnt_vec_scaled = pnt_vec / line_len
    t = dot(line_unitvec.T, pnt_vec_scaled)[0]
    t = min(max(t, 0.0), 1.0)
    nearest = line_vec * t
    dist = norm(nearest - pnt_vec)
    nearest = nearest + line_start
    return dist, nearest


def angle_between_vector(v1: expr_matrix, v2: expr_matrix) -> expr_symbol:
    v1 = v1[:3]
    v2 = v2[:3]
    return acos(dot(v1.T, v2) / (norm(v1) * norm(v2)))


def angle_from_matrix(R: expr_matrix, hint: Callable) -> expr_symbol:
    """
    :param R:
    :param hint: A function who's sign of the result will be used to to determine if angle should be positive or
                    negative
    :return:
    """
    axis, angle = axis_angle_from_matrix(R)
    return normalize_angle(if_greater_zero(hint(axis),
                                           if_result=angle,
                                           else_result=-angle))


def velocity_limit_from_position_limit(acceleration_limit: expr_symbol,
                                       position_limit: expr_symbol,
                                       current_position: expr_symbol,
                                       step_size: expr_symbol,
                                       eps: float = 1e-5) -> expr_symbol:
    """
    Computes the velocity limit given a distance to the position limits, an acceleration limit and a step size
    :param acceleration_limit:
    :param step_size:
    :param eps: 
    :return: 
    """
    distance_to_position_limit = position_limit - current_position
    acceleration_limit *= step_size
    distance_to_position_limit /= step_size
    m = 1 / acceleration_limit
    acceleration_limit *= m
    distance_to_position_limit *= m
    sign_ = sign(distance_to_position_limit)
    error = abs(distance_to_position_limit)
    # reverse gausssche summenformel to compute n from sum
    n = sqrt(2 * error + (1 / 4)) - 1 / 2
    # round up if very close to the ceiling to avoid precision errors
    n = if_less(1 - (n - floor(n)), eps, np.ceil(n), np.floor(n))
    error_rounded = (n ** 2 + n) / 2
    rest = error - error_rounded
    rest = rest / (n + 1)
    velocity_limit = n + rest
    velocity_limit *= sign_
    velocity_limit /= m
    return velocity_limit


# def position_with_max_velocity(velocity_limit: expr_symbol, jerk_limit: expr_symbol) -> expr_symbol:
#     t = np.sqrt(np.abs(velocity_limit / jerk_limit))
#     return -t * velocity_limit


# def t_til_pos2(position_error, jerk_limit):
#     return (position_error / (2 * jerk_limit)) ** (1 / 3)


# def position_till_b(jerk_limit, t):
#     return (1 / 6) * jerk_limit * t ** 3


# def position_till_a(jerk_limit, t, t_offset, velocity_limit):
#     return (1 / 6) * jerk_limit * t ** 3 - 0.5 * jerk_limit * t_offset * t ** 2 + 0.5 * jerk_limit * t_offset ** 2 * t + velocity_limit * t


# def velocity(velocity_limit, jerk_limit, t):
#     t_b = np.sqrt(np.abs(velocity_limit / jerk_limit))
#     t_a = t_b * 2
#     if t < t_b:
#         return velocity_limit + 0.5 * jerk_limit * t ** 2
#     if t < t_a:
#         t -= t_a
#         return -0.5 * jerk_limit * t ** 2
#     return velocity_limit


# def position(jerk_limit, t, velocity_limit):
#     t_b = np.sqrt(np.abs(velocity_limit / jerk_limit))
#     t_a = t_b * 2
#     if t < t_b:
#         return (1 / 6) * jerk_limit * t ** 3 + velocity_limit * t - velocity_limit * t_b
#     if t < t_a:
#         t -= t_a
#         return -(1 / 6) * jerk_limit * t ** 3
#     return velocity_limit * t


# def compute_t_from_position(jerk_limit, position_error, velocity_limit):
#     t_b = np.sqrt(np.abs(velocity_limit / jerk_limit))
#     a = position_with_max_velocity(velocity_limit, jerk_limit)
#     b = -(1 / 6) * jerk_limit * (-t_b) ** 3
#     t_a = t_b * 2
#     if position_error < b:
#         asdf = (-(6 * position_error) / jerk_limit)
#         return np.sign(asdf) * np.abs(asdf) ** (1 / 3) + t_a
#     if position_error < a:
#         return np.real(-1.44224957030741 * (-0.5 - 0.866025403784439j) * \
#                        (((-t_b * velocity_limit - position_error) ** 2 / jerk_limit ** 2 + (
#                                8 / 9) * velocity_limit ** 3 / jerk_limit ** 3) ** (0.5 + 0j) + (1 / 6) *
#                         (-6.0 * t_b * velocity_limit - 6.0 * position_error) / jerk_limit) ** (1 / 3) \
#                        + 1.38672254870127 * velocity_limit * (-0.5 + 0.866025403784439j) / \
#                        (jerk_limit * (((-t_b * velocity_limit - position_error) ** 2 / jerk_limit ** 2 + (
#                                8 / 9) * velocity_limit ** 3 / jerk_limit ** 3) ** (0.5 + 0j)
#                                       + (1 / 6) * (
#                                               -6.0 * t_b * velocity_limit - 6.0 * position_error) / jerk_limit) ** (
#                                 1 / 3)))
#     return 0


# def jerk_limits_from_everything(position_limit, velocity_limit, jerk_limit, current_position, current_velocity,
#                                 current_acceleration, t, step_size, eps=1e-5):
#     """
#     Computes the velocity limit given a distance to the position limits, an acceleration limit and a step size
#     :param acceleration_limit:
#     :param distance_to_position_limit:
#     :param step_size:
#     :param eps:
#     :return:
#     """
#     # p(t) describes slowdown with max vel/jerk down to 0
#     # 1. get t from p(t)=position_limit - current_position
#     # 2. plug t into v(t) to get vel limit
#
#     a = position_with_max_velocity(velocity_limit, jerk_limit)
#     t_b = t_til_pos2(a, jerk_limit)
#     t_a = t_b * 2


def to_str(expression: expr_symbol) -> str:
    """
    Turns expression into a more or less readable string.
    """
    s = str(expression)
    parts = s.split(', ')
    result = parts[-1]
    for x in reversed(parts[:-1]):
        index, sub = x.split('=')
        if index not in result:
            raise Exception('fuck')
        result = result.replace(index, sub)
    return result
    pass


def total_derivative(expr: any_expr, symbols: expr_matrix, symbols_dot: expr_matrix) \
        -> any_expr:
    expr_jacobian = jacobian(expr, symbols)
    last_velocities = Matrix(symbols_dot)
    velocity = dot(expr_jacobian, last_velocities)
    if velocity.shape[0] * velocity.shape[0] == 1:
        return velocity[0]
    else:
        return velocity
