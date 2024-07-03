import giskardpy.casadi_wrapper as cas

a_T_b = cas.TransMatrix()
b_T_c = cas.TransMatrix()

c_P_x = cas.Point3()
c_V_x = cas.Vector3()

a_T_c = a_T_b.dot(b_T_c)
a_P_x = a_T_c.dot(c_P_x)
a_V_x = a_T_c.dot(c_V_x)
