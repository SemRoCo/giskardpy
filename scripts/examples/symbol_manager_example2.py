from geometry_msgs.msg import PoseStamped

from giskardpy.god_map import god_map
from giskardpy.symbol_manager import symbol_manager

god_map.muh = PoseStamped()
god_map.muh.pose.orientation.w = 1

pose = symbol_manager.get_expr('god_map.muh')
print(pose)
# @1=sq(god_map.muh.pose.orientation.w),
# @2=sq(god_map.muh.pose.orientation.x),
# @3=sq(god_map.muh.pose.orientation.y),
# @4=sq(god_map.muh.pose.orientation.z),
# @5=2,
# @6=0,
# [[(((@1+@2)-@3)-@4), (((@5*god_map.muh.pose.orientation.x)*god_map.muh.pose.orientation.y)-((@5*god_map.muh.pose.orientation.w)*god_map.muh.pose.orientation.z)), (((@5*god_map.muh.pose.orientation.x)*god_map.muh.pose.orientation.z)+((@5*god_map.muh.pose.orientation.w)*god_map.muh.pose.orientation.y)), god_map.muh.pose.position.x],
#  [(((@5*god_map.muh.pose.orientation.x)*god_map.muh.pose.orientation.y)+((@5*god_map.muh.pose.orientation.w)*god_map.muh.pose.orientation.z)), (((@1-@2)+@3)-@4), (((@5*god_map.muh.pose.orientation.y)*god_map.muh.pose.orientation.z)-((@5*god_map.muh.pose.orientation.w)*god_map.muh.pose.orientation.x)), god_map.muh.pose.position.y],
#  [(((@5*god_map.muh.pose.orientation.x)*god_map.muh.pose.orientation.z)-((@5*god_map.muh.pose.orientation.w)*god_map.muh.pose.orientation.y)), (((@5*god_map.muh.pose.orientation.y)*god_map.muh.pose.orientation.z)+((@5*god_map.muh.pose.orientation.w)*god_map.muh.pose.orientation.x)), (((@1-@2)-@3)+@4), god_map.muh.pose.position.z],
#  [@6, @6, @6, 1]]


pose_compiled = pose.compile()
args = symbol_manager.resolve_symbols(pose_compiled.str_params)
print(pose_compiled.fast_call(args))
# [[1. 0. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 0. 1. 0.]
#  [0. 0. 0. 1.]]

god_map.muh.pose.position.x = 1
god_map.muh.pose.orientation.x = 1
god_map.muh.pose.orientation.w = 0

args = symbol_manager.resolve_symbols(pose_compiled.str_params)
print(pose_compiled.fast_call(args))
# [[ 1.  0.  0.  1.]
#  [ 0. -1.  0.  0.]
#  [ 0.  0. -1.  0.]
#  [ 0.  0.  0.  1.]]
