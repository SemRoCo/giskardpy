world = [u'world']
robot = world + [u'robot']
fk_pose = robot + [u'get_fk_pose']
fk_np = robot + [u'get_fk_np']
joint_states = robot + [u'joint_state']

constraints_identifier = [u'constraints']
trajectory = [u'traj']
time = [u'time']
cmd = [u'cmd']
last_cmd = [u'last_cmd']
closest_point = [u'cpi']
collisions = [u'collisions']
collision_goal_identifier = [u'collision_goal']
soft_constraint_identifier = [u'soft_constraints']
execute = [u'execute']
check_reachability = [u'check_reachability']
next_move_goal = [u'next_move_goal']
qp_data = [u'qp_data']
A = qp_data + [u'A']
H = qp_data + [u'H']
lbA = qp_data + [u'lbA']
ubA = qp_data + [u'ubA']
lb = qp_data + [u'lb']
ub = qp_data + [u'ub']
xdot_full = qp_data + [u'xdot_full']
weight_keys = qp_data + [u'weight_keys']
b_keys = qp_data + [u'b_keys']
bA_keys = qp_data + [u'bA_keys']
xdot_keys = qp_data + [u'xdot_keys']

wiggle_detection_samples = [u'wiggle_detection_samples']
post_processing = [u'post_processing']
soft_constraints = post_processing + [u'soft_constraints']




#stuff from rosparam
robot_description = [u'robot_description']

rosparam = [u'rosparam']
gui = rosparam + [u'enable_gui']
data_folder = rosparam + [u'path_to_data_folder']


# config file
# general options
general_options = rosparam + [u'general_options']
sample_period = general_options + [u'sample_period']
map_frame = general_options + [u'map_frame']
debug = general_options + [u'debug']
fill_velocity_values = general_options + [u'fill_velocity_values']
joint_convergence_threshold = general_options + [u'joint_convergence_threshold']

joint_velocity_linear_limit = general_options + [u'joint_vel_limit', u'linear']
default_joint_velocity_linear_limit = joint_velocity_linear_limit + [u'default']

joint_velocity_angular_limit = general_options + [u'joint_vel_limit', u'angular']
default_joint_velocity_angular_limit = joint_velocity_angular_limit + [u'default']

joint_acceleration_linear_limit = general_options + [u'joint_acceleration_limit', u'linear']
default_joint_acceleration_linear_limit = joint_acceleration_linear_limit + [u'default']

joint_acceleration_angular_limit = general_options + [u'joint_acceleration_limit', u'angular']
default_joint_acceleration_angular_limit = joint_acceleration_angular_limit + [u'default']

joint_cost = general_options + [u'joint_weights']
default_joint_cost_identifier = joint_cost + [u'default']

# qp solver
qp_solver = rosparam + [u'qp_solver']
nWSR = qp_solver + [u'nWSR']

# plugins
plugins = rosparam + [u'plugins']
enable_VisualizationBehavior = plugins + [u'VisualizationBehavior', u'enabled']
enable_CPIMarker = plugins + [u'CPIMarker', u'enabled']
enable_PlotTrajectory = plugins + [u'PlotTrajectory', u'enabled']
wiggle_detection_threshold = plugins + [u'WiggleCancel', u'wiggle_detection_threshold']
num_samples_in_fft = plugins + [u'WiggleCancel', u'num_samples_in_fft']
wiggle_frequency_range = plugins + [u'WiggleCancel', u'wiggle_frequency_range']

# reachability check
reachability_check = rosparam + [u'reachability_check']
rc_sample_period = reachability_check + [u'rc_sample_period']
rc_prismatic_velocity = reachability_check + [u'rc_prismatic_velocity']
rc_continuous_velocity = reachability_check + [u'_continuous_velocity']
rc_revolute_velocity = reachability_check + [u'rc_revolute_velocity']
rc_other_velocity = reachability_check + [u'rc_other_velocity']


# behavior tree
behavior_tree = rosparam + [u'behavior_tree']
tree_tick_rate = behavior_tree + [u'tree_tick_rate']

# collision avoidance
collision_avoidance = rosparam + [u'collision_avoidance']

distance_thresholds = collision_avoidance + [u'distance_thresholds']
default_collision_distances = distance_thresholds + [u'default']

self_collision_avoidance = collision_avoidance + [u'self_collision_avoidance']
ignored_self_collisions = self_collision_avoidance + [u'ignore']
added_self_collisions = self_collision_avoidance + [u'add']

external_collision_avoidance = collision_avoidance + [u'external_collision_avoidance']
number_of_repeller = external_collision_avoidance + [u'number_of_repeller']



