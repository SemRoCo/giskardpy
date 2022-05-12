world = [u'world']
robot = world + [u'robot']
fk_pose = robot + [u'get_fk_pose']
fk_np = robot + [u'get_fk_np']
joint_states = robot + [u'joint_state']
last_joint_states = [u'last_joint_state']

constraints_identifier = [u'constraints']
trajectory = [u'traj']
time = [u'time']
cmd = [u'cmd']
# last_cmd = [u'last_cmd']
closest_point = [u'cpi']
# collisions = [u'collisions']
collision_goal = [u'collision_goal']
soft_constraint_identifier = [u'soft_constraints']
joint_constraint_identifier = [u'joint_constraints']
hard_constraint_identifier = [u'hard_constraints']

execute = [u'execute']
skip_failures = [u'skip_failures']
check_reachability = [u'check_reachability']
cut_off_shaking = [u'cut_off_shaking']
next_move_goal = [u'next_move_goal']
cmd_id = [u'cmd_id']

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

post_processing = [u'post_processing']
soft_constraints = post_processing + [u'soft_constraints']
result_message = [u'result_message']




#stuff from rosparam
robot_description = [u'robot_description']

rosparam = [u'rosparam']


# config file
# general options
general_options = rosparam + [u'general_options']
gui = general_options + [u'enable_gui']
data_folder = general_options + [u'path_to_data_folder']
sample_period = general_options + [u'sample_period']
map_frame = general_options + [u'map_frame']
debug = general_options + [u'debug']
fill_velocity_values = general_options + [u'fill_velocity_values']

joint_velocity_linear_limit = general_options + [u'joint_vel_limit', u'linear']
joint_velocity_linear_limit_default = joint_velocity_linear_limit + [u'default']
joint_velocity_linear_limit_override = joint_velocity_linear_limit + [u'override']

joint_velocity_angular_limit = general_options + [u'joint_vel_limit', u'angular']
joint_velocity_angular_limit_default = joint_velocity_angular_limit + [u'default']
joint_velocity_angular_limit_override = joint_velocity_angular_limit + [u'override']

joint_acceleration_linear_limit = general_options + [u'joint_acceleration_limit', u'linear']
joint_acceleration_linear_limit_default = joint_acceleration_linear_limit + [u'default']
joint_acceleration_linear_limit_override = joint_acceleration_linear_limit + [u'override']

joint_acceleration_angular_limit = general_options + [u'joint_acceleration_limit', u'angular']
joint_acceleration_angular_limit_default = joint_acceleration_angular_limit + [u'default']
joint_acceleration_angular_limit_override = joint_acceleration_angular_limit + [u'override']

joint_weight = general_options + [u'joint_weights']
joint_weight_default = joint_weight + [u'default']
joint_weight_override = joint_weight + [u'override']

# qp solver
qp_solver = rosparam + [u'qp_solver']
nWSR = qp_solver + [u'nWSR']

# plugins
plugins = rosparam + [u'plugins']
enable_VisualizationBehavior = plugins + [u'VisualizationBehavior', u'enabled']
enable_WorldVisualizationBehavior = plugins + [u'WorldVisualizationBehavior', u'enabled']
enable_CPIMarker = plugins + [u'CPIMarker', u'enabled']
enable_PlotTrajectory = plugins + [u'PlotTrajectory', u'enabled']
PlotTrajectory_velocity_threshold = plugins + [u'PlotTrajectory', u'velocity_threshold']
PlotTrajectory_scaling = plugins + [u'PlotTrajectory', u'scaling']
PlotTrajectory_normalize_position = plugins + [u'PlotTrajectory', u'normalize_position']
PlotTrajectory_tick_stride = plugins + [u'PlotTrajectory', u'tick_stride']
fft_duration = plugins + [u'WiggleCancel', u'fft_duration']
amplitude_threshold = plugins + [u'WiggleCancel', u'amplitude_threshold']
num_samples_in_fft = plugins + [u'WiggleCancel', u'window_size']
frequency_range = plugins + [u'WiggleCancel', u'frequency_range']

joint_convergence_threshold = plugins + [u'GoalReached', u'joint_convergence_threshold']
GoalReached_window_size = plugins + [u'GoalReached', u'window_size']

LoopDetector_precision = plugins + [u'LoopDetector', u'precision']

publish_attached_objects = plugins + [u'tf_publisher', u'publish_attached_objects']
publish_world_objects = plugins + [u'tf_publisher', u'publish_world_objects']
tf_topic = plugins + [u'tf_publisher', u'tf_topic']

# reachability check
reachability_check = rosparam + [u'reachability_check']
rc_sample_period = reachability_check + [u'sample_period']
rc_prismatic_velocity = reachability_check + [u'prismatic_velocity']
rc_continuous_velocity = reachability_check + [u'continuous_velocity']
rc_revolute_velocity = reachability_check + [u'revolute_velocity']
rc_other_velocity = reachability_check + [u'other_velocity']


# behavior tree
behavior_tree = rosparam + [u'behavior_tree']
tree_tick_rate = behavior_tree + [u'tree_tick_rate']
tree_manager = behavior_tree + [u'tree_manager']

# collision avoidance
collision_avoidance = rosparam + [u'collision_avoidance']
maximum_collision_threshold = collision_avoidance + [u'maximum_collision_threshold']
added_collision_checks = collision_avoidance + [u'added_collision_checks']

self_collision_avoidance = collision_avoidance + [u'self_collision_avoidance']
self_collision_avoidance_distance = self_collision_avoidance + [u'distance_thresholds']
self_collision_avoidance_default_threshold = self_collision_avoidance_distance + [u'default']
self_collision_avoidance_default_override = self_collision_avoidance_distance + [u'override']
ignored_self_collisions = self_collision_avoidance + [u'ignore']
added_self_collisions = self_collision_avoidance + [u'add']
self_collision_avoidance_repeller = self_collision_avoidance + [u'number_of_repeller']

external_collision_avoidance = collision_avoidance + [u'external_collision_avoidance']
external_collision_avoidance_distance = external_collision_avoidance + [u'distance_thresholds']
external_collision_avoidance_default_threshold = external_collision_avoidance_distance + [u'default']
external_collision_avoidance_default_override = external_collision_avoidance_distance + [u'override']
external_collision_avoidance_repeller = external_collision_avoidance + [u'number_of_repeller', u'default']
external_collision_avoidance_repeller_eef = external_collision_avoidance + [u'number_of_repeller', u'end_effector_joints']



