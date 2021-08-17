world = [u'world']
robot = world + [u'robot']
fk_pose = robot + [u'get_fk_pose']
fk_np = robot + [u'get_fk_np']
joint_states = robot + [u'joint_state']

# goal_params = [u'goal_params']
trajectory = [u'traj']
order = [u'order']
debug_trajectory = [u'lbA_traj']
time = [u'time']
qp_solver_solution = [u'qp_solver_solution']
# last_cmd = [u'last_cmd']
closest_point = [u'cpi']
# collisions = [u'collisions']
collision_goal = [u'collision_goal']
constraints = [u'constraints']
goals = [u'goals']
vel_constraints = [u'vel_constraints']
free_variables = [u'free_variables']
debug_expressions = [u'debug_expressions']
debug_expressions_evaluated = [u'debug_expressions_evaluated']

execute = [u'execute']
skip_failures = [u'skip_failures']
check_reachability = [u'check_reachability']
cut_off_shaking = [u'cut_off_shaking']
next_move_goal = [u'next_move_goal']
cmd_id = [u'cmd_id']

post_processing = [u'post_processing']
soft_constraints = post_processing + [u'soft_constraints']
result_message = [u'result_message']

# stuff from rosparam
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
test_mode = general_options + [u'test_mode']

joint_limits = general_options + [u'joint_limits']

joint_acceleration_linear_limit = general_options + [u'joint_limits', u'acceleration', u'linear', u'override']
joint_acceleration_angular_limit = general_options + [u'joint_limits', u'acceleration', u'angular', u'override']

joint_jerk_linear_limit = general_options + [u'joint_limits', u'jerk', u'linear', u'override']
joint_jerk_angular_limit = general_options + [u'joint_limits', u'jerk', u'angular', u'override']

joint_weights = general_options + [u'joint_weights']

# qp solver
qp_solver = rosparam + [u'qp_solver']
qp_solver_name = qp_solver + [u'name']
prediction_horizon = qp_solver + [u'prediction_horizon']
retries_with_relaxed_constraints = qp_solver + [u'hard_constraint_handling', u'retries_with_relaxed_constraints']
retry_added_slack = qp_solver + [u'hard_constraint_handling', u'added_slack']
retry_weight_factor = qp_solver + [u'hard_constraint_handling', u'weight_factor']

# tree
plugins = rosparam + [u'plugins']
enable_VisualizationBehavior = plugins + [u'VisualizationBehavior', u'enabled']
enable_WorldVisualizationBehavior = plugins + [u'WorldVisualizationBehavior', u'enabled']
enable_CPIMarker = plugins + [u'CPIMarker', u'enabled']

PlotTrajectory = plugins + [u'PlotTrajectory']
PlotTrajectory_enabled = PlotTrajectory + [u'enabled']

PlotDebugTrajectory = plugins + [u'PlotDebugTrajectory']
PlotDebugTrajectory_enabled = PlotDebugTrajectory + [u'enabled']

MaxTrajectoryLength = plugins + [u'MaxTrajectoryLength']
MaxTrajectoryLength_enabled = MaxTrajectoryLength + [u'enabled']

fft_duration = plugins + [u'WiggleCancel', u'fft_duration']
amplitude_threshold = plugins + [u'WiggleCancel', u'amplitude_threshold']
num_samples_in_fft = plugins + [u'WiggleCancel', u'window_size']
frequency_range = plugins + [u'WiggleCancel', u'frequency_range']

joint_convergence_threshold = plugins + [u'GoalReached', u'joint_convergence_threshold']
GoalReached_window_size = plugins + [u'GoalReached', u'window_size']

TFPublisher = plugins + [u'TFPublisher']

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
added_collision_checks = collision_avoidance + [u'added_collision_checks']

self_collision_avoidance = collision_avoidance + [u'self_collision_avoidance', u'override']
ignored_self_collisions = self_collision_avoidance[:-1] + [u'ignore']
added_self_collisions = self_collision_avoidance[:-1] + [u'add']

external_collision_avoidance = collision_avoidance + [u'external_collision_avoidance', u'override']
