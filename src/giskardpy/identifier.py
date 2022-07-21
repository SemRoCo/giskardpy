world = ['world']
robot_group_name = ['robot_name']
fk_pose = world + ['compute_fk_pose']
fk_np = world + ['compute_fk_np']
joint_states = world + ['state']
controlled_joints = ['controlled_joints']

# goal_params = ['goal_params']
trajectory = ['traj']
debug_trajectory = ['lbA_traj']
time = ['time']
qp_solver_solution = ['qp_solver_solution']
# last_cmd = ['last_cmd']
# collisions = ['collisions']
goal_msg = ['goal_msg']
goals = ['goals']
constraints = ['constraints']
vel_constraints = ['vel_constraints']
free_variables = ['free_variables']
debug_expressions = ['debug_expressions']
debug_expressions_evaluated = ['debug_expressions_evaluated']

execute = ['execute']
skip_failures = ['skip_failures']
check_reachability = ['check_reachability']
cut_off_shaking = ['cut_off_shaking']
next_move_goal = ['next_move_goal']
number_of_move_cmds = ['number_of_move_cmds']
cmd_id = ['cmd_id']

post_processing = ['post_processing']
soft_constraints = post_processing + ['soft_constraints']
result_message = ['result_message']

# stuff from rosparam
robot_description = ['robot_description']

rosparam = ['rosparam']

# config file
# general options
general_options = rosparam + ['general_options']
control_mode = general_options + ['mode']
action_server_name = general_options + ['action_server_name']
gui = general_options + ['enable_gui']
data_folder = general_options + ['path_to_data_folder']
sample_period = general_options + ['sample_period']
map_frame = general_options + ['map_frame']
debug = general_options + ['debug']
fill_velocity_values = general_options + ['fill_velocity_values']
test_mode = general_options + ['test_mode']

joint_limits = general_options + ['joint_limits']

joint_acceleration_linear_limit = general_options + ['joint_limits', 'acceleration', 'linear', 'override']
joint_acceleration_angular_limit = general_options + ['joint_limits', 'acceleration', 'angular', 'override']

joint_jerk_linear_limit = general_options + ['joint_limits', 'jerk', 'linear', 'override']
joint_jerk_angular_limit = general_options + ['joint_limits', 'jerk', 'angular', 'override']

joint_weights = general_options + ['joint_weights']

# qp solver
qp_solver = rosparam + ['qp_solver']
qp_solver_name = qp_solver + ['name']
prediction_horizon = qp_solver + ['prediction_horizon']
retries_with_relaxed_constraints = qp_solver + ['hard_constraint_handling', 'retries_with_relaxed_constraints']
retry_added_slack = qp_solver + ['hard_constraint_handling', 'added_slack']
retry_weight_factor = qp_solver + ['hard_constraint_handling', 'weight_factor']

# tree
plugins = rosparam + ['plugins']
enable_VisualizationBehavior = plugins + ['VisualizationBehavior', 'enabled']
VisualizationBehavior_in_planning_loop = plugins + ['VisualizationBehavior', 'in_planning_loop']
enable_WorldVisualizationBehavior = plugins + ['WorldVisualizationBehavior', 'enabled']
enable_CPIMarker = plugins + ['CPIMarker', 'enabled']
CPIMarker_in_planning_loop = plugins + ['CPIMarker', 'in_planning_loop']

PlotTrajectory = plugins + ['PlotTrajectory']
PlotTrajectory_enabled = PlotTrajectory + ['enabled']

PlotDebugTrajectory = plugins + ['PlotDebugTrajectory']
PlotDebugTrajectory_enabled = PlotDebugTrajectory + ['enabled']

PlotDebugTF = plugins + ['PlotDebugTF']
PlotDebugTF_enabled = PlotDebugTF + ['enabled']

MaxTrajectoryLength = plugins + ['MaxTrajectoryLength']
MaxTrajectoryLength_enabled = MaxTrajectoryLength + ['enabled']

fft_duration = plugins + ['WiggleCancel', 'fft_duration']
amplitude_threshold = plugins + ['WiggleCancel', 'amplitude_threshold']
num_samples_in_fft = plugins + ['WiggleCancel', 'window_size']
frequency_range = plugins + ['WiggleCancel', 'frequency_range']

LoopDetector_precision = plugins + [u'LoopDetector', u'precision']

joint_convergence_threshold = plugins + ['GoalReached', 'joint_convergence_threshold']
GoalReached_window_size = plugins + ['GoalReached', 'window_size']

TFPublisher = plugins + ['TFPublisher']

# reachability check
reachability_check = rosparam + ['reachability_check']
rc_sample_period = reachability_check + ['sample_period']
rc_prismatic_velocity = reachability_check + ['prismatic_velocity']
rc_continuous_velocity = reachability_check + ['continuous_velocity']
rc_revolute_velocity = reachability_check + ['revolute_velocity']
rc_other_velocity = reachability_check + ['other_velocity']

# behavior tree
tree_manager = ['tree_manager']
behavior_tree = rosparam + ['behavior_tree']
tree_tick_rate = behavior_tree + ['tree_tick_rate']

# collision avoidance
collision_scene = ['collision_scene']
collision_matrix = ['collision_matrix']
closest_point = ['cpi']
added_collision_checks = ['added_collision_checks']

collision_avoidance = rosparam + ['collision_avoidance']
collision_checker = collision_avoidance + ['collision_checker']

self_collision_avoidance = collision_avoidance + ['self_collision_avoidance', 'override']
ignored_self_collisions = self_collision_avoidance[:-1] + ['ignore']
added_self_collisions = self_collision_avoidance[:-1] + ['add']

external_collision_avoidance = collision_avoidance + ['external_collision_avoidance', 'override']

# robot interface
robot_interface = rosparam + ['robot_interface']

# rnd stuff
timer_collector = ['timer_collector']
