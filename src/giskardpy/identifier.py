hack = ['hack']
world = ['world']
robot_group_name = ['robot_name']
fk_pose = world + ['compute_fk_pose']
fk_np = world + ['compute_fk_np']
joint_states = world + ['state']
controlled_joints = ['controlled_joints']

fill_trajectory_velocity_values = ['fill_trajectory_velocity_values']

# goal_params = ['goal_params']
trajectory = ['traj']
debug_trajectory = ['lbA_traj']
time = ['time']
qp_solver_solution = ['qp_solver_solution']
# last_cmd = ['last_cmd']
# collisions = ['collisions']
goal_msg = ['goal_msg']
goals = ['goals']
drive_goals = ['drive_goals']
eq_constraints = ['eq_constraints']
neq_constraints = ['neq_constraints']
derivative_constraints = ['derivative_constraints']
free_variables = ['free_variables']
debug_expressions = ['debug_expressions']

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
tracking_start_time = ['tracking_start_time']

# stuff from rosparam
robot_descriptions = ['robot_descriptions']

giskard = ['giskard']

# config file
# general options
general_options = giskard + ['_general_config']
max_derivative = general_options + ['maximum_derivative']
action_server_name = general_options + ['action_server_name']
tmp_folder = general_options + ['path_to_data_folder']
debug_expr_needed = ['debug_expr_needed']
test_mode = general_options + ['test_mode']
control_mode = general_options + ['control_mode']

joint_limits = general_options + ['joint_limits']


# qp solver
qp_solver_config = giskard + ['_qp_solver_config']
sample_period = qp_solver_config + ['sample_period']
qp_controller = giskard + ['qp_controller']
debug_expressions_evaluated = qp_controller + ['evaluated_debug_expressions']
joint_weights = qp_solver_config + ['joint_weights']
qp_solver_name = qp_solver_config + ['qp_solver']
prediction_horizon = qp_solver_config + ['prediction_horizon']
retries_with_relaxed_constraints = qp_solver_config + ['retries_with_relaxed_constraints']
retry_added_slack = qp_solver_config + ['added_slack']
retry_weight_factor = qp_solver_config + ['weight_factor']

# tree
plugins = giskard + ['behavior_tree_config', 'plugin_config']
enable_VisualizationBehavior = plugins + ['VisualizationBehavior', 'enabled']
VisualizationBehavior_in_planning_loop = plugins + ['VisualizationBehavior', 'in_planning_loop']
enable_WorldVisualizationBehavior = plugins + ['WorldVisualizationBehavior', 'enabled']
enable_CPIMarker = plugins + ['CollisionMarker', 'enabled']
CPIMarker_in_planning_loop = plugins + ['CollisionMarker', 'in_planning_loop']

PlotTrajectory = plugins + ['PlotTrajectory']
PlotTrajectory_enabled = PlotTrajectory + ['enabled']

PlotDebugTrajectory = plugins + ['PlotDebugExpressions']
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
TFPublisher_enabled = TFPublisher + ['enabled']

SyncOdometry = plugins + ['SyncOdometry']

SyncTfFrames = plugins + ['SyncTfFrames']
frames_to_add = ['frames_to_add']
joints_to_add = ['joints_to_add']

PublishDebugExpressions = plugins + ['PublishDebugExpressions']

# behavior tree
tree_manager = giskard + ['_tree']
tree_tick_rate = giskard + ['behavior_tree_config', 'tree_tick_rate']

# collision avoidance
collision_avoidance_configs = giskard + ['_collision_avoidance_configs']
collision_scene = ['collision_scene']
collision_matrix = ['collision_matrix']
closest_point = ['cpi']
added_collision_checks = ['added_collision_checks']

collision_checker = giskard + ['collision_checker']

# robot interface
robot_interface_configs = giskard + ['robot_interface_configs']
hardware_config = giskard + ['hardware_config']

# rnd stuff
timer_collector = ['timer_collector']
