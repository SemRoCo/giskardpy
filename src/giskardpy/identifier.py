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
drive_goals = ['drive_goals']
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
tracking_start_time = ['tracking_start_time']

# stuff from rosparam
robot_description = ['robot_description']

giskard = ['giskard']

# config file
# general options
general_options = giskard + ['general_config']
action_server_name = general_options + ['action_server_name']
data_folder = general_options + ['path_to_data_folder']
debug = general_options + ['debug']
test_mode = general_options + ['test_mode']

joint_limits = general_options + ['joint_limits']

joint_acceleration_linear_limit = general_options + ['joint_limits', 'acceleration']
joint_acceleration_angular_limit = general_options + ['joint_limits', 'acceleration']

joint_jerk_linear_limit = general_options + ['joint_limits', 'jerk']
joint_jerk_angular_limit = general_options + ['joint_limits', 'jerk']


# qp solver
qp_solver_config = giskard + ['qp_solver_config']
sample_period = qp_solver_config + ['sample_period']
qp_controller = giskard + ['qp_controller']
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

SyncOdometry = plugins + ['SyncOdometry']

SyncTfFrames = plugins + ['SyncTfFrames']
SyncTfFrames_frames = SyncTfFrames + ['frames']

PublishDebugExpressions = plugins + ['PublishDebugExpressions']

# reachability check
reachability_check = giskard + ['reachability_check']
rc_sample_period = reachability_check + ['sample_period']
rc_prismatic_velocity = reachability_check + ['prismatic_velocity']
rc_continuous_velocity = reachability_check + ['continuous_velocity']
rc_revolute_velocity = reachability_check + ['revolute_velocity']
rc_other_velocity = reachability_check + ['other_velocity']

# behavior tree
tree_manager = giskard + ['_tree']
tree_tick_rate = giskard + ['behavior_tree_config', 'tree_tick_rate']

# collision avoidance
collision_avoidance_config = giskard + ['collision_avoidance_config']
collision_scene = ['collision_scene']
collision_matrix = ['collision_matrix']
closest_point = ['cpi']
added_collision_checks = ['added_collision_checks']

collision_checker = collision_avoidance_config + ['collision_checker']

self_collision_avoidance = collision_avoidance_config + ['_self_collision_avoidance']
added_self_collisions = collision_avoidance_config + ['_add_self_collisions']
ignored_self_collisions = collision_avoidance_config + ['_ignored_self_collisions']

external_collision_avoidance = collision_avoidance_config + ['_external_collision_avoidance']

# robot interface
robot_interface_config = giskard + ['robot_interface_config']
robot_interface = robot_interface_config + ['follow_joint_trajectory_interfaces']
robot_base_drive = robot_interface_config + ['drive_interface']
joint_state_topic = robot_interface_config + ['joint_state_topic']

# rnd stuff
timer_collector = ['timer_collector']
