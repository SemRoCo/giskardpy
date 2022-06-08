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

# stuff from rosparam
robot_description = ['robot_description']

giskard = ['giskard']

# config file
# general options
action_server_name = giskard + ['action_server_name']
gui = giskard + ['enable_gui']
data_folder = giskard + ['path_to_data_folder']
sample_period = giskard + ['sample_period']
map_frame = giskard + ['map_frame']
debug = giskard + ['debug']
fill_velocity_values = giskard + ['fill_velocity_values']
test_mode = giskard + ['test_mode']

joint_limits = giskard + ['joint_limits']

joint_acceleration_linear_limit = giskard + ['joint_limits', 'acceleration']
joint_acceleration_angular_limit = giskard + ['joint_limits', 'acceleration']

joint_jerk_linear_limit = giskard + ['joint_limits', 'jerk']
joint_jerk_angular_limit = giskard + ['joint_limits', 'jerk']

joint_weights = giskard + ['joint_weights']

# qp solver
qp_controller = giskard + ['qp_controller']
qp_solver = giskard + ['qp_solver_config']
qp_solver_name = qp_solver + ['qp_solver']
prediction_horizon = giskard + ['prediction_horizon']
retries_with_relaxed_constraints = qp_solver + ['retries_with_relaxed_constraints']
retry_added_slack = qp_solver + ['added_slack']
retry_weight_factor = qp_solver + ['weight_factor']

# tree
plugins = giskard + ['plugin_config']
enable_VisualizationBehavior = plugins + ['VisualizationBehavior', 'enabled']
VisualizationBehavior_in_planning_loop = plugins + ['VisualizationBehavior', 'in_planning_loop']
enable_WorldVisualizationBehavior = plugins + ['WorldVisualizationBehavior', 'enabled']
enable_CPIMarker = plugins + ['CollisionMarker', 'enabled']
CPIMarker_in_planning_loop = plugins + ['CollisionMarker', 'in_planning_loop']

PlotTrajectory = plugins + ['PlotTrajectory']
PlotTrajectory_enabled = PlotTrajectory + ['enabled']

PlotDebugTrajectory = plugins + ['PlotDebugExpressions']
PlotDebugTrajectory_enabled = PlotDebugTrajectory + ['enabled']

MaxTrajectoryLength = plugins + ['MaxTrajectoryLength']
MaxTrajectoryLength_enabled = MaxTrajectoryLength + ['enabled']

fft_duration = plugins + ['WiggleCancel', 'fft_duration']
amplitude_threshold = plugins + ['WiggleCancel', 'amplitude_threshold']
num_samples_in_fft = plugins + ['WiggleCancel', 'window_size']
frequency_range = plugins + ['WiggleCancel', 'frequency_range']

joint_convergence_threshold = plugins + ['GoalReached', 'joint_convergence_threshold']
GoalReached_window_size = plugins + ['GoalReached', 'window_size']

TFPublisher = plugins + ['TFPublisher']

SyncOdometry = plugins + ['SyncOdometry']

SyncTfFrames = plugins + ['SyncTfFrames']
SyncTfFrames_frames = SyncTfFrames + ['frames']

# reachability check
reachability_check = giskard + ['reachability_check']
rc_sample_period = reachability_check + ['sample_period']
rc_prismatic_velocity = reachability_check + ['prismatic_velocity']
rc_continuous_velocity = reachability_check + ['continuous_velocity']
rc_revolute_velocity = reachability_check + ['revolute_velocity']
rc_other_velocity = reachability_check + ['other_velocity']

# behavior tree
tree_manager = giskard + ['tree']
tree_tick_rate = giskard + ['tree_tick_rate']

# collision avoidance
collision_scene = ['collision_scene']
collision_matrix = ['collision_matrix']
closest_point = ['cpi']
added_collision_checks = ['added_collision_checks']

collision_avoidance = giskard + ['collision_avoidance']
collision_checker = giskard + ['collision_checker']

self_collision_avoidance = giskard + ['self_collision_avoidance']
added_self_collisions = giskard + ['add_self_collisions']
ignored_self_collisions = giskard + ['ignored_self_collisions']

external_collision_avoidance = giskard + ['external_collision_avoidance']

# robot interface
robot_interface = giskard + ['follow_joint_trajectory_interfaces']
robot_base_drive = giskard + ['drive_interface']

# rnd stuff
timer_collector = ['timer_collector']
