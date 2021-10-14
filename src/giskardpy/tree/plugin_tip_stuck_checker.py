from copy import deepcopy

import numpy as np
import yaml
from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.tree.plugin import GiskardBehavior
from giskardpy.utils.tfwrapper import pose_stamped_to_np
from giskardpy.utils.utils import trajectory_to_np


class TipStuckChecker(GiskardBehavior):
    def __init__(self, name):
        super(TipStuckChecker, self).__init__(name)
        self.min_trajectory_time_length = 1.0
        self.max_trajectory_time_length = 2.0
        self.relative_joint_movement_factor = 0.0001
        self.vel_atol = 0.01 # 1cm/s
        self.pos_atol = 0.001 # 1mm
        self.sample_period = self.get_god_map().get_data(identifier.sample_period)
        self.max_trajectory_length = int(self.max_trajectory_time_length/self.sample_period)
        self.tip_poses = list() # type: List[geometry_msgs.msg.PoseStamped]
        self.tip_link = None

    def _get_tip_pose_np(self):
        goal = self.get_god_map().get_data(identifier.next_move_goal)
        if goal:
            for constraint in goal.constraints:
                if constraint.type == 'CartesianPosition':
                    d = yaml.load(constraint.parameter_value_pair)
                    tip_link = d['tip_link']
                    tip_pose = self.get_robot().get_fk_pose(self.get_robot().get_root(), tip_link)
                    p, q = pose_stamped_to_np(tip_pose)
                    return [p, q], tip_link
        return None, None

    def _log_tip_pose_np(self):
        tip_pose, tip_link = self._get_tip_pose_np()
        if tip_pose is not None and tip_link is not None:
            if tip_link != self.tip_link:
                self.tip_link = tip_link
                self.tip_poses = list()
            if len(self.tip_poses) >= self.max_trajectory_length:
                self.tip_poses = deepcopy(self.tip_poses[1:])
            self.tip_poses.append(tip_pose)

    def _is_tip_pose_repeated(self):
        checked_poses = []
        repetitions = 0

        cur_pose_np, _ = self._get_tip_pose_np()
        if cur_pose_np is not None:

            p = cur_pose_np[0]
            cur_pos_np_repeated = np.repeat([p], len(self.tip_poses), axis=0)
            tip_pos_np = np.concatenate(np.array(self.tip_poses)[:,0]).reshape(cur_pos_np_repeated.shape)
            distances = np.sqrt(np.sum((cur_pos_np_repeated - tip_pos_np)**2, axis=1))
            i = np.argmin(distances)
            if np.isclose(distances[i], 0, atol=self.pos_atol).all():
                j = len(self.tip_poses) - 1 if i == 0 else i - 1
            else:
                return False

            while len(checked_poses) < len(self.tip_poses):
                if len(checked_poses) > 1 and np.allclose(cur_pose_np[0], checked_poses[-1][0], atol=self.pos_atol):
                    repetitions += 1
                checked_poses.append(self.tip_poses[j])
                j = len(self.tip_poses)-1 if j == 0 else j-1

            return repetitions > 1
        else:
            return False

    def _has_tip_moved_at_least(self, n):
        cur_pose_np, _ = self._get_tip_pose_np()
        if cur_pose_np is not None:
            ret = np.sqrt(np.sum((cur_pose_np[0] - self.tip_poses[-1][0])**2))
            for i in reversed(range(0, len(self.tip_poses))):
                if i != 0:
                    ret += np.sqrt(np.sum((self.tip_poses[i][0] - self.tip_poses[i-1][0])**2))
            return ret >= n
        return True

    def get_latest_velocity(self, trajectory, controlled_joints):
        names, position, velocity, _ = trajectory_to_np(trajectory, controlled_joints)
        if len(velocity[:, 0]) * self.sample_period > self.max_trajectory_time_length:
            latest_start = len(velocity[:, 0]) - self.max_trajectory_length
            velocity_latest = velocity[latest_start:, :]
        else:
            velocity_latest = velocity[:, :]
        return velocity_latest

    def is_tip_stuck(self):
        """
        """
        trajectory = self.get_god_map().get_data(identifier.trajectory)
        robot = self.get_robot()
        if trajectory and robot:
            controlled_joints = self.get_robot().controlled_joints
            velocity_latest = self.get_latest_velocity(trajectory, controlled_joints)
            start = 0
            trajectory_len = len(velocity_latest[:, 0])
            while (trajectory_len - start) * self.sample_period >= self.min_trajectory_time_length:
                vel_means = [np.mean(velocity_latest[start:, joint_i]) for joint_i in range(0, len(controlled_joints))]
                zero_means = []
                for joint_i in range(0, len(controlled_joints)):
                    zero_means.append(np.isclose(vel_means[joint_i], 0, atol=self.vel_atol))
                if zero_means and all(zero_means) and self._is_tip_pose_repeated():
                    return True
                start += 1
        return False

    def is_tip_stuck_fft(self):
        """
        """
        trajectory = self.get_god_map().get_data(identifier.trajectory)
        robot = self.get_robot()
        if trajectory and robot:
            controlled_joints = self.get_robot().controlled_joints
            names, position, velocity, _ = trajectory_to_np(trajectory, controlled_joints)
            trajectory_len = len(velocity[:, 0])
            joint_mean_zero = []
            if self.min_trajectory_time_length > trajectory_len * self.sample_period:
                return False
            for joint_i in range(0, len(controlled_joints)):
                joint_vel_moved = np.sum(np.abs(velocity[:, joint_i])) > self.vel_atol
                if joint_vel_moved:
                    fft = np.fft.rfft(velocity[:, joint_i])
                    freqs = np.fft.rfftfreq(len(velocity[:, joint_i]), d=self.sample_period)
                    f = freqs[np.argmax(fft)]
                    if f > 0:
                        period_length = int(((1 / f) / self.sample_period))
                        num_periods = int(trajectory_len / period_length)
                        tmp = []
                        for num_period_i in range(1, num_periods+1):
                            new_start = trajectory_len - num_period_i * period_length
                            mean = np.mean(velocity[new_start:, joint_i])
                            tmp.append(np.isclose(mean, 0, atol=0.001))
                        joint_mean_zero.append(any(tmp))
                    else:
                        start = 0
                        patterns = []
                        while (trajectory_len - start) * self.sample_period >= self.min_trajectory_time_length/5:
                            patterns.append(self.check_pattern(velocity[start:, joint_i].tolist()) != 0)
                            start += 1
                        if any(patterns):
                            return False
            return joint_mean_zero and all(joint_mean_zero)
        return False

    def is_tip_stuck_pattern(self):
        """
        """
        trajectory = self.get_god_map().get_data(identifier.trajectory)
        robot = self.get_robot()
        if trajectory and robot:
            controlled_joints = self.get_robot().controlled_joints
            names, position, velocity, _ = trajectory_to_np(trajectory, controlled_joints)
            trajectory_len = len(velocity[:, 0])
            if self.min_trajectory_time_length > trajectory_len * self.sample_period > self.max_trajectory_time_length:
                return False
            start = 0
            joint_mean_zero = {}
            for joint_i in range(0, len(controlled_joints)):
                joint_vel_moved = np.sum(np.abs(velocity[:, joint_i])) > self.vel_atol
                if joint_vel_moved:
                    while (trajectory_len - start) * self.sample_period >= self.min_trajectory_time_length/10:
                        pattern = self.check_pattern(velocity[start:, joint_i].tolist())
                        if pattern != 0:
                            joint_vel_mean_zero = np.isclose(np.mean(pattern), 0, atol=self.vel_atol).all()
                            if start in joint_mean_zero:
                                joint_mean_zero[start].append(joint_vel_mean_zero)
                            else:
                                joint_mean_zero[start] = [joint_vel_mean_zero]
                        start += 1
            for _, mean_zero in joint_mean_zero.items():
                if mean_zero and all(mean_zero) and self._is_tip_pose_repeated():
                    return True
        return False

    def check_pattern(self, nums):
        p = []
        i = 0
        pattern = True
        while i < len(nums) // 2:
            p.append(nums[i])
            for j in range(0, len(nums) - (len(nums) % len(p)), len(p)):
                if nums[j:j + len(p)] != p:
                    pattern = False
                    break
                else:
                    pattern = True
            # print(nums[-(len(nums) % len(p)):], p[:(len(nums) % len(p))])
            if pattern and nums[-(len(nums) % len(p)) if (len(nums) % len(p)) > 0 else -len(p):] == \
                    p[:(len(nums) % len(p)) if (len(nums) % len(p)) > 0 else len(p)]:
                return p
            i += 1
        return 0

    @profile
    def update(self):

        self._log_tip_pose_np()
        if self.is_tip_stuck():
            raise Exception
        return Status.RUNNING
