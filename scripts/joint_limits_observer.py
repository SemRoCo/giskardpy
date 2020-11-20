#!/usr/bin/env python

import rospy
import sys
from giskardpy import urdf_object
from sensor_msgs.msg import JointState
import math
import curses
from signal import signal, SIGINT
from threading import Lock
import time

# Visualizes the given joint states w.r.t. their limits
# rosrun giskardpy joint_limit_observer <joint_state_topic>

warning_threshold = 0.1
critical_threshold = 0.02
scale_steps = 40


js = None
js_lock = Lock()

def handler(sig, stackframe):
    sys.exit(0)

def js_cb(msg):
    with js_lock:
        global js
        js = msg

def main(stdscr):
    signal(SIGINT, handler)
    stdscr.nodelay(True)
    curses.init_pair(1, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
    rospy.init_node(u'joint_limit_observer')
    while not rospy.has_param(u'/robot_description'):
        stdscr.addstr(0, 0, "waiting for parameter /robot_description")
        stdscr.refresh()
        cmd = stdscr.getch()
        while (cmd != curses.ERR):
            if cmd == 113: # q pressed
                sys.exit()
            cmd = stdscr.getch()
    robot_desc = rospy.get_param(u'robot_description')
    js_topic = u'/joint_states' if len(sys.argv) == 1 else sys.argv[1]
    robot = urdf_object.URDFObject(robot_desc)
    joint_limits = robot.get_all_joint_limits()
    num_joints = len(joint_limits)
    rospy.Subscriber(js_topic, JointState, js_cb)
    pad = curses.newpad(num_joints, 200)
    pad_pos = 0
    pad.nodelay(True)

    while True:
        with js_lock:
            js_tmp = js
        cmd = stdscr.getch()
        while (cmd != curses.ERR):
            if cmd == curses.KEY_DOWN:
                pad_pos = min(num_joints - y, pad_pos + 1)
            elif cmd == curses.KEY_UP:
                pad_pos = max(0, pad_pos - 1)
            elif cmd == 113:  # q pressed
                sys.exit(0)
            cmd = stdscr.getch()
        if js_tmp is None:
            pad.clear()
            pad.addstr(0, 0, "no js update received yet")
            y, x = stdscr.getmaxyx()
            pad.refresh(0, 0, 0, 0, y - 1, x - 1)
            time.sleep(0.1)
            continue
        output = []
        output_color = []
        for joint_name in js_tmp.name:
            joint_pos = js_tmp.position[js_tmp.name.index(joint_name)]
            lower, upper = joint_limits[joint_name]
            if lower is None or upper is None:
                lower = -math.pi
                upper = math.pi
                joint_pos_without_offset = joint_pos - 2 * math.pi * math.copysign((abs(joint_pos) + math.pi) // (2 * math.pi), joint_pos)
                js_range = upper - lower
                relative = int(round(scale_steps / js_range * (joint_pos_without_offset - lower)))
                s = u'-' * (scale_steps + 1)
                s_form = u'{}{}{}'.format(s[:relative], u'0', s[relative + 1:])
                output.append(u'     inf [{}]      inf   {}'.format(s_form, joint_name))
                output_color.append(curses.color_pair(0))
            else:
                js_range = upper - lower
                relative = int(round(scale_steps / js_range * (joint_pos - lower)))
                output_color.append(curses.color_pair(0))
                limit_distance = min(abs(joint_pos - lower), abs(joint_pos - upper))
                if limit_distance < js_range * critical_threshold:
                    output_color[-1] = curses.color_pair(2)
                elif limit_distance < js_range * warning_threshold:
                    output_color[-1] = curses.color_pair(1)
                s = u'-' * (scale_steps + 1)
                s_form = u'{}{}{}'.format(s[:relative], u'0', s[relative + 1:])
                output.append(u'{:8.2f} [{}] {:8.2f}   {}'.format(lower, s_form, upper, joint_name))

        for i in range(num_joints):
            if(curses.has_colors()):
                pad.addstr(i, 0, output[i], output_color[i])
            else:
                pad.addstr(i, 0, output[i])

        y, x = stdscr.getmaxyx()
        pad.refresh(pad_pos, 0, 0, 0, y - 1, x - 1)
        time.sleep(0.1)




if __name__ == '__main__':
    curses.wrapper(main)
