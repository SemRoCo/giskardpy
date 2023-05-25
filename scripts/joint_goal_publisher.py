#!/usr/bin/env python

from __future__ import division

import sys

from defusedxml import minidom

import rospy
import random
from tkinter import *   ## notice lowercase 't' in tkinter here

from giskardpy.python_interface import GiskardWrapper
# import xml.dom.minidom
from sensor_msgs.msg import JointState
from math import pi
from control_msgs.msg import JointTrajectoryControllerState
from giskardpy.utils import logging


def get_param(name, value=None):
    private = "{}".format(name)
    if rospy.has_param(private):
        return rospy.get_param(private)
    elif rospy.has_param(name):
        return rospy.get_param(name)
    else:
        return value


class JointGoalPublisher(object):
    def init_collada(self, robot):
        """
        reads the controllable joints from a collada
        :param robot:
        """
        robot = robot.getElementsByTagName('kinematics_model')[0].getElementsByTagName('technique_common')[0]
        for child in robot.childNodes:
            if child.nodeType is child.TEXT_NODE:
                continue
            if child.localName == 'joint':
                name = child.getAttribute('name')
                if name not in self.giskard_joints:
                    continue

                if child.getElementsByTagName('revolute'):
                    joint = child.getElementsByTagName('revolute')[0]
                else:
                    logging.logwarn("Unknown joint type %s", child)
                    continue

                if joint:
                    limit = joint.getElementsByTagName('limits')[0]
                    minval = float(limit.getElementsByTagName('min')[0].childNodes[0].nodeValue)
                    maxval = float(limit.getElementsByTagName('max')[0].childNodes[0].nodeValue)
                    if minval == maxval:  # this is fixed joint
                        continue

                    self.joint_list.append(name)
                    joint = {'min':minval*pi/180.0, 'max':maxval*pi/180.0, 'zero':0, 'position':0, 'velocity':0, 'effort':0}
                    self.free_joints[name] = joint

    def init_urdf(self, robot):
        """
        reads the controllable joints from a urdfs
        :param robot:
        """
        robot = robot.getElementsByTagName('robot')[0]
        # Find all non-fixed joints that are controlled by giskard
        for child in robot.childNodes:
            if child.nodeType is child.TEXT_NODE:
                continue
            if child.localName == 'joint':
                jtype = child.getAttribute('type')
                if jtype in ['fixed', 'floating', 'planar']:
                    continue
                name = child.getAttribute('name')
                if name not in self.giskard_joints:
                    continue

                self.joint_list.append(name)
                if jtype == 'continuous':
                    minval = -pi
                    maxval = pi
                else:
                    try:
                        limit = child.getElementsByTagName('limit')[0]
                        minval = float(limit.getAttribute('lower'))
                        maxval = float(limit.getAttribute('upper'))
                    except:
                        logging.logwarn("%s is not fixed, nor continuous, but limits are not specified!" % name)
                        continue

                safety_tags = child.getElementsByTagName('safety_controller')
                if self.use_small and len(safety_tags) == 1:
                    tag = safety_tags[0]
                    if tag.hasAttribute('soft_lower_limit'):
                        minval = max(minval, float(tag.getAttribute('soft_lower_limit')))
                    if tag.hasAttribute('soft_upper_limit'):
                        maxval = min(maxval, float(tag.getAttribute('soft_upper_limit')))

                mimic_tags = child.getElementsByTagName('mimic')
                if self.use_mimic and len(mimic_tags) == 1:
                    tag = mimic_tags[0]
                    entry = {'parent': tag.getAttribute('joint')}
                    if tag.hasAttribute('multiplier'):
                        entry['factor'] = float(tag.getAttribute('multiplier'))
                    if tag.hasAttribute('offset'):
                        entry['offset'] = float(tag.getAttribute('offset'))

                    self.dependent_joints[name] = entry
                    continue

                if name in self.dependent_joints:
                    continue

                if self.zeros and name in self.zeros:
                    zeroval = self.zeros[name]
                elif minval > 0 or maxval < 0:
                    zeroval = (maxval + minval)/2
                else:
                    zeroval = 0

                joint = {'min': minval, 'max': maxval, 'zero': zeroval}
                #if self.pub_def_positions:
                  #  joint['position'] = zeroval
                #if self.pub_def_vels:
                 #   joint['velocity'] = 0.0
                #if self.pub_def_efforts:
                 #   joint['effort'] = 0.0

                if jtype == 'continuous':
                    joint['continuous'] = True
                self.free_joints[name] = joint

    def send_goal(self, goal):
        """
        sends a joint goal to giskard
        :param goal:
        :type goal: dict
        """
        self.giskard_wrapper.set_joint_goal(goal)
        self.giskard_wrapper.plan_and_execute(False)

    def cancel_goal(self):
        self.giskard_wrapper.cancel_all_goals()


    def __init__(self):
        description = get_param('robot_description')

        self.giskard_wrapper = GiskardWrapper()

        self.free_joints = {}
        self.joint_list = [] # for maintaining the original order of the joints
        self.dependent_joints = get_param("dependent_joints", {})
        self.use_mimic = get_param('~use_mimic_tags', True)
        self.use_small = get_param('~use_smallest_joint_limits', True)

        self.zeros = get_param("zeros")

        #self.pub_def_positions = get_param("publish_default_positions", True)
        #self.pub_def_vels = get_param("publish_default_velocities", False)
        #self.pub_def_efforts = get_param("publish_default_efforts", False)

        self.giskard_joints = self.giskard_wrapper.get_controlled_joints(self.giskard_wrapper.robot_name)

        robot = minidom.parseString(description)
        if robot.getElementsByTagName('COLLADA'):
            self.init_collada(robot)
        else:
            self.init_urdf(robot)



class JointGoalPublisherGui(Frame):

    def __init__(self, jgp, master=None):
        """
        :param jgp: The JointGoalPublisher that this gui will represent
        :type jgp: JointGoalPublisher
        :param master:
        :type master: Tk
        """
        Frame.__init__(self, master)
        self.master = master
        self.jgp = jgp
        self.joint_map = {}
        self.collision_distance = get_param('~collision_distance', 0.1)
        self.slider_resolution = get_param('~slider_resolution', 0.01)
        self.allow_self_collision = IntVar(value=1)

        self.master.title("Giskard Joint Goal Publisher")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)

        self.slider_frame = self.VerticalScrolledFrame(self)
        self.slider_frame.grid(row=0, stick="ns")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.sliders = {}
        r = 0
        for name in self.jgp.joint_list:
            if name not in self.jgp.free_joints:
                continue
            joint = self.jgp.free_joints[name]

            if joint['min'] == joint['max']:
                continue

            l = Label(self.slider_frame.interior, text=name)
            l.grid(column=0, row=r)
            slider = Scale(self.slider_frame.interior, from_=joint['min'], to=joint['max'], orient=HORIZONTAL, resolution=self.slider_resolution)
            slider.set(joint['zero'])
            slider.grid(column=1, row=r)
            slider.bind('<Enter>', lambda event, bound_slider=slider: self._slider_bound_to_mousewheel(event, bound_slider))
            slider.bind('<Leave>', lambda event, bound_slider=slider: self._slider_unbound_to_mousewheel(event, bound_slider))
            self.sliders[name] = slider
            r += 1

        self.current_joint_states()
        buttonFrame=Frame(self)
        sendGoalButton = Button(buttonFrame, text="send goal", command=self.send_goal)
        randomizeButton = Button(buttonFrame, text="randomize", command=self.randomize)
        resetButton = Button(buttonFrame, text="default Js", command=self.reset_sliders)
        currentJsButton = Button(buttonFrame, text="current Js", command=self.current_joint_states)
        cancelGoalButton = Button(buttonFrame, text="cancel", command=self.cancel_goal)

        selfCollisionButton = Checkbutton(buttonFrame, text="allow self collision", variable=self.allow_self_collision, onvalue=1, offvalue=0)

        sendGoalButton.grid(row=1)
        randomizeButton.grid(row=1, column=1)
        resetButton.grid(row=1, column=2)
        currentJsButton.grid(row=1, column=3)
        cancelGoalButton.grid(row=1, column=4)
        selfCollisionButton.grid(row=1, column=5)

        buttonFrame.grid(row=1)

    def _slider_scroll_handler(self, event, slider):
        if event.num == 5 or event.delta == -120:
            slider.set(slider.get() + self.slider_resolution)
        if event.num == 4 or event.delta == 120:
            slider.set(slider.get() - self.slider_resolution)

    def _slider_bound_to_mousewheel(self, event, slider):
        slider.bind_all("<Button-4>", lambda event, s=slider: self._slider_scroll_handler(event, s))
        slider.bind_all("<Button-5>", lambda event, s=slider: self._slider_scroll_handler(event, s))

    def _slider_unbound_to_mousewheel(self, event, slider):
        slider.unbind_all("<Button-4>")
        slider.unbind_all("<Button-5>")

    def send_goal(self):
        """
        sends a joint goal with the joint states set in the sliders
        """
        goal_dict = {}
        for key, value in self.sliders.items():
            goal_dict[key] = value.get()

        if self.allow_self_collision.get():
            jgp.giskard_wrapper.allow_self_collision()
        else:
            jgp.giskard_wrapper.avoid_collision(self.collision_distance, body_b=jgp.giskard_wrapper.get_robot_name())

        self.jgp.send_goal(goal_dict)

    def randomize(self):
        """
        sets every slider to a random value
        """
        for key in self.sliders.keys():
            val = random.uniform(self.jgp.free_joints[key]['min'], self.jgp.free_joints[key]['max'])
            self.sliders[key].set(val)

    def reset_sliders(self):
        """
        sets the value of every slider to its zerovalue
        """
        for key in self.sliders:
            self.sliders[key].set(self.jgp.free_joints[key]['zero'])

    def current_joint_states(self):
        """
        sets the value of every slider to its corresponding current joint state
        """
        msg = rospy.wait_for_message('joint_states', JointState)
        for i in range(len(msg.name)):
            if msg.name[i] in self.sliders:
                self.sliders[msg.name[i]].set(msg.position[i])

    def cancel_goal(self):
        """
        cancels the current goal
        """
        self.jgp.cancel_goal()


    class VerticalScrolledFrame(Frame):
        """A pure Tkinter scrollable frame that actually works!
        * Use the 'interior' attribute to place widgets inside the scrollable frame
        * Construct and pack/place/grid normally
        * This frame only allows vertical scrolling
        source: https://stackoverflow.com/questions/16188420/python-tkinter-scrollbar-for-frame
        """

        def __init__(self, parent, *args, **kw):
            Frame.__init__(self, parent, *args, **kw)

            # create a canvas object and a vertical scrollbar for scrolling it
            vscrollbar = Scrollbar(self, orient=VERTICAL)
            vscrollbar.pack(fill=Y, side=RIGHT, expand=FALSE)
            self.canvas = Canvas(self, bd=0, highlightthickness=0,
                            yscrollcommand=vscrollbar.set)
            self.canvas.pack(side=LEFT, fill=BOTH, expand=TRUE)
            vscrollbar.config(command=self.canvas.yview)

            # reset the view
            self.canvas.xview_moveto(0)
            self.canvas.yview_moveto(0)

            # create a frame inside the canvas which will be scrolled with it
            self.interior = interior = Frame(self.canvas)
            interior_id = self.canvas.create_window(0, 0, window=interior, anchor=NW)

            self.bind('<Enter>', self._bound_to_mousewheel)
            self.bind('<Leave>', self._unbound_to_mousewheel)


            # track changes to the canvas and frame width and sync them,
            # also updating the scrollbar
            def _configure_interior(event):
                # update the scrollbars to match the size of the inner frame
                size = (interior.winfo_reqwidth(), interior.winfo_reqheight())
                self.canvas.config(scrollregion="0 0 %s %s" % size)
                if interior.winfo_reqwidth() != self.canvas.winfo_width():
                    # update the canvas's width to fit the inner frame
                    self.canvas.config(width=interior.winfo_reqwidth())

            interior.bind('<Configure>', _configure_interior)

            def _configure_canvas(event):
                if interior.winfo_reqwidth() != self.canvas.winfo_width():
                    # update the inner frame's width to fill the canvas
                    self.canvas.itemconfigure(interior_id, width=self.canvas.winfo_width())

                self.canvas.bind('<Configure>', _configure_canvas)

        def _scroll_handler(self, event):
            if event.num == 5 or event.delta == -120:
                self.canvas.yview_scroll(1, "units")
            if event.num == 4 or event.delta == 120:
                self.canvas.yview_scroll(-1, "units")

        def _bound_to_mousewheel(self, event):
            self.canvas.bind_all("<Button-4>", self._scroll_handler)
            self.canvas.bind_all("<Button-5>", self._scroll_handler)

        def _unbound_to_mousewheel(self, event):
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")


if __name__ == '__main__':
    try:
        rospy.init_node('joint_goal_publisher')
        jgp = JointGoalPublisher()

        root = Tk()
        root.geometry("590x600")
        gui = JointGoalPublisherGui(jgp, root)

        root.mainloop()

    except rospy.ROSInterruptException:
        pass
