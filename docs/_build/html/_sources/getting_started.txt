===============================
Getting Started with GiskardPy
===============================

This page is a brief introduction to the Giskard motion library which explains basic concepts and will give a practical example for building a first Giskard controller.

Installation
============

TODO: I'll add more detailed instructions later. However you'll need `symengine <https://github.com/ARoefer/symengine.git>`_, `symengine.py <https://github.com/ARoefer/symengine.py.git>`_ and `llvm-4.0 <https://apt.llvm.org/>`_ or greater. The first two you can clone from GitHub. I'd suggest you clone those versions and build them from source as they're what I'm using to write this tutorial.

This tutorial is using the `iai_naive_kinematics_sim <https://github.com/suturo16/iai_naive_kinematics_sim>`_ to simulate robots.

What is Giskard?
================

Giskard is a robotic motion library that expresses motions as constraints on changes made to mathematical expressions. By satisfying the constraints and applying the computed changes to the robot, these constraints iteratively describe a trajectory.

Great cliff-notes version, but not really helpful. So let's look at an example.

If you've used other motion libraries before, they probably provided functions that allowed you to input a 6-DoF pose as goal for your robot's endeffector and then it'd generate a trajectory to match that goal.
In Giskard you have to describe the change that you desire to happen to the position and orientation of your robot's endeffector.
A constraint :math:`C` is a triple consisting of a lower limit :math:`lb`, an upper limit :math:`ub` and an expression :math:`x`. :math:`lb` and :math:`ub` limit how much :math:`x` can be changed during one iteration. So if :math:`C = (-1, 1.5, x)` and :math:`x = 2` then the new value for :math:`x` will satisfy :math:`1 \leq x \leq 3.5`.

Let's look at a simple example that moves two points to the same location using one constraint.
Let :math:`E` be the robot's endeffector frame and :math:`\vec{E_p}` its position and let :math:`\vec{G_p}` be the goal location. Then we can define the following constraint :math:`C` that will move :math:`\vec{E_p}` to :math:`\vec{G_p}`.

.. math::

    d &= ||\vec{G_p} - \vec{E_p}|| \\
    C_p &= (-d, -d, d)


The constraint demands that :math:`d` satisfy :math:`d-d \leq d \leq d-d` which is equal to :math:`0 \leq d \leq 0`. As a result of satisfying this constraint, the two points :math:`\vec{E_p}` and :math:`\vec{G_p}` will find themselves at the same location in space.
We can add two more constraints to align the rotation of the endeffector with the rotation of the goal frame. Let :math:`\vec{E_x}, \vec{E_z}, \vec{G_x}, \vec{G_z}` denote the endeffector's and goal's forward and upward pointing axes, then we can define the following constraints:

.. math::

    a_x &= \vec{E_x} \cdot \vec{G_x} \\
    a_z &= \vec{E_z} \cdot \vec{G_z} \\
    C_x &= (1-a_x, 1-a_x, a_x) \\
    C_z &= (1-a_z, 1-a_z, a_z)


Here the constraints :math:`C_x, C_z` drive the dot products of the axes towards the constant value :math:`1`. As a result of satisfying both constraints, the two frames' rotations are going
to be aligned.

Constraint types
````````````````

Giskard uses three types of constraints:

Controllable Constraints
    These constraints constrain the changes that the solver will make to joints in order to satisfy the other constraints. They can be modeled as a four-tuple :math:`(lb, ub, c, x)` where :math:`c` denotes the cost of changing :math:`x`.

Hard constraints
    Hard Constraints are used to enforce hard limits on the generated motion. A typical hard limit are the joint limits of a robot. They are triples of the form that were used in the examples above.

.. NOTE::
    If Giskard can't satisfy all hard constraints, it will view the posed problem as unsolvable and terminate. So be careful about the hard constraints you impose on your controller.

Soft Constraints
    Soft Constraints are used to express the task constraints of a motion which don't have to be satisfied necessarily. The goal of aligning :math:`\vec{E_p}, \vec{G_p}` would usually be expressed as such a constraint, since it might be possible that :math:`\vec{E_p}` can only be moved by :math:`0.1` units at a time because of other constraints. Soft constraints can be modeled as another four-tuple :math:`(lb, ub, w, x)` with :math:`w` being a weight that tells Giskard how important this soft constraint is in comparison to others.

.. NOTE::
    All constraints have to satisfy :math:`lb \leq ub` at all times. Otherwise the posed problem will be seen as unsolvable and Giskard will terminate.


Let's Go to Code
================

Concepts
````````

After this brief introduction to the basic workings of Giskard, let's talk about how to use it in practice. There are two main concepts currently implemented in Giskard:

Robots
    GiskardPy supplies a basic superclass for all robots. It can be found in :code:`giskardpy.robot` and provides functionality for loading a robot from a URDF and generating matching controllable and hard constraints for it.

QPController
    The QPController is a basic controller skeleton. It is the interface to the underlying solver system and keeps track of the controller's variables and constraints. Custom controllers should be derived from this class.


Control Flow
````````````

.. figure:: Giskard_Control_Flow.png
    :alt: Typical Giskard Control Flow
    :align: center

    A very basic controller setup for a ROS node using a Giskard controller.


The figure above depicts the simplest of system setups for running a Giskard controller in a ROS environment. The robot publishes its state on a topic which the controller node subscribes to. Each time it receives a new state, the new state is used to update the values in the controller. Based on the new state, the QPController then computes new commands based on the updated state, which are sent to the robot.
The changes to joint positions computed by the QPController are sent as desired joint velocities to the robot.


A Basic Position Controller
```````````````````````````

In this section we're going to implement a tiny controller node that will passed the path to a urdf file, the name of an endeffector to control and the x, y, z coordinates of a goal position.

We will implement a custom controller class that will move the robot's endeffector to the given goal location. Additionally we are going to implement a second class that interfaces this controller with the ROS system.

The Controller Class
--------------------

To start, let's import a couple of things that we're going to need for this class.

.. code-block:: python
    :caption:   Imports for controller class

    from giskardpy.qpcontroller import QPController
    from giskardpy.qp_problem_builder import SoftConstraint
    from giskardpy.input_system import *
    from giskardpy.symengine_wrappers import *


The two imports :code:`giskardpy.input_system` and :code:`giskardpy.symengine_wrappers` warrant further explanation.
The first imports a variety of generators that automatically generate expressions for variable inputs to a controller.
The second import imports a bunch of wrapper functions that make it easier to use the SymEngine library. Aside from making the use easier, it also overrides a few functions from SymEngine and replaces them with differentiable ones, which is a requirement for the expressions of the constraints in Giskard's controllers.

Let's move on to creating our custom class and have it inherit the functionality of :code:`QPController`. Its constructor will accept two arguments: a robot and the name of the endeffector that it's supposed to control.


.. code-block:: python
    :caption:   Custom Controller Header and Constructor

    class MyPositionController(QPController):
        def __init__(self, robot, eef_name):
            self.eef_name = eef_name
            super(MyPositionController, self).__init__(robot)


The most important thing to note about the constructor is the fact that the superclass' constructor is called at the end of this one, rather than at its beginning. This is done because the super constructor calls all initialization functions.

Now we're going to add an input for a point to our controller. We're going to do so by overriding the :code:`add_inputs()` function. Note that even if your controller has no inputs, you still need to override this function because its default implementation throws an exception.

.. code-block:: python
    :caption:   Override of add_inputs()

    def add_inputs(self, robot):
        self.position_input = Point3Input('goal')

This implementation of :code:`add_inputs()` creates an new input to that represents a point called *goal*. We're going to use this point in our override of the :code:`make_constraints()` function.


.. code-block:: python
    :caption:   Override of the make_constraints() function

    def make_constraints(self, robot):
        super(MyPositionController, self).make_constraints(robot)

        d = norm(self.position_input.get_expression() - pos_of(robot.frames[self.eef_name]))
        self._soft_constraints['position constraint'] = SoftConstraint(-d, -d, 1, d)


In this override it is necessary to call the superclass' implementation, as it automatically adds the controllable constraints and hard constraints defined by the robot. The other two lines just generate the expression from the introduction. Note that :code:`norm()` and :code:`pos_of()` are custom functions provided by the wrapper library. The first computes the magnitude of a vector, the second extracts the translational column from a 4x4 matrix.

The last thing left to do now is to add a function to the controller, that allows us to set the coordinates of our goal. We're going to call it :code:`set_goal()` and are going to pass the x, y, z coordinates as parameters. Each input class has a :code:`get_update_dict()` function that returns a dictionary containing the new values for all the input's internal variables. This dictionary can be passed to the QPController's :code:`update_observables()` function which uses it to update its internal state.

.. code-block:: python
    :caption:   Implementation of set_goal()

    def set_goal(self, x, y, z):
        self.update_observables(self.position_input.get_update_dict(x, y, z))

The completed controller class should now look something like this:

.. code-block:: python
    :caption:   Completed Controller Class

    class MyPositionController(QPController):
        def __init__(self, robot, eef_name):
            self.eef_name = eef_name
            super(MyPositionController, self).__init__(robot)

        def add_inputs(self, robot):
            self.position_input = Point3Input('goal')

        def make_constraints(self, robot):
            super(MyPositionController, self).make_constraints(robot)

            d = norm(self.position_input.get_expression() - pos_of(robot.frames[self.eef_name]))
            self._soft_constraints['position constraint'] = SoftConstraint(-d, -d, 1, d)

        def set_goal(self, x, y, z):
            self.update_observables(self.position_input.get_update_dict(x, y, z))



Quick Intermission...
---------------------

Before we proceed to implementing the node class let's briefly talk about an implementation detail:
GiskardPy uses 4x4 matrices to represent transformations. This means that there is a difference between vectors and points. This also means that some mathematical operations are syntactically possible, but semantically pointless. Just as a reminder, a short list of operations and their results:

- Point  + Point  -> ?(weird scaled thing)
- Point  - Point  -> Vector
- Point  + Vector -> Point
- Point  - Vector -> Point
- Vector + Point  -> Point
- Vector - Point  -> ?(weird mirror-world point)
- Vector - Vector -> Vector
- Vector + Vector -> Vector


The Node Class
--------------

The node class will be the interface between our Giskard controller and the ROS environment. In its constructor, it will load a robot from a given file path and then pass along the loaded robot and a given endeffector name to a new instance of our controller. After it's instantiated, the node will set the controller's goal to a given position. Lastly, the constructor will advertise its commands on a topic - in this example *simulator/commands* - and subscribe to the */joint_states* topic to receive state updates from the robot/simulator.

.. code-block:: python
    :caption: Controller Node's Constructor

    import rospy
    from sensor_msgs.msg import JointState
    from giskardpy.robot import Robot

    class MyPositionControllerNode(object):
        def __init__(self, robot_file, eef_name, x, y, z):
            self.robot = Robot()
            self.robot.load_from_urdf_path(robot_file, 'base_link', [eef_name])
            self.controller = MyPositionController(self.robot, eef_name)
            self.controller.set_goal(x,y,z)
            self.cmd_pub = rospy.Publisher('simulator/commands', JointState, queue_size=1)
            self.cmd_sub = rospy.Subscriber('/joint_states', JointState, self.js_callback, queue_size=1)

The most interesting part of this constructor is the way the robot is loaded. First a blank robot is instantiated, then it is initialized using a urdf file. The loading function needs to be given the name of the root of the kinematic tree and also a list of leaf nodes. It will automatically construct the expressions for all the frames in the tree and will also generate controllable constraints and hard constraints relevant to these frames.
The given codes assumes that the root is always called *base_link*.

Now we need to implement the callback function :code:`js_callback()` referenced by our subscriber. This function will receive a :code:`sensor_msgs/JointState` message, convert it into a dictionary which maps joint names to their respective positions. This dictionary is passed to the robot so it can update its internal state. Then the function will get the next command from the controller. The returned dictionary is then converted into another :code:`sensor_msgs/JointState` message where the dictionary's keys become the joints' names and the dictionary's values become their velocities. This message is finally published using the node's command publisher.

.. code-block:: python
    :caption:   Implementation of Joint State Callback

    def js_callback(self, joint_state):
        js_dict = {joint_state.name[x]: joint_state.position[x] for x in range(len(joint_state.name))}
        self.robot.set_joint_state(js_dict)
        cmd = self.controller.get_next_command()

        cmd_msg = JointState()
        cmd_msg.header.stamp = rospy.Time.now()
        for joint_name, velocity in cmd.items():
            cmd_msg.name.append(joint_name)
            cmd_msg.velocity.append(velocity)
            cmd_msg.position.append(0)
            cmd_msg.effort.append(0)

        self.cmd_pub.publish(cmd_msg)


The complete class should now look something like this:

.. code-block:: python
    :caption:   Complete Node Implementation

    class MyPositionControllerNode(object):
        def __init__(self, robot_file, eef_name, x, y, z):
            self.robot = Robot()
            self.robot.load_from_urdf_path(robot_file, 'base_link', [eef_name])
            self.controller = MyPositionController(self.robot, eef_name)
            self.controller.set_goal(x,y,z)
            self.cmd_pub = rospy.Publisher('simulator/commands', JointState, queue_size=1)
            self.cmd_sub = rospy.Subscriber('/joint_states', JointState, self.js_callback, queue_size=1)

        def js_callback(self, joint_state):
            js_dict = {joint_state.name[x]: joint_state.position[x] for x in range(len(joint_state.name))}
            self.robot.set_joint_state(js_dict)
            cmd = self.controller.get_next_command()

            cmd_msg = JointState()
            cmd_msg.header.stamp = rospy.Time.now()
            for joint_name, velocity in cmd.items():
                cmd_msg.name.append(joint_name)
                cmd_msg.velocity.append(velocity)
                cmd_msg.position.append(0)
                cmd_msg.effort.append(0)

            self.cmd_pub.publish(cmd_msg)

Finishing Up
------------

All that's left to do now, is providing a small main function to that initializes the ROS node, passes the command-line arguments to our node implementation and stops the process from dying prematurely. This would be one possible implementation:

.. code-block:: python
    :caption:   Main Function Implementation

    if __name__ == '__main__':
        if len(sys.argv) < 6:
            print('Please provide: <urdf file> <endeffector name> <goal x> <goal y> <goal z>')
            exit(0)

        rospy.init_node('basic_eef_position_controller')

        node = MyPositionControllerNode(sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))

        rospy.spin()


The final code should look something like this:

.. code-block:: python
    :caption:   Final Code

    #!/usr/bin/env python
    import sys
    import rospy
    from sensor_msgs.msg import JointState

    from giskardpy.robot import Robot
    from giskardpy.qpcontroller import QPController
    from giskardpy.qp_problem_builder import SoftConstraint
    from giskardpy.input_system import *
    from giskardpy.symengine_wrappers import *

    class MyPositionController(QPController):
        def __init__(self, robot, eef_name):
            self.eef_name = eef_name
            super(MyPositionController, self).__init__(robot)

        def add_inputs(self, robot):
            self.position_input = Point3Input('goal')

        def make_constraints(self, robot):
            super(MyPositionController, self).make_constraints(robot)

            d = norm(self.position_input.get_expression() - pos_of(robot.frames[self.eef_name]))
            self._soft_constraints['position constraint'] = SoftConstraint(-d, -d, 1, d)

        def set_goal(self, x, y, z):
            self.update_observables(self.position_input.get_update_dict(x, y, z))


    class MyPositionControllerNode(object):
        def __init__(self, robot_file, eef_name, x, y, z):
            self.robot = Robot()
            self.robot.load_from_urdf_path(robot_file, 'base_link', [eef_name])
            self.controller = MyPositionController(self.robot, eef_name)
            self.controller.set_goal(x,y,z)
            self.cmd_pub = rospy.Publisher('simulator/commands', JointState, queue_size=1)
            self.cmd_sub = rospy.Subscriber('/joint_states', JointState, self.js_callback, queue_size=1)

        def js_callback(self, joint_state):
            js_dict = {joint_state.name[x]: joint_state.position[x] for x in range(len(joint_state.name))}
            self.robot.set_joint_state(js_dict)
            cmd = self.controller.get_next_command()

            cmd_msg = JointState()
            cmd_msg.header.stamp = rospy.Time.now()
            for joint_name, velocity in cmd.items():
                cmd_msg.name.append(joint_name)
                cmd_msg.velocity.append(velocity)
                cmd_msg.position.append(0)
                cmd_msg.effort.append(0)

            self.cmd_pub.publish(cmd_msg)


    if __name__ == '__main__':
        if len(sys.argv) < 6:
            print('Please provide: <urdf file> <endeffector name> <goal x> <goal y> <goal z>')
            exit(0)

        rospy.init_node('basic_eef_position_controller')

        node = MyPositionControllerNode(sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))

        rospy.spin()


Don't forget to mark your file as executable using :code:`chmod +x`.


Run Forrest, Run!
-----------------
Last thing to do now is run the controller. We can use the lightweight *iai_naive_kinematics_sim* to simulate a robot locally.To simulate a *Fetch* robot, you can clone the `fetch_giskard repository <https://github.com/ARoefer/fetch_giskard.git>`_. Its not necessary to build it successfully, but if you do want to you'll need the *robot_controllers* package. You can install it using apt:

.. code-block:: bash

    sudo apt-get install ros-indigo-robot-controllers


Additionally you will need the *fetch_description* package:

.. code-block:: bash

    sudo apt-get install ros-indigo-fetch-description


Once you got all the dependencies and have started a *roscore*, you can launch the simulator using:

.. code-block:: bash

    roslaunch fetch_giskard fetch_sim.launch


Fire up RVIZ and add a new robot model display. You should be seeing something like this:

.. figure:: rviz_sim.png
    :alt: Initial Posture of the Simulated Fetch
    :align:  center

    RVIZ display of initial robot configuration


Now all thats left to do is run your node. Like this for example:

.. code-block:: bash
    :caption:   Example run of the controller

    roscd fetch_giskard
    rosrun your_package your_node.py robots/fetch.urdf gripper_link 0.8 0.3 1



Last Thoughts
=============

Hopefully this tutorial helped you to better understand the Giskard workflow.
As a next step I'd recommend familiarizing yourself with the functions provided by :code:`giskardpy.symengine_wrappers` and :code:`giskardpy.input_system`. As an exercise you could try to implement a the constraints for rotational alignment from the introduction.
When you need more complex functions like *sin* don't forget to use the SymEngine implementation by importing :code:`symengine`.
If you want to import all functions from SymEngine by doing :code:`from symengine import *` don't forget to import the SymEngine wrappers afterwards, as they override some of the functionality.
