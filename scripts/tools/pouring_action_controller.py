#!/usr/bin/env python
import rospy
import threading
import signal
import os
import sys
import logging
from pynput import keyboard
from std_msgs.msg import String

# Global variable to store the currently pressed keys
pressed_keys = set()


# Function to update the set of pressed keys
def on_key_press(key):
    try:
        pressed_keys.add(key.char)
    except AttributeError:
        # Key is not a character key (e.g., special keys), so we ignore it
        pass


# Function to update the set of pressed keys when a key is released
def on_key_release(key):
    try:
        pressed_keys.remove(key.char)
    except:
        # Key was not in the set (e.g., special keys), so we ignore it
        pass


# Start listening for keyboard events
listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
listener.start()

# Initialize ROS node and publisher
rospy.init_node('pouringActionController')
key_pub = rospy.Publisher('pouringActions', String, queue_size=1)

print('Commands')
print({'w': 'forward',
       's': 'backward',
       'a': 'left',
       'd': 'right',
       'u': 'up',
       'j': 'down',
       'y': 'move_to',
       'g': 'tilt',
       'h': 'tilt_back',
       'q': 'keep_upright'})


# Function to publish the concatenated pressed keys
def publish_pressed_keys():
    global pressed_keys
    if pressed_keys:
        pressed_keys_str = ";".join(pressed_keys)
        key_pub.publish(pressed_keys_str)

    # Set the timer to call itself after a delay
    timer = threading.Timer(1.0 / 200.0, publish_pressed_keys)  # 50 Hz frequency
    timer.daemon = True
    timer.start()


# Start publishing timer
publish_pressed_keys()


# Define the handler for Ctrl+C
def signal_handler(sig, frame):
    print("Exiting...")
    rospy.signal_shutdown("Ctrl+C signal received")


# Attach the Ctrl+C handler
signal.signal(signal.SIGINT, signal_handler)

# Run ROS loop
rospy.spin()
