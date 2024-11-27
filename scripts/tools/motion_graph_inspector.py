import sys
import rospy
import pydot
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox, QHBoxLayout, QPushButton, QSizePolicy, QLabel
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtCore import Qt, QTimer, QRectF
from giskard_msgs.msg import ExecutionState
from giskardpy_ros.tree.behaviors.plot_motion_graph import execution_state_to_dot_graph
from PyQt5.QtCore import QMutex, QMutexLocker


class MySvgWidget(QSvgWidget):

    def __init__(self, *args):
        QSvgWidget.__init__(self, *args)
        self.mutex = QMutex()  # Mutex for synchronizing access to the widget

    def paintEvent(self, event):
        with QMutexLocker(self.mutex):  # Lock the mutex to prevent concurrent access
            renderer = self.renderer()
            if renderer is not None:
                painter = QPainter(self)
                size = renderer.defaultSize()
                ratio = size.height() / size.width()
                frame_ratio = self.height() / self.width()
                if frame_ratio > ratio:
                    new_width, new_height = self.width(), self.width() * ratio
                else:
                    new_width, new_height = self.height() / ratio, self.height()
                if new_width < self.width():
                    left = (self.width() - new_width) / 2
                else:
                    left = 0
                renderer.render(painter, QRectF(left, 0, new_width, new_height))
                painter.end()


class DotGraphViewer(QWidget):
    last_goal_id: int

    def __init__(self):
        super().__init__()
        self.last_goal_id = -1

        # Initialize the ROS node
        rospy.init_node('motion_graph_viewer', anonymous=True)

        # Set up the GUI components
        self.svg_widget = MySvgWidget(self)
        self.svg_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.svg_widget.setMinimumSize(600, 400)

        self.topic_selector = QComboBox(self)
        self.topic_selector.activated.connect(self.on_topic_selector_clicked)

        # Navigation buttons
        self.first_button = QPushButton('First')
        self.prev_goal_button = QPushButton('Prev Goal')
        self.prev_button = QPushButton('<')
        self.next_button = QPushButton('>')
        self.next_goal_button = QPushButton('Next Goal')
        self.latest_button = QPushButton('Last')

        # Position label
        self.position_label = QLabel(self)

        # Connect navigation buttons to their functions
        self.first_button.clicked.connect(self.show_first_image)
        self.prev_goal_button.clicked.connect(self.show_prev_goal)
        self.prev_button.clicked.connect(self.show_previous_image)
        self.next_button.clicked.connect(self.show_next_image)
        self.next_goal_button.clicked.connect(self.show_next_goal)
        self.latest_button.clicked.connect(self.show_latest_image)

        # Layout for topic selection
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.topic_selector)

        # Layout for navigation buttons and position label
        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.first_button)
        nav_layout.addWidget(self.prev_goal_button)
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.position_label)
        nav_layout.addWidget(self.next_button)
        nav_layout.addWidget(self.next_goal_button)
        nav_layout.addWidget(self.latest_button)

        # Main layout
        layout = QVBoxLayout()
        layout.addLayout(top_layout)
        layout.addWidget(self.svg_widget)
        layout.addLayout(nav_layout)
        self.setLayout(layout)

        self.setWindowTitle('Motion Graph Viewer')
        self.resize(800, 600)

        # Initialize graph history and goal tracking
        self.graphs_by_goal = {}
        self.goals = []
        self.current_goal_index = -1
        self.current_message_index = -1

        # Timer for periodically refreshing topics
        self.topic_refresh_timer = QTimer(self)
        self.topic_refresh_timer.timeout.connect(self.refresh_topics)
        self.topic_refresh_timer.start(1000)  # Refresh every 5 seconds

        # Populate the dropdown with available topics if none is selected
        self.refresh_topics()

    def refresh_topics(self) -> None:
        if self.topic_selector.currentText() == '':
            # Find all topics of type ExecutionState
            topics = rospy.get_published_topics()
            execution_state_topics = [topic for topic, msg_type in topics if msg_type == 'giskard_msgs/ExecutionState']

            self.topic_selector.clear()
            self.topic_selector.addItems(execution_state_topics)
            if len(execution_state_topics) > 0:
                self.on_topic_selected(0)

    def on_topic_selector_clicked(self) -> None:
        # Stop refreshing topics once a topic is selected
        if self.topic_selector.currentIndex() != -1:
            self.topic_refresh_timer.stop()
            self.on_topic_selected(self.topic_selector.currentIndex())

    def on_topic_selected(self, index: int) -> None:
        topic_name = self.topic_selector.currentText()
        if topic_name:
            rospy.Subscriber(topic_name, ExecutionState, self.on_new_message_received, queue_size=50)

    def on_new_message_received(self, msg: ExecutionState) -> None:
        if len(self.goals) > 0:
            navigator_at_end = (self.current_goal_index == self.goals[-1]
                                and self.current_message_index == len(self.graphs_by_goal[self.goals[-1]]) - 1)
        else:
            navigator_at_end = True
        # Extract goal_id and group graphs by goal_id
        if self.last_goal_id == msg.goal_id:
            goal_id = self.goals[-1]
        else:
            self.last_goal_id = msg.goal_id
            goal_id = len(self.goals)
            self.graphs_by_goal[goal_id] = []
            self.goals.append(goal_id)

        graph = execution_state_to_dot_graph(msg, use_state_color=True)

        self.graphs_by_goal[goal_id].append(graph)

        # Update the display to show the latest graph
        if navigator_at_end:
            self.current_goal_index = len(self.goals) - 1
            self.current_message_index = len(self.graphs_by_goal[goal_id]) - 1

        self.update_position_label()

        if navigator_at_end:
            self.display_graph(self.current_goal_index, self.current_message_index, update_position_label=False)

    def display_graph(self, goal_index: int, message_index: int, update_position_label: bool = True) -> None:
        # Display the graph based on goal and message index
        goal_id = self.goals[goal_index]
        graph = self.graphs_by_goal[goal_id][message_index]
        # Update the position label
        if update_position_label:
            self.update_position_label()

        svg_path = 'graph.svg'
        graph.write_svg(svg_path)
        # graph.write_pdf('graph.pdf')
        with QMutexLocker(self.svg_widget.mutex):  # Lock the mutex during SVG loading
            self.svg_widget.load(svg_path)


    def update_position_label(self) -> None:
        goal_count = len(self.goals)
        if goal_count == 0:
            self.position_label.setText('goal 0/0, update 0/0')
            return

        goal_id = self.goals[self.current_goal_index]
        message_count = len(self.graphs_by_goal[goal_id])
        position_text = f'goal {self.current_goal_index + 1}/{goal_count}, update {self.current_message_index + 1}/{message_count}'
        # print(position_text)
        self.position_label.setText(position_text)

    def show_first_image(self) -> None:
        if self.goals:
            self.current_goal_index = 0
            self.current_message_index = 0
            self.display_graph(self.current_goal_index, self.current_message_index)

    def show_previous_image(self) -> None:
        if self.goals:
            if self.current_message_index > 0:
                self.current_message_index -= 1
            else:
                if self.current_goal_index > 0:
                    self.current_goal_index -= 1
                    self.current_message_index = len(self.graphs_by_goal[self.goals[self.current_goal_index]]) - 1
            self.display_graph(self.current_goal_index, self.current_message_index)

    def show_next_image(self) -> None:
        if self.goals:
            if self.current_message_index < len(self.graphs_by_goal[self.goals[self.current_goal_index]]) - 1:
                self.current_message_index += 1
            else:
                if self.current_goal_index < len(self.goals) - 1:
                    self.current_goal_index += 1
                    self.current_message_index = 0
            self.display_graph(self.current_goal_index, self.current_message_index)

    def show_prev_goal(self) -> None:
        if self.goals and self.current_goal_index > 0:
            self.current_goal_index -= 1
            self.current_message_index = 0
            self.display_graph(self.current_goal_index, self.current_message_index)

    def show_next_goal(self) -> None:
        if self.goals and self.current_goal_index < len(self.goals) - 1:
            self.current_goal_index += 1
            self.current_message_index = 0
            self.display_graph(self.current_goal_index, self.current_message_index)

    def show_latest_image(self) -> None:
        if self.goals:
            self.current_goal_index = len(self.goals) - 1
            self.current_message_index = len(self.graphs_by_goal[self.goals[self.current_goal_index]]) - 1
            self.display_graph(self.current_goal_index, self.current_message_index)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = DotGraphViewer()
    viewer.show()
    sys.exit(app.exec_())
