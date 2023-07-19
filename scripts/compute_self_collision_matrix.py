import traceback
import typing
from typing import Set, Tuple, List, Optional, Dict
import rospy
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox, QWidget, \
    QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QFileDialog, QMessageBox, QProgressBar, QLabel
from PyQt5.QtCore import Qt
import xml.etree.ElementTree as ET
import pandas as pd
from collections import defaultdict
import sys
import os

from std_msgs.msg import ColorRGBA

from giskardpy import identifier
from giskardpy.model.better_pybullet_syncer import BetterPyBulletSyncer
from giskardpy.model.collision_world_syncer import DisableCollisionReason
from giskardpy.model.ros_msg_visualization import ROSMsgVisualization
from giskardpy.model.utils import robot_name_from_urdf_string
from giskardpy.model.world import WorldTree
from giskardpy.my_types import PrefixName


def extract_link_names(urdf_file):
    # Parse the URDF file
    tree = ET.parse(urdf_file)

    # Get root of the XML document
    root = tree.getroot()

    # List to store the names of all links
    link_names = []

    # Iterate over 'link' entries
    for link in root.findall('link'):
        # Get the name of the link
        link_name = link.get('name')

        # Append the name to the list
        link_names.append(link_name)

    return link_names


def extract_collision_data(black_list: Set[Tuple[str, str]], link_names: List[str]):
    # This allows us to assign values to keys that don't yet exist in the dictionary
    data = defaultdict(lambda: defaultdict(lambda: False))

    # Iterate over 'disable_collisions' entries
    for link1, link2 in black_list:
        link1 = link1
        link2 = link2
        # Assign True to the corresponding entries in the dictionary
        data[link1][link2] = True
        data[link2][link1] = True  # Since collision disabling is bidirectional

    # Convert the nested dictionary to a DataFrame
    df = pd.DataFrame(data, index=link_names, columns=link_names)

    # Replace NaN values (which mean that there was no 'disable_collisions' entry for a pair of links) with False
    df.fillna(False, inplace=True)

    # Sort rows and columns alphabetically
    df.sort_index(axis=0, inplace=True)
    df.sort_index(axis=1, inplace=True)

    return df


reason_color_map = {
    DisableCollisionReason.Never: (163, 177, 233),  # blue
    DisableCollisionReason.Adjacent: (233, 163, 163),  # red
    DisableCollisionReason.AlmostAlways: (233, 163, 231),  # purple
    DisableCollisionReason.Default: (233, 231, 163),  # yellow
    DisableCollisionReason.Unknown: (166, 166, 166),  # grey
    None: (255, 255, 255),  # white
}


class ReasonCheckBox(QCheckBox):
    reason: Optional[DisableCollisionReason]
    x: int
    y: int
    table: QTableWidget

    def __init__(self, table: QTableWidget, x: int, y: int) -> None:
        super().__init__()
        self.reason = None
        self.x = x
        self.y = y
        self.table = table

    def connect_callback(self):
        self.stateChanged.connect(self.checkbox_callback)

    def set_reason(self, reason: Optional[DisableCollisionReason]):
        self.setChecked(reason is not None)
        self.reason = reason
        self.setStyleSheet(f"background-color: rgb{reason_color_map[reason]};")

    def checkbox_callback(self, state, copy_to_twin: bool = True):
        if state == Qt.Checked:
            self.set_reason(DisableCollisionReason.Unknown)
        else:
            self.set_reason(None)
        if copy_to_twin:
            other_checkbox = self.table.cellWidget(self.y, self.x).layout().itemAt(0).widget()
            other_checkbox.checkbox_callback(state, False)


class Table(QMainWindow):
    robot_description = 'robot_description'

    _reasons: Dict[Tuple[PrefixName, PrefixName], DisableCollisionReason]

    def __init__(self):
        super().__init__()
        self.world = WorldTree.empty_world()
        self.world.default_link_color = ColorRGBA(0.5, 0.5, 0.5, 1)
        self.collision_scene = BetterPyBulletSyncer.empty(self.world)
        self.ros_visualizer = ROSMsgVisualization('map')
        self.df = pd.DataFrame()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Link Collisions")
        self.setMinimumSize(800, 600)

        self.progress = QProgressBar()

        # Create QLineEdit for the URDF file path
        self.urdf_file_path_input = QLineEdit()

        # Create Browse button for URDF file
        self.urdf_browse_button = QPushButton("...")
        self.urdf_browse_button.clicked.connect(self.urdf_browse)

        # Create QLineEdit for the SRDF file path
        self.srdf_file_path_input = QLineEdit()

        # Create Browse button for SRDF file
        self.srdf_browse_button = QPushButton("...")
        self.srdf_browse_button.clicked.connect(self.srdf_browse)

        # Create Load and Save buttons
        self.load_urdf_file_button = QPushButton("Load urdf from file")
        self.load_urdf_file_button.clicked.connect(self.load_urdf_from_path)
        self.load_urdf_param_button = QPushButton("Load urdf from parameter server")
        self.load_urdf_param_button.clicked.connect(self.load_urdf_from_paramserver)

        self.load_srdf_button = QPushButton("Load srdf")
        self.load_srdf_button.clicked.connect(self.load)
        self.compute_srdf_button = QPushButton("Compute self collision matrix")
        self.compute_srdf_button.clicked.connect(self.compute_self_collision_matrix)
        self.save_srdf_button = QPushButton("Save srdf")
        self.save_srdf_button.clicked.connect(self.save_srdf)

        # Create horizontal box layouts for the QLineEdits and Browse buttons
        urdf_text = QHBoxLayout()
        urdf_text.addWidget(self.urdf_file_path_input)
        urdf_text.addWidget(self.urdf_browse_button)

        urdf_bottoms = QHBoxLayout()
        urdf_bottoms.addWidget(self.load_urdf_file_button)
        urdf_bottoms.addWidget(self.load_urdf_param_button)

        srdf_text = QHBoxLayout()
        srdf_text.addWidget(self.srdf_file_path_input)
        srdf_text.addWidget(self.srdf_browse_button)

        srdf_bottoms = QHBoxLayout()
        srdf_bottoms.addWidget(self.load_srdf_button)
        srdf_bottoms.addWidget(self.compute_srdf_button)
        srdf_bottoms.addWidget(self.save_srdf_button)

        self.table = QTableWidget()
        self.table.cellClicked.connect(self.table_item_callback)

        # Create the layout
        layout = QVBoxLayout()

        # Add horizontal box layouts and Save button to the layout
        layout.addLayout(urdf_text)
        layout.addLayout(urdf_bottoms)
        layout.addLayout(srdf_text)
        layout.addLayout(srdf_bottoms)
        layout.addWidget(self.progress)
        layout.addLayout(self._legend_box_layout())
        layout.addWidget(self.table)

        # Set layout
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.set_progress(0, 'no urdf loaded')

    def _legend_box_layout(self) -> QHBoxLayout:
        legend = QHBoxLayout()

        for reason, color in reason_color_map.items():
            if reason is not None:
                label = QLabel(reason.name)
            else:
                label = QLabel('check collision')
            label.setStyleSheet(f"background-color: rgb{color}; color: black;")
            if reason == DisableCollisionReason.Never:
                label.setToolTip("These links are never in contact.")
            elif reason == DisableCollisionReason.Unknown:
                label.setToolTip("This link pair was disabled for an unknown reason.")
            elif reason == DisableCollisionReason.Adjacent:
                label.setToolTip("This link pair is only connected by joints that cannot move.")
            elif reason == DisableCollisionReason.Default:
                label.setToolTip("This link pair is in collision in the robot's default state.")
            elif reason == DisableCollisionReason.AlmostAlways:
                label.setToolTip("This link pair is almost always in collision.")
            else:
                label.setToolTip("Collisions will be computed.")
            legend.addWidget(label)
        return legend

    def set_progress(self, value: int, text: Optional[str] = None):
        value = min(max(value, 0), 100)
        self.progress.setValue(value)
        if text is not None:
            self.progress.setFormat(f'{text}: %p%')

    def urdf_browse(self):
        # Open a file dialog and get the selected file path
        file_path, _ = QFileDialog.getOpenFileName()

        # If a file path was selected, update the QLineEdit
        if file_path:
            self.urdf_file_path_input.setText(file_path)

    def compute_self_collision_matrix(self):
        self._reasons = self.collision_scene.compute_self_collision_matrix(self.group_name,
                                                                           save_to_tmp=False,
                                                                           non_controlled=True,
                                                                           progress_callback=self.set_progress)
        self.update_table(self.str_reasons)
        self.set_progress(100, 'done checking collisions')

    @property
    def str_reasons(self):
        return self.prefix_reasons_to_str_reasons(self._reasons)

    @property
    def reasons(self):
        return self._reasons

    def srdf_browse(self):
        # Open a file dialog and get the selected file path
        file_path, _ = QFileDialog.getOpenFileName()

        # If a file path was selected, update the QLineEdit
        if file_path:
            self.srdf_file_path_input.setText(file_path)

    def load_urdf_from_paramserver(self):
        if rospy.has_param(self.robot_description):
            urdf = rospy.get_param(self.robot_description)
            self.load_urdf(urdf)
            self.urdf_file_path_input.setText(f'<loaded \"{self.group_name}\" from \"{self.robot_description}\">')

    def load_urdf(self, urdf):
        self.world._clear()
        self.set_progress(0, 'loading urdf from parameter server')
        group_name = robot_name_from_urdf_string(urdf)
        self.set_progress(10, 'parsing urdf')
        self.world.add_urdf(urdf, group_name)
        self._reasons = {}
        self.set_progress(50, 'updating table')
        self.update_table(self.reasons)
        self.set_tmp_srdf_path()
        self.world.god_map.set_data(identifier.controlled_joints, self.world.movable_joint_names)
        self.set_progress(100, 'done loading urdf')

    def set_tmp_srdf_path(self):
        if len(self.world.group_names) > 0 and self.srdf_file_path_input.text() == '':
            self.srdf_file_path_input.setText(self.collision_scene.get_path_to_self_collision_matrix(self.group_name))

    def load(self):
        srdf_file = self.srdf_file_path_input.text()

        try:

            # Extract collision data from SRDF file
            if os.path.isfile(srdf_file):
                self._reasons = self.collision_scene.load_black_list_from_srdf(srdf_file, self.group_name, False)
            # Update the table
            self.update_table(self.str_reasons)
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error", str(e))

    def prefix_reasons_to_str_reasons(self, reasons: Dict[Tuple[PrefixName, PrefixName], DisableCollisionReason]) \
            -> Dict[Tuple[str, str], DisableCollisionReason]:
        return {(x[0].short_name, x[1].short_name): reason for x, reason in reasons.items()}

    def load_urdf_from_path(self):
        urdf_file = self.urdf_file_path_input.text()

        if not os.path.isfile(urdf_file):
            QMessageBox.critical(self, "Error", f"File does not exist: \n{urdf_file}")
            return

        with open(urdf_file, 'r') as f:
            self.load_urdf(f.read())

    @property
    def link_names(self) -> List[str]:
        return list(sorted(x.short_name for x in self.world.link_names_with_collisions))

    @property
    def group_name(self):
        return list(self.world.group_names)[0]

    def table_item_callback(self, row, column):
        self.ros_visualizer.clear_marker()
        self.collision_scene.sync()
        for link_name in self.world.link_names_with_collisions:
            self.world.links[link_name].dye_collisions(self.world.default_link_color)
        link1 = self.world.search_for_link_name(self.link_names[row])
        link2 = self.world.search_for_link_name(self.link_names[column])
        key = self.world.sort_links(link1, link2)
        reason = self.reasons.get(key, None)
        color = reason_color_map[reason]
        color_msg = ColorRGBA(color[0]/255, color[1]/255, color[2]/255, 1)
        self.world.links[link1].dye_collisions(color_msg)
        self.world.links[link2].dye_collisions(color_msg)
        self.world.reset_cache()
        self.ros_visualizer.publish_markers()

    def update_table(self, reasons):
        # Reset the table
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)

        # Row count
        self.table.setRowCount(len(self.link_names))

        # Column count
        self.table.setColumnCount(len(self.link_names))

        # Table Headers
        self.table.setHorizontalHeaderLabels(self.link_names)
        self.table.setVerticalHeaderLabels(self.link_names)

        # Populate the table with checkboxes
        for x, link1 in enumerate(self.link_names):
            for y, link2 in enumerate(self.link_names):
                key = (link1, link2)
                r_key = (link2, link1)
                if key in reasons:
                    reason = reasons[key]
                elif r_key in reasons:
                    reason = reasons[r_key]
                else:
                    reason = None
                self.add_table_item(x, y, reason)

        num_rows = self.table.rowCount()

        widths = []

        for x in range(num_rows):
            item = self.table.item(x, 0)
            if item is not None:
                widths.append(item.sizeHint().width())
        if widths:
            self.table.setColumnWidth(0, max(widths))

    def add_table_item(self, x, y, reason):
        checkbox = ReasonCheckBox(self.table, x, y)
        checkbox.set_reason(reason)
        checkbox.connect_callback()
        if x == y:
            checkbox.setDisabled(True)

        widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(checkbox)
        layout.setAlignment(checkbox, Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        widget.setLayout(layout)
        self.table.setCellWidget(x, y, widget)

    def save_srdf(self):
        new_blacklist = set()
        reasons = {}
        for i, link1 in enumerate(self.link_names):
            for j, link2 in enumerate(self.link_names):
                link1 = self.world.search_for_link_name(link1)
                link2 = self.world.search_for_link_name(link2)
                key = tuple(sorted((link1, link2)))
                checkbox = self.table.cellWidget(i, j)
                if checkbox.isChecked():
                    reason = checkbox.reason
                    reasons[key] = reason
                    new_blacklist.add(key)
        self.collision_scene.save_black_list(self.world.groups[self.group_name],
                                             new_blacklist,
                                             reasons,
                                             file_name=self.srdf_file_path_input.text())
        self.set_progress(100, f'Saved {self.srdf_file_path_input.text()}')


def main():
    # Display DataFrame in PyQt5 GUI
    app = QApplication(sys.argv)
    window = Table()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    rospy.init_node('self_collision_matrix_updater')
    main()

# TODO:
#   - click checkbox to show in rviz
