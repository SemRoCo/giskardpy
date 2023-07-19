from __future__ import annotations
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
    None: (255, 255, 255),
}


class ReasonCheckBox(QCheckBox):
    row: int
    column: int
    table: Table

    def __init__(self, table: Table, row: int, column: int) -> None:
        super().__init__()
        self.reason = None
        self.row = row
        self.column = column
        self.table = table

    def connect_callback(self):
        self.stateChanged.connect(self.checkbox_callback)

    def sync_reason(self):
        reason = self.table.reason_from_index(self.row, self.column)
        self.setChecked(reason is not None)
        self.setStyleSheet(f'background-color: rgb{reason_color_map[reason]};')

    def checkbox_callback(self, state):
        link1 = self.table.table_id_to_link_name(self.row)
        link2 = self.table.table_id_to_link_name(self.column)
        if state == Qt.Checked:
            reason = DisableCollisionReason.Unknown
        else:
            reason = None
        self.table.update_reason(link1, link2, reason)


class Table(QTableWidget):
    _reasons: Dict[Tuple[PrefixName, PrefixName], DisableCollisionReason]
    world: WorldTree

    def __init__(self, world: WorldTree, collision_scene: BetterPyBulletSyncer):
        super().__init__()
        self.cellClicked.connect(self.table_item_callback)
        self.world = world
        self.collision_scene = collision_scene
        self.ros_visualizer = ROSMsgVisualization('map')

    def get_widget(self, row, column):
        return self.cellWidget(row, column).layout().itemAt(0).widget()

    def prefix_reasons_to_str_reasons(self, reasons: Dict[Tuple[PrefixName, PrefixName], DisableCollisionReason]) \
            -> Dict[Tuple[str, str], DisableCollisionReason]:
        return {(x[0].short_name, x[1].short_name): reason for x, reason in reasons.items()}

    @property
    def str_reasons(self):
        return self.prefix_reasons_to_str_reasons(self._reasons)

    @property
    def reasons(self):
        return self._reasons

    def table_id_to_link_name(self, index: int):
        return self.link_names[index]

    def sort_links(self, link1, link2):
        return tuple(sorted((link1, link2)))

    def update_reason(self, link1: str, link2: str, new_reason: Optional[DisableCollisionReason]):
        link1 = self.world.search_for_link_name(link1)
        link2 = self.world.search_for_link_name(link2)
        key = self.sort_links(link1, link2)
        if new_reason is None:
            if key in self._reasons:
                del self._reasons[key]
        else:
            self._reasons[key] = new_reason
        row = self.link_names.index(link1.short_name)
        column = self.link_names.index(link2.short_name)
        self.get_widget(row, column).sync_reason()
        self.get_widget(column, row).sync_reason()

    def reason_from_index(self, row, column):
        link1 = self.table_id_to_link_name(row)
        link2 = self.table_id_to_link_name(column)
        key = tuple(sorted((link1, link2)))
        r_key = (key[1], key[0])
        reasons = self.str_reasons
        if key in reasons:
            return reasons[key]
        elif r_key in reasons:
            return reasons[r_key]
        return None

    def table_item_callback(self, row, column):
        self.ros_visualizer.clear_marker()
        self.collision_scene.sync()
        for link_name in self.world.link_names_with_collisions:
            self.world.links[link_name].dye_collisions(self.world.default_link_color)
        link1 = self.world.search_for_link_name(self.link_names[row])
        link2 = self.world.search_for_link_name(self.link_names[column])
        key = self.sort_links(link1, link2)
        reason = self.reasons.get(key, None)
        color = reason_color_map[reason]
        color_msg = ColorRGBA(color[0] / 255, color[1] / 255, color[2] / 255, 1)
        self.world.links[link1].dye_collisions(color_msg)
        self.world.links[link2].dye_collisions(color_msg)
        self.world.reset_cache()
        self.ros_visualizer.publish_markers()

    @property
    def link_names(self) -> List[str]:
        return list(sorted(x.short_name for x in self.world.link_names_with_collisions))

    def add_table_item(self, row, column):
        checkbox = ReasonCheckBox(self, row, column)
        checkbox.sync_reason()
        checkbox.connect_callback()
        if row == column:
            checkbox.setDisabled(True)

        widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(checkbox)
        layout.setAlignment(checkbox, Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        widget.setLayout(layout)
        self.setCellWidget(row, column, widget)

    def update_table(self, reasons: Dict[Tuple[PrefixName, PrefixName], DisableCollisionReason]):
        self._reasons = {self.sort_links(*k): v for k, v in reasons.items()}
        self.clear()
        self.setRowCount(len(self.link_names))
        self.setColumnCount(len(self.link_names))
        self.setHorizontalHeaderLabels(self.link_names)
        self.setVerticalHeaderLabels(self.link_names)

        for x, link1 in enumerate(self.link_names):
            for y, link2 in enumerate(self.link_names):
                self.add_table_item(x, y)

        num_rows = self.rowCount()

        widths = []

        for x in range(num_rows):
            item = self.item(x, 0)
            if item is not None:
                widths.append(item.sizeHint().width())
        if widths:
            self.setColumnWidth(0, max(widths))


def get_readable_color(red: float, green: float, blue: float) -> Tuple[int, int, int]:
    luminance = ((0.299 * red) + (0.587 * green) + (0.114 * blue)) / 255
    if luminance > 0.5:
        return 0, 0, 0
    else:
        return 255, 255, 255


class Application(QMainWindow):
    robot_description = 'robot_description'

    def __init__(self):
        super().__init__()
        self.world = WorldTree.empty_world()
        self.world.default_link_color = ColorRGBA(0.5, 0.5, 0.5, 1)
        self.collision_scene = BetterPyBulletSyncer.empty(self.world)
        self.df = pd.DataFrame()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Link Collisions')
        self.setMinimumSize(800, 600)

        self.progress = QProgressBar()

        # Create QLineEdit for the URDF file path
        self.urdf_file_path_input = QLineEdit()
        self.urdf_file_path_input.setText('robot_description')

        # Create Browse button for URDF file
        self.urdf_browse_button = QPushButton('...')
        self.urdf_browse_button.clicked.connect(self.urdf_browse)

        # Create QLineEdit for the SRDF file path
        self.srdf_file_path_input = QLineEdit()

        # Create Browse button for SRDF file
        self.srdf_browse_button = QPushButton('...')
        self.srdf_browse_button.clicked.connect(self.srdf_browse)

        # Create Load and Save buttons
        self.load_urdf_file_button = QPushButton('Load urdf from file')
        self.load_urdf_file_button.clicked.connect(self.load_urdf_from_path)
        self.load_urdf_param_button = QPushButton('Load urdf from parameter server')
        self.load_urdf_param_button.clicked.connect(self.load_urdf_from_paramserver)

        self.load_srdf_button = QPushButton('Load srdf')
        self.load_srdf_button.clicked.connect(self.load_srdf)
        self.compute_srdf_button = QPushButton('Compute self collision matrix')
        self.compute_srdf_button.clicked.connect(self.compute_self_collision_matrix)
        self.save_srdf_button = QPushButton('Save srdf')
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

        self.table = Table(self.world, self.collision_scene)

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
            label.setStyleSheet(f'background-color: rgb{color}; color: rgb{get_readable_color(*color)};')
            if reason == DisableCollisionReason.Never:
                label.setToolTip('These links are never in contact.')
            elif reason == DisableCollisionReason.Unknown:
                label.setToolTip('This link pair was disabled for an unknown reason.')
            elif reason == DisableCollisionReason.Adjacent:
                label.setToolTip('This link pair is only connected by joints that cannot move.')
            elif reason == DisableCollisionReason.Default:
                label.setToolTip('This link pair is in collision in the robot\'s default state.')
            elif reason == DisableCollisionReason.AlmostAlways:
                label.setToolTip('This link pair is almost always in collision.')
            else:
                label.setToolTip('Collisions will be computed.')
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
        reasons = self.collision_scene.compute_self_collision_matrix(self.group_name,
                                                                     save_to_tmp=False,
                                                                     non_controlled=True,
                                                                     progress_callback=self.set_progress)
        self.table.update_table(reasons)
        self.set_progress(100, 'done checking collisions')

    def srdf_browse(self):
        # Open a file dialog and get the selected file path
        file_path, _ = QFileDialog.getOpenFileName()

        # If a file path was selected, update the QLineEdit
        if file_path:
            self.srdf_file_path_input.setText(file_path)

    def load_urdf_from_paramserver(self):
        robot_description = self.urdf_file_path_input.text()
        if rospy.has_param(robot_description):
            urdf = rospy.get_param(self.robot_description)
            self.load_urdf(urdf)
        else:
            QMessageBox.critical(self, 'Error', f'Parameter not found: \n{robot_description}')

    def load_urdf(self, urdf):
        self.world._clear()
        self.set_progress(0, 'loading urdf from parameter server')
        group_name = robot_name_from_urdf_string(urdf)
        self.set_progress(10, 'parsing urdf')
        self.world.add_urdf(urdf, group_name)
        self.set_progress(50, 'updating table')
        self.table.update_table({})
        self.set_tmp_srdf_path()
        self.world.god_map.set_data(identifier.controlled_joints, self.world.movable_joint_names)
        self.set_progress(100, 'done loading urdf')

    def set_tmp_srdf_path(self):
        if len(self.world.group_names) > 0 and self.srdf_file_path_input.text() == '':
            self.srdf_file_path_input.setText(self.collision_scene.get_path_to_self_collision_matrix(self.group_name))

    def load_srdf(self):
        srdf_file = self.srdf_file_path_input.text()

        try:

            # Extract collision data from SRDF file
            if os.path.isfile(srdf_file):
                reasons = self.collision_scene.load_black_list_from_srdf(srdf_file, self.group_name, False)
                self.table.update_table(reasons)
            else:
                QMessageBox.critical(self, 'Error', f'File does not exist: \n{srdf_file}')
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, 'Error', str(e))

    def load_urdf_from_path(self):
        urdf_file = self.urdf_file_path_input.text()

        if not os.path.isfile(urdf_file):
            QMessageBox.critical(self, 'Error', f'File does not exist: \n{urdf_file}')
            return

        with open(urdf_file, 'r') as f:
            self.load_urdf(f.read())

    @property
    def group_name(self):
        return list(self.world.group_names)[0]

    def save_srdf(self):
        self.collision_scene.save_black_list(self.world.groups[self.group_name],
                                             self.table.reasons,
                                             file_name=self.srdf_file_path_input.text())
        self.set_progress(100, f'Saved {self.srdf_file_path_input.text()}')


def main():
    # Display DataFrame in PyQt5 GUI
    app = QApplication(sys.argv)
    window = Application()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    rospy.init_node('self_collision_matrix_updater')
    main()
