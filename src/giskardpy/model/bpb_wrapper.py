import os
from typing import List, Tuple, Optional

import betterpybullet as pb
import numpy as np
import trimesh

from giskardpy import identifier
from giskardpy.god_map import GodMap
from giskardpy.model.collision_world_syncer import Collision
from giskardpy.model.links import Link, LinkGeometry, BoxGeometry, SphereGeometry, CylinderGeometry, MeshGeometry
from giskardpy.my_types import my_string, PrefixName
from giskardpy.utils import logging
from giskardpy.utils.math import inverse_frame
from giskardpy.utils.utils import resolve_ros_iris, to_tmp_path, write_to_tmp, suppress_stdout

CollisionObject = pb.CollisionObject

if not hasattr(pb, '__version__') or pb.__version__ != '1.0.0':
    raise ImportError('Betterpybullet is outdated.')


class BPCollisionWrapper(Collision):
    def __init__(self, pb_collision: pb.Collision):
        self.pb_collision = pb_collision
        self.link_a = self.pb_collision.obj_a.name
        self.link_b = self.pb_collision.obj_b.name
        self.original_link_a = self.link_a
        self.original_link_b = self.link_b
        self.is_external = None
        self.new_a_P_pa = None
        self.new_b_P_pb = None
        self.new_b_V_n = None

    @property
    def map_P_pa(self):
        return self.pb_collision.map_P_pa

    @property
    def map_P_pb(self):
        return self.pb_collision.map_P_pb

    @property
    def map_V_n(self):
        return self.pb_collision.world_V_n

    @property
    def a_P_pa(self):
        return self.pb_collision.a_P_pa

    @property
    def b_P_pb(self):
        return self.pb_collision.b_P_pb

    @property
    def contact_distance(self):
        return self.pb_collision.contact_distance

    @property
    def link_b_hash(self):
        return self.link_b.__hash__()


def create_cube_shape(extents: Tuple[float, float, float]) -> pb.BoxShape:
    out = pb.BoxShape(pb.Vector3(*[extents[x] * 0.5 for x in range(3)])) if type(
        extents) is not pb.Vector3 else pb.BoxShape(extents)
    out.margin = 0.001
    return out


def to_giskard_collision(collision: pb.Collision):
    return BPCollisionWrapper(collision)


def create_cylinder_shape(diameter: float, height: float) -> pb.CylinderShape:
    # out = pb.CylinderShapeZ(pb.Vector3(0.5 * diameter, 0.5 * diameter, height * 0.5))
    # out.margin = 0.001
    # Weird thing: The default URDF loader in bullet instantiates convex meshes. Idk why.
    return load_convex_mesh_shape(resolve_ros_iris('package://giskardpy/test/urdfs/meshes/cylinder.obj'),
                                  single_shape=True,
                                  scale=[diameter, diameter, height])


def create_sphere_shape(diameter: float) -> pb.SphereShape:
    out = pb.SphereShape(0.5 * diameter)
    out.margin = 0.001
    return out


def create_shape_from_geometry(geometry: LinkGeometry) -> pb.CollisionShape:
    if isinstance(geometry, BoxGeometry):
        shape = create_cube_shape((geometry.depth, geometry.width, geometry.height))
    elif isinstance(geometry, SphereGeometry):
        shape = create_sphere_shape(geometry.radius * 2)
    elif isinstance(geometry, CylinderGeometry):
        shape = create_cylinder_shape(geometry.radius * 2, geometry.height)
    elif isinstance(geometry, MeshGeometry):
        shape = load_convex_mesh_shape(geometry.file_name_absolute, scale=geometry.scale)
        geometry.set_collision_file_name(shape.file_path)
    else:
        raise NotImplementedError()
    return shape


def create_shape_from_link(link: Link, collision_id: int = 0) -> pb.CollisionObject:
    # if len(link.collisions) > 1:
    shapes = []
    map_T_o = None
    for collision_id, geometry in enumerate(link.collisions):
        if map_T_o is None:
            shape = create_shape_from_geometry(geometry)
        else:
            shape = create_shape_from_geometry(geometry)
        link_T_geometry = pb.Transform.from_np(geometry.link_T_geometry.evaluate())
        shapes.append((link_T_geometry, shape))
    shape = create_compound_shape(shapes_poses=shapes)
    # else:
    #     shape = create_shape_from_geometry(link.collisions[0])
    return create_object(link.name, shape, pb.Transform.identity())


def create_compound_shape(shapes_poses: List[Tuple[pb.Transform, pb.CollisionShape]] = None) -> pb.CompoundShape:
    out = pb.CompoundShape()
    for t, s in shapes_poses:
        out.add_child(t, s)
    return out


# Technically the tracker is not required here,
# since the loader keeps references to the loaded shapes.
def load_convex_mesh_shape(pkg_filename: str, single_shape=False, scale=(1, 1, 1)) -> pb.ConvexShape:
    if not pkg_filename.endswith('.obj'):
        obj_pkg_filename = convert_to_decomposed_obj_and_save_in_tmp(pkg_filename)
    else:
        obj_pkg_filename = pkg_filename
    return pb.load_convex_shape(resolve_ros_iris(obj_pkg_filename),
                                single_shape=single_shape,
                                scaling=pb.Vector3(scale[0], scale[1], scale[2]))


def convert_to_decomposed_obj_and_save_in_tmp(file_name: str, log_path='/tmp/giskardpy/vhacd.log'):
    first_group_name = list(GodMap().get_data(identifier.world).groups.keys())[0]
    resolved_old_path = resolve_ros_iris(file_name)
    short_file_name = file_name.split('/')[-1][:-3]
    obj_file_name = f'{first_group_name}/{short_file_name}obj'
    new_path_original = to_tmp_path(obj_file_name)
    if not os.path.exists(new_path_original):
        mesh = trimesh.load(resolved_old_path, force='mesh')
        obj_str = trimesh.exchange.obj.export_obj(mesh)
        write_to_tmp(obj_file_name, obj_str)
        logging.loginfo(f'Converted {file_name} to obj and saved in {new_path_original}.')
    new_path = new_path_original

    new_path_decomposed = new_path_original.replace('.obj', '_decomposed.obj')
    if not os.path.exists(new_path_decomposed):
        mesh = trimesh.load(new_path_original, force='mesh')
        if not trimesh.convex.is_convex(mesh):
            logging.loginfo(f'{file_name} is not convex, applying vhacd.')
            with suppress_stdout():
                pb.vhacd(new_path_original, new_path_decomposed, log_path)
            new_path = new_path_decomposed
    else:
        new_path = new_path_decomposed

    return new_path


def create_object(name: PrefixName, shape: pb.CollisionShape, transform: Optional[pb.Transform] = None) \
        -> pb.CollisionObject:
    if transform is None:
        transform = pb.Transform.identity()
    out = pb.CollisionObject(name)
    out.collision_shape = shape
    out.collision_flags = pb.CollisionObject.KinematicObject
    out.transform = transform
    return out


def create_cube(extents, transform=pb.Transform.identity()):
    return create_object(create_cube_shape(extents), transform)


def create_sphere(diameter, transform=pb.Transform.identity()):
    return create_object(create_sphere_shape(diameter), transform)


def create_cylinder(diameter, height, transform=pb.Transform.identity()):
    return create_object(create_cylinder_shape(diameter, height), transform)


def create_compund_object(shapes_transforms, transform=pb.Transform.identity()):
    return create_object(create_compound_shape(shapes_transforms), transform)


def create_convex_mesh(pkg_filename, transform=pb.Transform.identity()):
    return create_object(load_convex_mesh_shape(pkg_filename), transform)


def vector_to_cpp_code(vector):
    return 'btVector3({:.6f}, {:.6f}, {:.6f})'.format(vector.x, vector.y, vector.z)


def quaternion_to_cpp_code(quat):
    return 'btQuaternion({:.6f}, {:.6f}, {:.6f}, {:.6f})'.format(quat.x, quat.y, quat.z, quat.w)


def transform_to_cpp_code(transform):
    return 'btTransform({}, {})'.format(quaternion_to_cpp_code(transform.rotation),
                                        vector_to_cpp_code(transform.origin))


BOX = 0
SPHERE = 1
CYLINDER = 2
COMPOUND = 3
CONVEX = 4


def shape_to_cpp_code(s, shape_names, shape_type_names):
    buf = ''
    if isinstance(s, pb.BoxShape):
        s_name = 'shape_box_{}'.format(shape_type_names[BOX])
        shape_names[s] = s_name
        buf += 'auto {} = std::make_shared<btBoxShape>({});\n'.format(s_name, vector_to_cpp_code(s.extents * 0.5))
        shape_type_names[BOX] += 1
    elif isinstance(s, pb.SphereShape):
        s_name = 'shape_sphere_{}'.format(shape_type_names[SPHERE])
        shape_names[s] = s_name
        buf += 'auto {} = std::make_shared<btSphereShape>({:.3f});\n'.format(s_name, s.radius)
        shape_type_names[SPHERE] += 1
    elif isinstance(s, pb.CylinderShape):
        height = s.height
        diameter = s.radius * 2
        s_name = 'shape_cylinder_{}'.format(shape_type_names[CYLINDER])
        shape_names[s] = s_name

        buf += 'auto {} = std::make_shared<btCylinderShapeZ>({});\n'.format(s_name, vector_to_cpp_code(
            pb.Vector3(diameter, diameter, height)))
        shape_type_names[CYLINDER] += 1
    elif isinstance(s, pb.CompoundShape):
        if s.file_path != '':
            s_name = 'shape_convex_{}'.format(shape_type_names[CONVEX])
            shape_names[s] = s_name
            buf += 'auto {} = load_convex_shape("{}", false);\n'.format(s_name, s.file_path)
            shape_type_names[CONVEX] += 1
        else:
            s_name = 'shape_compound_{}'.format(shape_type_names[COMPOUND])
            shape_names[s] = s_name
            buf += 'auto {} = std::make_shared<btCompoundShape>();\n'.format(s_name)
            shape_type_names[COMPOUND] += 1
            for x in range(s.nchildren):
                ss = s.get_child(x)
                buf += shape_to_cpp_code(ss, shape_names, shape_type_names)
                buf += '{}->addChildShape({}, {});\n'.format(s_name, transform_to_cpp_code(s.get_child_transform(x)),
                                                             shape_names[ss])
    elif isinstance(s, pb.ConvexHullShape):
        if s.file_path != '':
            s_name = 'shape_convex_{}'.format(shape_type_names[CONVEX])
            shape_names[s] = s_name
            buf += 'auto {} = load_convex_shape("{}", true);\n'.format(s_name, s.file_path)
            shape_type_names[CONVEX] += 1
    return buf


def world_to_cpp_code(subworld):
    shapes = {o.collision_shape for o in subworld.collision_objects}
    shape_names = {}

    shape_type_names = {BOX: 0, SPHERE: 0, CYLINDER: 0, COMPOUND: 0, CONVEX: 0}

    buf = 'KineverseWorld world;\n\n'
    buf += '\n'.join(shape_to_cpp_code(s, shape_names, shape_type_names) for s in shapes)

    obj_names = []  # store the c++ names
    for name, obj in sorted(subworld.named_objects.items()):
        o_name = '_'.join(name)
        obj_names.append(o_name)
        buf += 'auto {o_name} = std::make_shared<KineverseCollisionObject>();\n{o_name}->setWorldTransform({transform});\n{o_name}->setCollisionShape({shape});\n\n'.format(
            o_name=o_name, transform=transform_to_cpp_code(obj.transform), shape=shape_names[obj.collision_shape])

    buf += '\n'.join('world.addCollisionObject({});'.format(n) for n in obj_names)

    return buf + '\n'
