import os
import tempfile
from typing import List, Tuple, Optional

import giskardpy_bullet_bindings as pb
import trimesh
from pkg_resources import resource_filename

from giskardpy.middleware import get_middleware
from giskardpy.model.collisions import GiskardCollision
from giskardpy.utils.utils import suppress_stdout, create_path
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import (
    Shape,
    Box,
    Sphere,
    Cylinder,
    Mesh,
    Scale,
    TriangleMesh,
)
from semantic_digital_twin.world_description.world_entity import Body

CollisionObject = pb.CollisionObject


def create_collision(pb_collision: pb.Collision, world: World) -> GiskardCollision:
    collision = GiskardCollision(
        body_a=world.get_kinematic_structure_entity_by_name(pb_collision.obj_a.name),
        body_b=world.get_kinematic_structure_entity_by_name(pb_collision.obj_b.name),
        contact_distance_input=pb_collision.contact_distance,
        map_P_pa=pb_collision.map_P_pa,
        map_P_pb=pb_collision.map_P_pb,
        map_V_n_input=pb_collision.world_V_n,
        a_P_pa=pb_collision.a_P_pa,
        b_P_pb=pb_collision.b_P_pb,
    )
    collision.original_body_a = collision.body_a
    collision.original_body_b = collision.body_b
    collision.is_external = None
    return collision


def create_cube_shape(extents: Tuple[float, float, float]) -> pb.BoxShape:
    out = (
        pb.BoxShape(pb.Vector3(*[extents[x] * 0.5 for x in range(3)]))
        if type(extents) is not pb.Vector3
        else pb.BoxShape(extents)
    )
    out.margin = 0.001
    return out


def create_cylinder_shape(
    diameter: float, height: float, tmp_folder: str
) -> pb.CylinderShape:
    # out = pb.CylinderShapeZ(pb.Vector3(0.5 * diameter, 0.5 * diameter, height * 0.5))
    # out.margin = 0.001
    # Weird thing: The default URDF loader in bullet instantiates convex meshes. Idk why.
    file_name = resource_filename("giskardpy", "../resources/meshes/cylinder.obj")
    return load_convex_mesh_shape(
        pkg_filename=file_name,
        tmp_folder=tmp_folder,
        single_shape=True,
        scale=Scale(diameter, diameter, height),
    )


def create_sphere_shape(diameter: float) -> pb.SphereShape:
    out = pb.SphereShape(0.5 * diameter)
    out.margin = 0.001
    return out


def create_shape_from_geometry(geometry: Shape, tmp_folder: str) -> pb.CollisionShape:
    if isinstance(geometry, Box):
        shape = create_cube_shape(
            (geometry.scale.x, geometry.scale.y, geometry.scale.z)
        )
    elif isinstance(geometry, Sphere):
        shape = create_sphere_shape(geometry.radius * 2)
    elif isinstance(geometry, Cylinder):
        shape = create_cylinder_shape(
            diameter=geometry.width, height=geometry.height, tmp_folder=tmp_folder
        )
    elif isinstance(geometry, TriangleMesh):
        f = tempfile.NamedTemporaryFile(delete=False, suffix=".obj")
        with open(f.name, "w") as fd:
            fd.write(trimesh.exchange.obj.export_obj(geometry.mesh))
        shape = load_convex_mesh_shape(
            pkg_filename=f.name,
            tmp_folder=tmp_folder,
            single_shape=False,
            scale=geometry.scale,
        )
    elif isinstance(geometry, Mesh):
        shape = load_convex_mesh_shape(
            pkg_filename=geometry.filename,
            tmp_folder=tmp_folder,
            single_shape=False,
            scale=geometry.scale,
        )
        # todo geometry.set_collision_file_name(shape.file_path)
    else:
        raise NotImplementedError()
    return shape


def create_shape_from_link(
    link: Body, tmp_folder: str, collision_id: int = 0
) -> pb.CollisionObject:
    # if len(link.collisions) > 1:
    shapes = []
    map_T_o = None
    for collision_id, geometry in enumerate(link.collision):
        if map_T_o is None:
            shape = create_shape_from_geometry(geometry=geometry, tmp_folder=tmp_folder)
        else:
            shape = create_shape_from_geometry(geometry=geometry, tmp_folder=tmp_folder)
        link_T_geometry = pb.Transform.from_np(geometry.origin.to_np())
        shapes.append((link_T_geometry, shape))
    shape = create_compound_shape(shapes_poses=shapes)
    # else:
    #     shape = create_shape_from_geometry(link.collisions[0])
    return create_object(link.name, shape, pb.Transform.identity())


def create_compound_shape(
    shapes_poses: List[Tuple[pb.Transform, pb.CollisionShape]] = None,
) -> pb.CompoundShape:
    out = pb.CompoundShape()
    for t, s in shapes_poses:
        out.add_child(t, s)
    return out


# Technically the tracker is not required here,
# since the loader keeps references to the loaded shapes.
def load_convex_mesh_shape(
    pkg_filename: str, tmp_folder: str, single_shape: bool, scale: Scale
) -> pb.ConvexShape:
    if not pkg_filename.endswith(".obj"):
        obj_pkg_filename = convert_to_decomposed_obj_and_save_in_tmp(
            file_name=pkg_filename, tmp_folder=tmp_folder
        )
    else:
        obj_pkg_filename = pkg_filename
    return pb.load_convex_shape(
        get_middleware().resolve_iri(obj_pkg_filename),
        single_shape=single_shape,
        scaling=pb.Vector3(scale.x, scale.y, scale.z),
    )


def convert_to_decomposed_obj_and_save_in_tmp(
    file_name: str, tmp_folder: str, log_path="/tmp/giskardpy/vhacd.log"
) -> str:
    hopefully_unique_name = "_".join(file_name.split("/")[-4:])
    resolved_old_path = get_middleware().resolve_iri(file_name)
    short_file_name = file_name.split("/")[-1].split(".")[0]
    obj_file_name = os.path.join(hopefully_unique_name, f"{short_file_name}.obj")
    new_path_original = os.path.join(tmp_folder, obj_file_name)
    if not os.path.exists(new_path_original):
        mesh = trimesh.load(resolved_old_path, force="mesh")
        obj_str = trimesh.exchange.obj.export_obj(mesh)

        create_path(new_path_original)
        with open(new_path_original, "w") as f:
            f.write(obj_str)

        get_middleware().loginfo(
            f"Converted {file_name} to obj and saved in {new_path_original}."
        )
    new_path = new_path_original

    new_path_decomposed = new_path_original.replace(".obj", "_decomposed.obj")
    if not os.path.exists(new_path_decomposed):
        mesh = trimesh.load(new_path_original, force="mesh")
        if not trimesh.convex.is_convex(mesh):
            get_middleware().loginfo(f"{file_name} is not convex, applying vhacd.")
            with suppress_stdout():
                pb.vhacd(new_path_original, new_path_decomposed, log_path)
            new_path = new_path_decomposed
    else:
        new_path = new_path_decomposed

    return new_path


def create_object(
    name: PrefixedName,
    shape: pb.CollisionShape,
    transform: Optional[pb.Transform] = None,
) -> pb.CollisionObject:
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


def create_convex_mesh(
    pkg_filename: str, tmp_folder: str, transform=pb.Transform.identity()
):
    return create_object(
        load_convex_mesh_shape(
            pkg_filename=pkg_filename,
            tmp_folder=tmp_folder,
            single_shape=False,
            scale=Scale(1.0, 1.0, 1.0),
        ),
        transform,
    )


def vector_to_cpp_code(vector):
    return "btVector3({:.6f}, {:.6f}, {:.6f})".format(vector.x, vector.y, vector.z)


def quaternion_to_cpp_code(quat):
    return "btQuaternion({:.6f}, {:.6f}, {:.6f}, {:.6f})".format(
        quat.x, quat.y, quat.z, quat.w
    )


def transform_to_cpp_code(transform):
    return "btTransform({}, {})".format(
        quaternion_to_cpp_code(transform.rotation), vector_to_cpp_code(transform.origin)
    )


BOX = 0
SPHERE = 1
CYLINDER = 2
COMPOUND = 3
CONVEX = 4


def shape_to_cpp_code(s, shape_names, shape_type_names):
    buf = ""
    if isinstance(s, pb.BoxShape):
        s_name = "shape_box_{}".format(shape_type_names[BOX])
        shape_names[s] = s_name
        buf += "auto {} = std::make_shared<btBoxShape>({});\n".format(
            s_name, vector_to_cpp_code(s.extents * 0.5)
        )
        shape_type_names[BOX] += 1
    elif isinstance(s, pb.SphereShape):
        s_name = "shape_sphere_{}".format(shape_type_names[SPHERE])
        shape_names[s] = s_name
        buf += "auto {} = std::make_shared<btSphereShape>({:.3f});\n".format(
            s_name, s.radius
        )
        shape_type_names[SPHERE] += 1
    elif isinstance(s, pb.CylinderShape):
        height = s.height
        diameter = s.radius * 2
        s_name = "shape_cylinder_{}".format(shape_type_names[CYLINDER])
        shape_names[s] = s_name

        buf += "auto {} = std::make_shared<btCylinderShapeZ>({});\n".format(
            s_name, vector_to_cpp_code(pb.Vector3(diameter, diameter, height))
        )
        shape_type_names[CYLINDER] += 1
    elif isinstance(s, pb.CompoundShape):
        if s.file_path != "":
            s_name = "shape_convex_{}".format(shape_type_names[CONVEX])
            shape_names[s] = s_name
            buf += 'auto {} = load_convex_shape("{}", false);\n'.format(
                s_name, s.file_path
            )
            shape_type_names[CONVEX] += 1
        else:
            s_name = "shape_compound_{}".format(shape_type_names[COMPOUND])
            shape_names[s] = s_name
            buf += "auto {} = std::make_shared<btCompoundShape>();\n".format(s_name)
            shape_type_names[COMPOUND] += 1
            for x in range(s.nchildren):
                ss = s.get_child(x)
                buf += shape_to_cpp_code(ss, shape_names, shape_type_names)
                buf += "{}->addChildShape({}, {});\n".format(
                    s_name,
                    transform_to_cpp_code(s.get_child_transform(x)),
                    shape_names[ss],
                )
    elif isinstance(s, pb.ConvexHullShape):
        if s.file_path != "":
            s_name = "shape_convex_{}".format(shape_type_names[CONVEX])
            shape_names[s] = s_name
            buf += 'auto {} = load_convex_shape("{}", true);\n'.format(
                s_name, s.file_path
            )
            shape_type_names[CONVEX] += 1
    return buf


def world_to_cpp_code(subworld):
    shapes = {o.collision_shape for o in subworld.collision_objects}
    shape_names = {}

    shape_type_names = {BOX: 0, SPHERE: 0, CYLINDER: 0, COMPOUND: 0, CONVEX: 0}

    buf = "KineverseWorld world;\n\n"
    buf += "\n".join(
        shape_to_cpp_code(s, shape_names, shape_type_names) for s in shapes
    )

    obj_names = []  # store the c++ names
    for name, obj in sorted(subworld.named_objects.items()):
        o_name = "_".join(name)
        obj_names.append(o_name)
        buf += "auto {o_name} = std::make_shared<KineverseCollisionObject>();\n{o_name}->setWorldTransform({transform});\n{o_name}->setCollisionShape({shape});\n\n".format(
            o_name=o_name,
            transform=transform_to_cpp_code(obj.transform),
            shape=shape_names[obj.collision_shape],
        )

    buf += "\n".join("world.addCollisionObject({});".format(n) for n in obj_names)

    return buf + "\n"
