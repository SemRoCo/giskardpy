import betterpybullet as pb

from giskardpy.utils.utils import resolve_ros_iris

class MyCollisionObject(pb.CollisionObject):
    def __init__(self, name):
        super(MyCollisionObject, self).__init__()
        self.name = name

    def __repr__(self):
        return str(self.name)

    def __str__(self):
        return str(self.name)


def create_cube_shape(extents):
    out = pb.BoxShape(pb.Vector3(*[extents[x] * 0.5 for x in range(3)])) if type(extents) is not pb.Vector3 else pb.BoxShape(extents)
    out.margin = 0.001
    return out

def create_cylinder_shape(diameter, height):
    # out = pb.CylinderShapeZ(pb.Vector3(0.5 * diameter, 0.5 * diameter, height * 0.5))
    # out.margin = 0.001
    # Weird thing: The default URDF loader in bullet instantiates convex meshes. Idk why.
    return load_convex_mesh_shape(resolve_ros_iris('package://giskardpy/test/urdfs/meshes/cylinder.obj'),
                                  single_shape=True, 
                                  scale=[diameter, diameter, height])

def create_sphere_shape(diameter):
    out = pb.SphereShape(0.5 * diameter)
    out.margin = 0.001
    return out

def create_compound_shape(shapes_poses=[]):
    out = pb.CompoundShape()
    for t, s in shapes_poses:
        out.add_child(t, s)
    return out

# Technically the tracker is not required here, 
# since the loader keeps references to the loaded shapes.
def load_convex_mesh_shape(pkg_filename, single_shape=False, scale=[1, 1, 1]):
    return pb.load_convex_shape(resolve_ros_iris(pkg_filename),
                                single_shape=single_shape, 
                                scaling=pb.Vector3(scale[0], scale[1], scale[2]))

def create_object(name, shape, transform=pb.Transform.identity()):
    out = MyCollisionObject(name)
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
        height   = s.height
        diameter = s.radius * 2
        s_name = 'shape_cylinder_{}'.format(shape_type_names[CYLINDER])
        shape_names[s] = s_name

        buf += 'auto {} = std::make_shared<btCylinderShapeZ>({});\n'.format(s_name, vector_to_cpp_code(pb.Vector3(diameter, diameter, height)))
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
                buf += '{}->addChildShape({}, {});\n'.format(s_name, transform_to_cpp_code(s.get_child_transform(x)), shape_names[ss])
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

    obj_names = [] # store the c++ names
    for name, obj in sorted(subworld.named_objects.items()):
        o_name = '_'.join(name)
        obj_names.append(o_name)
        buf += 'auto {o_name} = std::make_shared<KineverseCollisionObject>();\n{o_name}->setWorldTransform({transform});\n{o_name}->setCollisionShape({shape});\n\n'.format(o_name=o_name, transform=transform_to_cpp_code(obj.transform), shape=shape_names[obj.collision_shape])

    buf += '\n'.join('world.addCollisionObject({});'.format(n) for n in obj_names)

    return buf + '\n'
