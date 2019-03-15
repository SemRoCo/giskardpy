from giskard_msgs.srv import UpdateWorldRequest
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker
from giskard_msgs.msg import WorldBody
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose as PoseMsg, Point as PointMsg, Quaternion as QuaternionMsg, Vector3

def to_marker(thing):
    """
    :type thing: Union[UpdateWorldRequest, WorldBody]
    :rtype: MarkerArray
    """
    ma = MarkerArray()
    if isinstance(thing, UrdfObject):
        pass
        # TODO
        # return urdf_object_to_marker_msg(thing)
    elif isinstance(thing, WorldBody):
        ma.markers.append(world_body_to_marker_msg(thing))
    elif isinstance(thing, UpdateWorldRequest):
        ma.markers.append(update_world_to_marker_msg(thing))
    return ma

def update_world_to_marker_msg(update_world_req, id=1, ns=u''):
    """
    :type update_world_req: UpdateWorldRequest
    :type id: int
    :type ns: str
    :rtype: Marker
    """
    m = world_body_to_marker_msg(update_world_req.body, id, ns)
    m.header = update_world_req.pose.header
    m.pose = update_world_req.pose.pose
    m.frame_locked = update_world_req.rigidly_attached
    if update_world_req.operation == UpdateWorldRequest.ADD:
        m.action = Marker.ADD
    elif update_world_req.operation == UpdateWorldRequest.REMOVE:
        m.action = Marker.DELETE
    elif update_world_req.operation == UpdateWorldRequest.REMOVE_ALL:
        m.action = Marker.DELETEALL
    return m


def world_body_to_marker_msg(world_body, id=1, ns=u''):
    """
    :type world_body: WorldBody
    :rtype: Marker
    """
    m = Marker()
    m.ns = u'{}/{}'.format(ns, world_body.name)
    m.id = id
    if world_body.type == WorldBody.URDF_BODY:
        raise Exception(u'can\'t convert urdfs body world object to marker array')
    elif world_body.type == WorldBody.PRIMITIVE_BODY:
        if world_body.shape.type == SolidPrimitive.BOX:
            m.type = Marker.CUBE
            m.scale = Vector3(*world_body.shape.dimensions)
        elif world_body.shape.type == SolidPrimitive.SPHERE:
            m.type = Marker.SPHERE
            m.scale = Vector3(world_body.shape.dimensions[0],
                              world_body.shape.dimensions[0],
                              world_body.shape.dimensions[0])
        elif world_body.shape.type == SolidPrimitive.CYLINDER:
            m.type = Marker.CYLINDER
            m.scale = Vector3(world_body.shape.dimensions[SolidPrimitive.CYLINDER_RADIUS],
                              world_body.shape.dimensions[SolidPrimitive.CYLINDER_RADIUS],
                              world_body.shape.dimensions[SolidPrimitive.CYLINDER_HEIGHT])
        else:
            raise Exception(u'world body type {} can\'t be converted to marker'.format(world_body.shape.type))
    elif world_body.type == WorldBody.MESH_BODY:
        m.type = Marker.MESH_RESOURCE
        m.mesh_resource = world_body.mesh
    m.color = ColorRGBA(0,1,0,0.8)
    return m


def urdf_object_to_marker_msg(urdf_object):
    """
    :type urdf_object: UrdfObject
    :rtype: MarkerArray
    """
    ma = MarkerArray()
    for visual_property in urdf_object.visual_props:
        m = Marker()
        m.color.r = 0
        m.color.g = 1
        m.color.b = 0
        m.color.a = 0.8
        m.ns = u'bullet/{}'.format(urdf_object.name)
        m.action = Marker.ADD
        m.id = 1337
        m.header.frame_id = u'map'
        m.pose.position.x = visual_property.origin.translation.x
        m.pose.position.y = visual_property.origin.translation.y
        m.pose.position.z = visual_property.origin.translation.z
        m.pose.orientation.x = visual_property.origin.rotation.x
        m.pose.orientation.y = visual_property.origin.rotation.y
        m.pose.orientation.z = visual_property.origin.rotation.z
        m.pose.orientation.w = visual_property.origin.rotation.w
        if isinstance(visual_property.geometry, BoxShape):
            m.type = Marker.CUBE
            m.scale.x = visual_property.geometry.x
            m.scale.y = visual_property.geometry.y
            m.scale.z = visual_property.geometry.z
        if isinstance(visual_property.geometry, MeshShape):
            m.type = Marker.MESH_RESOURCE
            m.mesh_resource = visual_property.geometry.filename
            m.scale.x = visual_property.geometry.scale[0]
            m.scale.y = visual_property.geometry.scale[1]
            m.scale.z = visual_property.geometry.scale[2]
            m.mesh_use_embedded_materials = True
        ma.markers.append(m)
    return ma
