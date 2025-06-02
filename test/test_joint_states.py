import numpy as np
from giskardpy.data_types.data_types import JointStates, PrefixName, Derivatives


def test_joint_states():
    joint1 = PrefixName('joint1')
    joint2 = PrefixName('joint2')
    js = JointStates()

    # Accessing a nonexistent joint auto-creates it as [0,0,0,0]
    assert np.allclose(js[joint1].data, np.zeros(4))
    assert np.allclose(js[joint2].data, np.zeros(4))

    # Iterate over joint names:
    for name in js:
        assert np.allclose(js[name].data, np.zeros(4))

    # Set position only (row 0)
    js[joint1][Derivatives.position] = 1.23
    js[joint1].velocity = 23
    assert js[joint1].position == 1.23
    assert js[joint1][Derivatives.velocity] == 23

    # Or overwrite all four entries at once:
    reference = np.array([0.5, 0.1, 0.0, -0.01])
    js[joint2] = reference
    assert np.allclose(js[joint2].data, reference)

    # the data should be continuous
    assert js.data.flags['C_CONTIGUOUS']
    assert np.allclose(js.data.ravel(), np.array([1.23, 0.5,    # position
                                                  23.0, 0.1,     # velocity
                                                  0.0, 0.0,     # acceleration
                                                  0.0, -0.01])) # jerk
    assert np.allclose(js.positions, np.array([1.23, 0.5]))
    assert np.allclose(js.velocities, np.array([23.0, 0.1]))
    assert np.allclose(js.accelerations, np.array([0.0, 0.0]))
    assert np.allclose(js.jerks, np.array([0.0, -0.01]))

    # Delete a joint:
    del js[joint1]
    assert joint1 not in js

    # data should still be continuous
    assert js.data.flags['C_CONTIGUOUS']
    assert np.allclose(js.data.ravel(), np.array([0.5,    # position
                                                  0.1,     # velocity
                                                  0.0,     # acceleration
                                                  -0.01])) # jerk

