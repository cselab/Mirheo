def rotationToQuaternion(axis, angle):
    """
    Returns a unit quaternion corresponding to the rotation
    of angle `angle` around the axis `axis`
    """
    import numpy as np

    b = np.array(axis, dtype=float)
    
    b = b * (1.0 / np.linalg.norm(b))
    a = 0.5 * angle

    return np.concatenate((  [np.cos(a)], np.sin(a) * np.cos(b) ))
