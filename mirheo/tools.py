def rotationToQuaternion(axis, angle):
    """
    Returns a quaternion corresponding to the rotation
    of angle `angle` around the axis `axis`
    """
    import numpy as np

    b = np.array(axis, dtype=float)
    
    b = b * (1.0 / np.linalg.norm(b))
    a = 0.5 * angle
    return np.concatenate((  [np.cos(a)], np.sin(a) * np.cos(b) )).tolist()

def eulerToQuaternion(pitch, roll, yaw):
    """
    Returns a quaternion corresponding to the rotation
    of euler angles pitch, roll, yaw
    """
    import numpy as np
    
    cy = np.cos(yaw * 0.5);
    sy = np.sin(yaw * 0.5);
    cr = np.cos(roll * 0.5);
    sr = np.sin(roll * 0.5);
    cp = np.cos(pitch * 0.5);
    sp = np.sin(pitch * 0.5);
    
    q = [cy * cr * cp + sy * sr * sp,
         cy * sr * cp - sy * cr * sp,
         cy * cr * sp + sy * sr * cp,
         sy * cr * cp - cy * sr * sp]

    return q
