import numpy as np
from util.camera_pose_visualizer import CameraPoseVisualizer

if __name__ == '__main__':
    # argument : the minimum/maximum value of x, y, z
    # visualizer = CameraPoseVisualizer([-5000, 5000], [-5000, 5000], [0, 2000])
    visualizer = CameraPoseVisualizer([-5000, 5000], [-5000, 5000], [-2000, 2000])

    # argument : extrinsic matrix, color, scaled focal length(z-axis length of frame body of camera
    t = np.array([
        [-6.21646531e-01,  7.82776391e-01,  2.85781748e-02,-1.45034606e+01],
        [ 7.44849355e-02,  9.53930153e-02, -9.92649065e-01,1.17429720e+02],
        [-7.79748411e-01, -6.14948205e-01, -1.17605786e-01,2.90198438e+02],
        [0, 0, 0, 1]])

    visualizer.extrinsic2pyramid(t, 'c', 1000)

    visualizer.show()
