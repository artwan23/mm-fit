import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, writers    
    
    
def plot_3d_pose(ax, pose):
    """
    Visualize a 3D skeleton.
    :param pose: numpy array (3 x 17) with x, y, z coordinates with COCO keypoint format.
    :param elev: Elevation angle in the z plane.
    :param azim: Azimuth angle in the x, y plane.
    :param figsize: Figure size.
    :return: None
    """
    pose = pose.flatten(order='F')
    vals = np.reshape(pose, (17, -1))
    artists = []

    #ax.view_init(elev, azim)
    
    #fig = plt.figure(figsize=figsize)
    #ax = Axes3D(fig)
    #ax.view_init(elev, azim)
    
    
    limbs = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12),
             (12, 13), (8, 14), (14, 15), (15, 16)]
    left_right_limb = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    for i, limb in enumerate(limbs):
        x, y, z = [np.array([vals[limb[0], j], vals[limb[1], j]]) for j in range(3)]
        if left_right_limb[i] == 0:
            cc = 'blue'
        elif left_right_limb[i] == 1:
            cc = 'red'
        else:
            cc = 'black'
        lines = ax.plot(x, y, z, marker='o', markersize=2, lw=1, c=cc)
        artists.extend(lines)
        
    return artists

def average_neighbors(arr, window_size=5):
    """
    Funkcja przeprowadza uśrednianie sąsiadujących wierszy dla każdej warstwy w tablicy.
    
    :param arr: Tablica ndarray o wymiarach (3, N, 18)
    :param window_size: Liczba sąsiadujących wierszy do uśrednienia (domyślnie 5)
    :return: Nowa tablica ndarray z uśrednionymi wartościami
    """
    num_layers, num_rows, num_columns = arr.shape
    averaged_array = np.zeros((num_layers, num_rows - window_size + 1, num_columns))
    kernel = math.floor(window_size/2)

    for layer in range(num_layers):
        for i in range(kernel, num_rows - kernel):
            averaged_array[layer, i - kernel] = np.mean(arr[layer, i - kernel:i + kernel + 1], axis=0)
    
    return averaged_array

workout_num = '00'
exercise = 'jumping_jacks'

loaded_data_noise = np.load('/mnt/c/Users/Artur/Documents/Github/mm-fit/mm-fit/w' + workout_num + '/w' + workout_num + '_pose_3d.npy')
labels = pd.read_csv('/mnt/c/Users/Artur/Documents/Github/mm-fit/mm-fit/w' + workout_num + '/w' + workout_num + '_labels.csv', header=None)

exercise_labels = labels[labels[3] == exercise]
frames_offset = labels[0][0]
start_frame = exercise_labels[0][exercise_labels.index[0]] - frames_offset
end_frame = exercise_labels[1][exercise_labels.index[-1]] - frames_offset
end_frame = start_frame + 1800

loaded_data = average_neighbors(loaded_data_noise, 3)
# loaded_data = loaded_data_noise

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111,projection='3d')
#f= loaded_data.shape[1] - 60000
# ims =[]

# for frame in range(start_frame, end_frame):
#     data = loaded_data[:,frame,1:]
#     sf = plot_3d_pose(ax, pose=data)
#     ims.append(sf)
    
#skel_3d = loaded_data[:, 5, :]
#skel_3d.shape

#data = loaded_data[:,frame,:]
#plot_3d_pose(ax, pose=loaded_data[:,1,1:])

def animate(frame):
    ax.clear()
    ax.set_xlim((-1000, 1000))
    ax.set_ylim((-1000, 1000))
    ax.set_zlim((-500, 1500))
    ax.set_box_aspect([2000, 2000, 2000])
    data = loaded_data[:, start_frame + frame, 1:]
    artists = plot_3d_pose(ax, pose=data)
    return artists

f = end_frame - start_frame
anim = FuncAnimation(fig, animate, frames=f, interval=100/3, blit=True)

# anim= FuncAnimation(fig, animate, frames=f, interval=10)

# Writer = writers['ffmpeg']
# writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
# anim.save('/mnt/c/Users/Artur/Videos/3d_animation.mp4', writer=writer)

#a = animate(10)
plt.show()