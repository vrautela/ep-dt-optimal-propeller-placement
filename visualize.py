import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from optimizer import N

def sph2cart(r, theta, phi):
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)

    return x, y, z

def convert_array_to_cartesian(a):
    a2 = []
    for v in a:
        ox, oy, oz = sph2cart(1, v[3], v[4])
        a2.append([v[0], v[1], v[2], ox, oy, oz])
    
    return np.array(a2)

# TODO: change main so that it reads from result.txt
def main():
    res = [-0.8831457 ,  0.11481743, -0.4540797 ,  5.5846884 ,  2.07172036,
        0.85195907,  0.08302979, -0.51613984,  2.97327684,  0.17725267,
       -0.53108261,  0.58886146,  0.60884443,  5.77356432,  1.76495369,
        0.79889514,  0.15248755,  0.58035816,  0.60567167,  2.39875759,
       -0.33894067, -0.87878957,  0.33284919,  0.08579198,  1.24133127,
        0.56855128, -0.62818441, -0.52935466,  2.41784735,  2.55854404,
       -0.15472839, -0.53301054,  0.83100325,  0.24852747,  2.75233529,
        0.55380542,  0.68791383, -0.46824315,  5.15168833,  1.54771748]

    fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points on the blimp where the motors are
    xs = [res[5*i] for i in range(N)]
    ys = [res[5*i+1] for i in range(N)]
    zs = [res[5*i+2] for i in range(N)]
    default_s = mpl.rcParams['lines.markersize'] ** 2
    ax.scatter(xs, ys, zs, s=2*default_s, color='g')

    # Plot the arrows representing the directions of the motors
    pre_soa = [[res[5*i], res[5*i+1], res[5*i+2], res[5*i+3], res[5*i+4]] for i in range(8)]
    soa = convert_array_to_cartesian(pre_soa)

    X, Y, Z, U, V, W = zip(*soa)
    ax.quiver(X, Y, Z, U, V, W)

    # Plot the blimp (ellipsoid with semi-axes a, b, and c)
    a, b, c = 1, 1, 1
    # Radii corresponding to the coefficients:
    rx, ry, rz = a, b, c

    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):
    x = rx * np.outer(np.cos(u), np.sin(v))
    y = ry * np.outer(np.sin(u), np.sin(v))
    z = rz * np.outer(np.ones_like(u), np.cos(v))

    # Plot:
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='r')

    # Adjustment of the axes, so that they all have the same span:
    max_radius = max(rx, ry, rz)
    for axis in 'xyz':
        getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))

    plt.show()

if __name__ == "__main__":
    main()