import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from optimizer import N

from optimization.consts import a, b, c


def sph2cart(r, theta, phi):
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)

    return x, y, z

def convert_array_to_cartesian(a, r=1):
    a2 = []
    for v in a:
        ox, oy, oz = sph2cart(r, v[3], v[4])
        a2.append([v[0], v[1], v[2], ox, oy, oz])
    
    return np.array(a2)

# TODO: change main so that it reads from result.txt
def main():
    res = [-0.2549695 , -0.26024959, -0.04116991,  4.55302766,  2.34250403,
        0.26812244,  0.23423874,  0.04554912,  0.2848681 ,  0.43574213,
        0.23814835, -0.15050066,  0.07083664,  1.97243031,  2.85296347,
       -0.33386135,  0.14673292,  0.04089422,  4.44270847,  2.37191305,
       -0.11049478, -0.03485805,  0.09568184,  4.95568384,  2.36066821,
        0.14968423,  0.08287538, -0.0902829 ,  0.84890831,  2.31759077,
       -0.090786  ,  0.31188492, -0.05822582,  3.50233179,  1.81443672,
        0.28789819,  0.14467129, -0.05910549,  3.40934337,  2.84148023]

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
    soa = convert_array_to_cartesian(pre_soa, r=0.5)

    X, Y, Z, U, V, W = zip(*soa)
    ax.quiver(X, Y, Z, U, V, W)

    # Plot the blimp (ellipsoid with semi-axes a, b, and c)
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