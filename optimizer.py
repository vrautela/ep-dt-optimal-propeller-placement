import numpy as np
from math import pi
from scipy.optimize import Bounds, minimize, NonlinearConstraint


# TODO: define this like the author (ellipsoid of possible force vectors)
# list of force/moment vectors
min_fx, max_fx = -1, 2
min_fy, max_fy = -1, 1
min_fz, max_fz = -1, 1
min_tx, max_tx = -0.4, 0.4
min_ty, max_ty = -1.2, 1.2
min_tz, max_tz = -1.6, 1.6
F_des = []
for fx in np.linspace(min_fx, max_fx, 5):
    for fy in np.linspace(min_fy, max_fy, 5):
        for fz in np.linspace(min_fz, max_fz, 5):
            f = np.array([fx, fy, fz, 0, 0, 0])
            F_des.append(f)
for tx in np.linspace(min_tx, max_tx, 5):
    for ty in np.linspace(min_ty, max_ty, 5):
        for tz in np.linspace(min_tz, max_tz, 5):
            t = np.array([0, 0, 0, tx, ty, tz])
            F_des.append(t)

# number of propellers
N = 8

# number of variables to specify the position/orientation of a propeller
NUM_PROP_VAR = 5


def residual(T, G_p, F_des):
    # multiply G_p and T to get the produced force vector
    F = np.matmul(G_p, T)
    delta_F = F_des - F 
    mag_delta_F = np.linalg.norm(delta_F)

    # penalty factor associated with thrusts
    lambda_penalty = 0.01
    mag_T = np.linalg.norm(T)

    return mag_delta_F**2 + lambda_penalty*(mag_T**2)


def objective_function(x):
    total_residual = 0
    G_p = compute_gp_from_x(x)

    # add up the cost of the residuals for each desired force vector
    for f in F_des:
        # NOTE: there is a suboptimization problem here (WWTF T such that residual is minimized)
        # find T such that residual is minimized, return residual and T

        # initial (random) guess of T
        T0 = np.random.uniform(size=N)
        lower_bounds = [0] * N
        upper_bounds = [1] * N
        bounds = Bounds(lower_bounds, upper_bounds)

        res = minimize(residual, T0, args=(G_p, f,), bounds=bounds)
        min_residual = res.fun
        min_T = res.x

        total_residual += min_residual
    
    return total_residual


def compute_gp_from_x(x0):
    pre_G_p_transpose = []
    for i in range(N):
        # generate the vector of forces/torques for prop i
        var_start = NUM_PROP_VAR * i
        x = x0[var_start]
        y = x0[var_start+1]
        z = x0[var_start+2]
        theta = x0[var_start+3]
        alpha = x0[var_start+4]

        F_x = np.cos(theta) * np.sin(alpha)
        F_y = np.sin(theta) * np.sin(alpha)
        F_z = np.cos(alpha)
        M_x = F_z*y - F_y*z
        M_y = F_x*z - F_z*x
        M_z = F_y*x - F_x*y

        F_vec = np.array([F_x, F_y, F_z, M_x, M_y, M_z])
        pre_G_p_transpose.append(F_vec)
    
    G_p_transpose = np.array(pre_G_p_transpose)
    G_p = np.transpose(G_p_transpose)
    return G_p


def main():
    NUM_VAR = NUM_PROP_VAR * N
    # Initial guess needs to be of the position/orientation vars of the propellers
    # Orientation is specified by angles theta and alpha (in radians)
    # From that guess we can derive the value of G_p
    # TODO: choose a different/better initial guess
    x0 = [0] * NUM_VAR

    # CONSTRAINTS
    # positions of a prop have to satisfy x^2/a^2 + y^2/b^2 + z^2/c^2 = 1
    # TODO: ask Francesco for reasonable values of a,b,c
    a = 1
    b = 1
    c = 1

    def ellipsoid_vals(x):
        def ellipsoid_lhs(x, y, z):
            return (x**2)/(a**2) + (y**2)/(b**2) + (z**2)/(c**2)

        vals = []
        for i in range(N):
            var_start = NUM_PROP_VAR * i 
            x_i = x[var_start]
            y_i = x[var_start+1]
            z_i = x[var_start+2]
            vals.append(ellipsoid_lhs(x_i, y_i, z_i))
        return vals

    lb = [1] * N
    ub = [1] * N
    position_constraint = NonlinearConstraint(ellipsoid_vals, lb, ub) 

    lower_bounds = [-a, -b, -c, 0, 0] * N
    upper_bounds = [a, b, c, 2*pi, pi] * N
    bounds = Bounds(lower_bounds, upper_bounds) 

    res = minimize(objective_function, x0, constraints=position_constraint, bounds=bounds, options={'disp': True})
    print(res)

if __name__ == '__main__':
    main()