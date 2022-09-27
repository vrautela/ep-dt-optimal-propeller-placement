import numpy as np
from math import pi, sqrt
from scipy.optimize import Bounds, minimize, NonlinearConstraint

from optimization.consts import a, b, c


# min and max components of forces and torques expected from system
min_fx, max_fx = -1, 2
min_fy, max_fy = -1, 1
min_fz, max_fz = -1, 1
min_tx, max_tx = -0.4, 0.4
min_ty, max_ty = -1.2, 1.2
min_tz, max_tz = -1.6, 1.6

# generate a list of desired forces and torques by uniformly sampling from the min/max force and torques
def uniform_desired_forces():
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
    return F_des

# generate a list of desired force and torque vectors by sampling points along the ellipsoid defined by the min/max force and torque components 
def ellipsoid_desired_forces():
    F_des = []
    a, b, c = (max_fx - min_fx)/2, (max_fy - min_fy)/2, (max_fz - min_fz)/2, 
    x0, y0, z0 = (max_fx + min_fx)/2, (max_fy + min_fy)/2, (max_fz + min_fz)/2

    a_t, b_t, c_t = (max_tx - min_tx)/2, (max_ty - min_ty)/2, (max_tz - min_tz)/2
    x0_t, y0_t, z0_t = (max_tx + min_tx)/2, (max_ty + min_ty)/2, (max_tz + min_tz)/2

    thetas = np.linspace(0, np.pi, 9)
    phis = np.linspace(0, 2*np.pi, 18)

    for theta in thetas:
        for phi in phis:
            fx = x0 + a*np.sin(theta)*np.cos(phi)
            fy = y0 + b*np.sin(theta)*np.sin(phi)
            fz = z0 + c*np.cos(theta)

            f = np.array([fx, fy, fz, 0, 0, 0])
            F_des.append(f)

            tx = x0_t + a_t*np.sin(theta)*np.cos(phi)
            ty = y0_t + b_t*np.sin(theta)*np.sin(phi)
            tz = z0_t + c_t*np.cos(theta)

            t = np.array([0, 0, 0, tx, ty, tz])
            F_des.append(t)
    return F_des

# list of desired force and torque vectors
F_des = ellipsoid_desired_forces()

# number of propellers
N = 8

# number of variables specifying the position/orientation of a propeller
NUM_PROP_VAR = 5

# number of times to run the optimizer
NUM_OPTIMIZATION_ITERS = 5


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
        # NOTE: there is a suboptimization problem here (we want to find T such that residual is minimized)
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


# Compute the G_p matrix given a propeller configuration x0 which 
# represents a list of (x, y, z, theta, alpha) for each propeller
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

        # compute the matrix elements of the ith row
        #   NOTE: The matrix elements of G_p in Indoor Blimp Control (Aman, 2021) are WRONG
        #         and they should be as computed below
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
    # generate a random list of propeller configurations
    def generate_initial_guess():
        def generate_random_propeller_configuration():
            # choose x coordinate randomly from (-a, a)
            x_i = np.random.uniform(low=-a, high=a)

            # choose y coordinate randomly (based on the bounds set by x coordinate) 
            y_bounds = b * sqrt(1 - (x_i**2)/(a**2))
            y_i = np.random.uniform(low=-y_bounds, high=y_bounds)

            # choose z coordinate based on the bounds set by x and y coordinates
            t = c * sqrt(1 - (x_i**2)/(a**2) - (y_i**2)/(b**2))
            possible_zs = [-t, t]
            z_i = np.random.choice(possible_zs) 

            # choose theta and alpha randomly from [0, 2pi] and [0, pi]
            theta_i = np.random.uniform(low=0, high=2*pi)
            alpha_i = np.random.uniform(low=0, high=pi)

            return [x_i, y_i, z_i, theta_i, alpha_i]

        # initial guess containing N random propeller configurations
        x0 = []
        for _ in range(N):
            x0.extend(generate_random_propeller_configuration())
        return x0

    # compute x^2/a^2 + y^2/b^2 + z^2/c^2 of each position in the solution vector, x
    def ellipsoid_vals(x):
        # compute x^2/a^2 + y^2/b^2 + z^2/c^2 for a given x, y, z
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

    # CONSTRAINTS
    # positions of a prop have to satisfy x^2/a^2 + y^2/b^2 + z^2/c^2 = 1
    lb = [1] * N
    ub = [1] * N
    position_constraint = NonlinearConstraint(ellipsoid_vals, lb, ub) 

    # TODO: try removing these bounds (since they are encompassed by the constraint)
    lower_bounds = [-a, -b, -c, 0, 0] * N
    upper_bounds = [a, b, c, 2*pi, pi] * N
    bounds = Bounds(lower_bounds, upper_bounds) 

    # Initial guess needs to be of the position/orientation vars of the propellers
    # Orientation is specified by angles theta and alpha (in radians)
    # From that guess we can derive the value of G_p
    x0 = generate_initial_guess()
    print(f'x0: {x0}')

    best_solution = minimize(objective_function, x0, constraints=position_constraint, bounds=bounds, options={'disp': True})
    print(best_solution)
    
    # the solution with the minimum objective function value (so far)
    current_best_solution = best_solution
    # the value of the objective function for the current best solution 
    min_obj = best_solution.fun

    # repeat the optimization procedure with different initial guesses for a total of 
    # NUM_OPTIMIZATION_ITERS and return the solution with the lowest objective function
    for _ in range(NUM_OPTIMIZATION_ITERS - 1):
        x0 = generate_initial_guess()
        print(f'x0: {x0}')

        best_solution = minimize(objective_function, x0, constraints=position_constraint, bounds=bounds, options={'disp': True})
        print(best_solution)

        # if the current value of the objective function of this solution is less than 
        # the minimum one, then replace the minimum objective function value and change
        # the current best solution to this one
        curr_obj = best_solution.fun
        if curr_obj < min_obj:
            min_obj = curr_obj
            current_best_solution = best_solution
    
    print('----------------------------------------------')
    print(f'Best solution found after {NUM_OPTIMIZATION_ITERS} iterations')
    print(current_best_solution)
    print(f'Objective function value: {min_obj}')


if __name__ == '__main__':
    main()