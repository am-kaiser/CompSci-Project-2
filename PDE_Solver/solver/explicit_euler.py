"""Script to perform explicit Euler algorithm."""
from PDE_Solver.solver import euler_parameters
from PDE_Solver.utils import plot_solution

import numpy as np
import matplotlib.pyplot as plt


def initialize_output(input_params):
    output = np.empty([input_params.num_x_steps, input_params.num_time_steps])
    output[:] = np.nan
    return output


def create_grid(input_params):
    x_values = np.linspace(input_params.x_start, input_params.x_end, num=input_params.num_x_steps)
    t_values = np.linspace(input_params.t_start, input_params.t_end, num=input_params.num_time_steps)

    return x_values, t_values


def define_initial_conditions(u_values):
    u_values[0, :] = 0.0
    u_values[-1, :] = 0.0
    return u_values


def define_boundary_conditions(u_values, x_values):
    u_values[:, 0] = np.sin(np.pi * x_values)
    return u_values


def perform_euler_algorithm(u_values, input_params):
    alpha = input_params.dt / (input_params.dx ** 2)
    t_indices = range(input_params.num_time_steps + 1)  # r range does not include right boundary. i.e. range(1)=0
    x_indices = range(input_params.num_x_steps + 1)

    for t_index in t_indices[0:-2]:
        for x_index in x_indices[1:-2]:
            u_values[x_index, t_index + 1] = alpha * u_values[x_index + 1, t_index] + \
                                             (1 - 2 * alpha) * u_values[x_index, t_index] + \
                                             alpha * u_values[x_index - 1, t_index]

    return u_values


def explicit_solution(u_values_exp, x_values, t_values, input_params):
    t_indices = range(input_params.num_time_steps)  # range does not include right boundary. i.e. range(1)=0
    x_indices = range(input_params.num_x_steps)

    for t_index in t_indices:
        for x_index in x_indices:
            u_values_exp[x_index, t_index] = np.sin(np.pi * x_values[x_index]) * np.exp(
                -t_values[t_index] * (np.pi ** 2))

    return u_values_exp


def solve_pde_euler():
    # Load parameters
    params = euler_parameters.Parameters()

    # Initialize output array
    u_euler = initialize_output(params)
    u_explicit = initialize_output(params)

    # Create grid
    x, t = create_grid(params)

    # EULER ALGORITHM
    # Apply initial conditions
    u_euler = define_initial_conditions(u_euler)
    # Apply boundary conditions
    u_euler = define_boundary_conditions(u_euler, x)
    # Perform Euler algorithm
    u_euler = perform_euler_algorithm(u_euler, params)

    # EXPLICIT SOLUTION
    u_explicit = explicit_solution(u_explicit, x, t, params)

    # ERROR
    error = abs(u_euler - u_explicit)

    # Plot solution
    plot_solution.plot_3d_solution(t, x, u_euler, 't', 'x [m}', 'EULER: u')
    plot_solution.plot_3d_solution(t, x, u_explicit, 't', 'x [m]', 'ANALYTIC: u')
    plot_solution.plot_3d_solution(t, x, error, 't', 'x [m]', 'ERROR')
    plt.show()


if __name__ == "__main__":
    solve_pde_euler()
