from dataclasses import dataclass


@dataclass
class Parameters:
    dx: float = 1/10  # x step size
    x_start: float = 0.0  # start of domain
    L: float = 1  # rod length
    x_end: float = L  # end of domain in meters
    num_x_steps: float = int((x_end - x_start) / dx)  # number of steps in space

    dt: float = round(0.5 * (dx ** 2) * 0.5, 6)  # time step size
    t_start: float = 0.0  # start time
    t_end: float = 1.0  # end time in hours
    num_time_steps: float = int((t_end - t_start) / dt)  # number of time steps
