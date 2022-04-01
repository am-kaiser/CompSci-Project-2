from dataclasses import dataclass


@dataclass
class Parameters:
    dt: float = 600  # time step size
    t_start: float = 0.0  # start time
    t_end_h: float = 1  # end time in hours
    t_end_s: float = t_end_h * 3600  # end time in seconds
    num_time_steps: float = int((t_end_s - t_start) / dt) + 1  # number of time steps

    dx: float = 10  # x step size
    x_start: float = 0.0  # start of domain
    L: float = 1  # rod length
    x_end_m: float = L  # end of domain in meters
    x_end_cm: float = L * 100  # end of domain in centimeters
    num_x_steps: float = int((x_end_cm - x_start) / dx) + 1  # number of steps in space
