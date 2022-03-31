from dataclasses import dataclass


@dataclass
class Parameters:

    dt: float = 0.1  # time step size
    t_start: float = 0.0  # start time
    t_end: float = 10  # end time

    dx: float = 0.1  # x step size
    x_start: float = 0.0  # start of domain
    L: float = 1  # rod length
    x_end: float = L  # end of domain