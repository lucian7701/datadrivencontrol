import numpy as np
from Models.deepc_system import SystemBase
from scipy.integrate import solve_ivp


class NonLinearSystem(SystemBase):

    def __init__(self, x0: np.array, ode, dt, m, output_function=None, measurement_noise_std=0):
        super().__init__(x0, m=m, measurement_noise_std=measurement_noise_std)
        self.output_function = output_function
        self.ode = ode
        self.dt = dt

    def step(self, u: np.ndarray):
        T = len(u)
        t_span = (0, self.dt)
        y_list = []
        x_list = []

        for i in range(T):
        
            y = self.output_function(self.x0) \
                if self.output_function is not None else self.x0
            y_list.append(y)
            x_list.append(self.x0)
            
            solution = solve_ivp(self.ode, t_span, self.x0, args=(u[i],), method='RK45', t_eval=[self.dt])
            self.x0 = solution.y[:, -1]

        return np.array(y_list), np.array(x_list)
    