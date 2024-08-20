import cvxpy as cp
import numpy as np
import copy

import scipy as sp
from scipy import sparse

import matplotlib.pyplot as plt
from scipy.linalg import hankel

from Models.rl_model_own_cartpole import CustomContinuousCartPoleEnv
# import gymnasium as gym



# DeePC controller hyperparameters

# Set Hankel dimensions
T_past = 2
T_fut = 10
# T_past = 1
# T_fut = 20

n = 4 # state dim
m = 1 # num inputs
# p = 2 # num outputs
# obs_indeces = [1,3] #doc of observation indeces: https://www.gymlibrary.dev/environments/mujoco/inverted_pendulum/
obs_indeces = [] # if [], all observations are used
if obs_indeces != []:
    assert(len(obs_indeces)==p)

# T = (m+1)*(T_past + T_fut + n) + 10
T = 100

r = 1 # input cost weight
# r = 100 # input cost weight
q = 100 # state error cost weight

lam_sigma = 0.1 # weight of regularization on sigma
lam_g = 0.1 # weight of standard-regularization on g
lam_g_PI = 0.1 # weight of projection-regularization on g

y_constraints = np.array([1.5, 0.15, 0.5, 0.5])
u_constraints = np.array([3])
# y_constraints = None
# u_constraints = None

solver = cp.MOSEK # this solver requires some effort to install (see above), but works better than ECOS
# solver = None # default solver (e.g. ECOS) can work, but does not work as consistently as cp.MOSEK



T_sim = 250

def block_hankel(w: np.array, L: int) -> np.array:
    """
    Builds block Hankel matrix of order L from data w
    args:
        w : a T_k x d data matrix. T_k is the number of timesteps, d is the dimension of the signal
          e.g., if there are 6 timesteps and 4 entries at each timestep, w is 6 x 4
        L : order of hankel matrix
    """
    T_k = int(np.shape(w)[0]) # number of timesteps
    d = int(np.shape(w)[1]) # dimension of the signal
    if L > T_k:
        raise ValueError(f'L {L} must be smaller than T_k {T_k}')
    H = np.zeros((L*d, T_k-L+1))
    w_vec = w.reshape(-1)
    for i in range(0, T_k-L+1):
        H[:,i] = w_vec[d*i:d*(L+i)]
    return H


def mosaic_block_hankel(w: dict, L: int) -> np.array:
    """
    Builds block Hankel matrix of order L from data w from the mosaic blocks of w
    Mosaic Hankel def given in Fig. 6 of https://imarkovs.github.io/tutorial.pdf
    args:
        w : a dictionary of T x d data matrices (lists of lists). T is the number of timesteps, d is the dimension of the signal
          e.g., if there are 6 timesteps and 4 entries at each timestep, w is 6 x 4
          a dictionary is used because the T for each episode can be different
        L : order of hankel matrix
    """
    print("moasic being used!!!!!!!!!")
    d = int(np.shape(w[0])[1]) # dimension of the signal
    H = np.zeros((L*d, 0))

    for k in w: #keys of the dictionary are integers that give the index of the episodes
        T_k = int(np.shape(w[k])[0]) # number of timesteps
        if L > T:
            print(f'!!!!! WARNING: L {L} must be smaller than T {T_k}, not using episode {k} to build the mosaic Hankel matrix !!!!!')
        else:
            H_k = np.zeros((L*d, T-L+1))
            w_k_vec = w[k].reshape(-1)
            for i in range(0, T_k-L+1):
                H_k[:,i] = w_k_vec[d*i:d*(L+i)]
            H = np.hstack((H,H_k))

    return H



class DeePCcontrollerClass:
    def __init__(self, u_data, y_data, T_past, T_fut, r, q, lam_g, lam_sigma, lam_g_PI=None, u_constraints=None, y_constraints=None, solver=None):
        """
        args:
            u_data, y_data : can each be either
                - a T x d data matrix (np.arrays). T is the number of timesteps, d is the dimension of the input or output
                - Dictionary of T_k x d data matrices. T is the number of timesteps, d is the dimension of the input or output
        """
        if solver is None:
            self.solver = cp.ECOS
        else:
            self.solver = solver

        self.T_past = T_past
        self.T_fut = T_fut
        self.L = self.T_past + self.T_fut

        if type(u_data) is dict:
            if type(y_data) is not dict:
                raise Exception(f'y_data should be a dict like u_data, but instead it is a {type(y_data)}')
            self.H_u = mosaic_block_hankel(u_data, self.L)
            self.H_y = mosaic_block_hankel(y_data, self.L)
            self.m = np.shape(u_data[0])[1]
            self.p = np.shape(y_data[0])[1]
        else:
            self.H_u = block_hankel(u_data, self.L)
            self.H_y = block_hankel(y_data, self.L)
            self.m = np.shape(u_data)[1]
            self.p = np.shape(y_data)[1]

        self.r = r # input cost weight
        self.q = q # output ~
        self.Q = np.eye(T_fut*self.p)*q
        self.R = np.eye(T_fut*self.m)*r
        self.lam_g = lam_g # regularization weight on g
        self.lam_sigma = lam_sigma # regularization weight on sigma

        self.u_constraints = u_constraints
        self.y_constraints = y_constraints

        # PI = np.vstack([self.H_u[:m*T_past], self.H_y[:p*T_past], self.H_u[:m*T_fut]])
        PI = np.vstack((self.H_u, self.H_y[:p*T_past]))
        PI = np.linalg.pinv(PI)@PI
        I = np.eye(PI.shape[0])
        self.PI = I - PI
        self.lam_g_PI = lam_g_PI


    def step(self, u_past, y_past, y_ref):
        """
        args:
            u_past, y_past, Y_ref : T x d data matrices. T is the number of timesteps, d is the dimension of the input or output
        Returns the first action of input sequence.
        """
        verbose=False

        # set up the DeePC optimization in CVX
        # Helpful CVX functions: cp.Variable(), cp.quad_form()

        assert(np.shape(self.H_u)[0] == self.L*self.m)
        assert(np.shape(self.H_y)[0] == self.L*self.p)
        g = cp.Variable(np.shape(self.H_u)[1])
        u_fut = cp.Variable(self.T_fut*self.m)
        y_fut = cp.Variable(self.T_fut*self.p)

        # put data in vector form
        u_past = u_past.reshape(-1)
        y_past = y_past.reshape(-1)
        y_ref = y_ref.reshape(-1)

        ####################################################################################################
        ################################### Implement DeePC Controller Here ################################
        ####################################################################################################

        # Define the cost function (tracking error + control effort)
        cost = cp.quad_form(y_fut - y_ref, self.Q) + cp.quad_form(u_fut, self.R)

        # Constraints
        constraints = [
            self.H_u @ g == cp.hstack([u_past, u_fut]),  # Past and future inputs consistency
            self.H_y @ g == cp.hstack([y_past, y_fut]),  # Past and future outputs consistency
        ]

        # Add constraints for input limits if provided
        if self.u_constraints is not None:
            # Assuming u_constraints is a vector of length m (input dimensions)
            # Repeat it for each future time step
            expanded_u_constraints = np.tile(self.u_constraints, self.T_fut)
            constraints += [cp.abs(u_fut) <= expanded_u_constraints]

        # Add constraints for output limits if provided
        if self.y_constraints is not None:
            # Assuming y_constraints is a vector of length p (output dimensions)
            # Repeat it for each future time step
            expanded_y_constraints = np.tile(self.y_constraints, self.T_fut)
            constraints += [cp.abs(y_fut) <= expanded_y_constraints]

        # Add regularization if needed
        if self.lam_g is not None:
            cost += self.lam_g * cp.norm(g, 1)

        if self.lam_sigma is not None:
            sigma_y = cp.Variable(self.T_fut * self.p)
            constraints += [y_fut - sigma_y == y_ref]
            cost += self.lam_sigma * cp.norm(sigma_y, 1)

        # If projection regularization is specified
        if self.lam_g_PI is not None:
            cost += self.lam_g_PI * cp.norm(self.PI @ g, 1)

        ####################################################################################################


        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=self.solver, verbose=True)

        if prob.status != 'optimal':
            raise Exception(f'cvx had an issue with the optimization. got prob.status: {prob.status}')

        u_fut_val = u_fut.value
        y_fut_val = y_fut.value
        g_val = g.value
        if self.lam_sigma is None:
            sigma_val = None
        else:
            sigma_val = sigma_y.value

        return u_fut_val, y_fut_val, g_val, sigma_val





# Fix random seed for reproducible results
np.random.seed(1)

env = CustomContinuousCartPoleEnv()

# obs, info = env.reset()
obs, _ = env.reset()

if obs_indeces == []:
    obs_indeces = np.arange(len(obs))
    p = len(obs_indeces)

ud = np.empty((0,m))
yd = np.empty((0,p))
ud_dict = {}
yd_dict = {}
episode = 0
for i in np.arange(T):
    # if system goes out of quasi-linear regime
    if np.abs(obs[1]) > 0.1:
        # obs, info = env.reset()
        obs, _ = env.reset()
        print('!!!!! WARNING: data went out of linear regime, env reset and new episode started. !!!!!')
        ud_dict[episode] = ud
        yd_dict[episode] = yd
        episode += 1
        ud = np.empty((0,m))
        yd = np.empty((0,p))

    #feedback controller to keep the data in the linear regime
    action = [obs[0]*(1) + obs[1]*10 + obs[3]*2]

    # add white noise for exploration
    action = float(action[0] - np.random.randn()*0.04)

    obs, rew, done, info, _ = env.step(action)

    if len(obs_indeces) == 0:
        obs_selected = obs
    else:
        obs_selected = obs[obs_indeces]

    # if includeMeasNoise:
    #     obs += np.random.randn(p,)*measNoiseVariance
    ud = np.vstack((ud,action))
    yd = np.vstack((yd,obs_selected))


ud_dict[episode] = ud
yd_dict[episode] = yd






DeePCcontroller = DeePCcontrollerClass(ud_dict, yd_dict, T_past, T_fut, r, q, lam_g, lam_sigma, lam_g_PI, u_constraints, y_constraints, solver)



foundAnInitialization = False
i = 0
while not foundAnInitialization:
    T_i = len(ud_dict[i])
    if T_i > DeePCcontroller.T_past:
        u_past = ud_dict[i][:T_past]
        y_past = yd_dict[i][:T_past]
        foundAnInitialization = True
    else:
        i += 1
if foundAnInitialization == False:
    raise Exception('Could not find find an episode that is T_past-long, so DeePC could not be initialized. If you would like, change this exception to code that sets the initial values to zero.')

y_sim = copy.deepcopy(y_past)
u_sim = copy.deepcopy(u_past)

y_ref = np.zeros((T_fut,p))
obs = env.reset()
step = 0
while step < T_sim:

    ####################################################################################################
    ################################### Deploy the DeePC Controller Here ###############################
    ####################################################################################################
    # action =
    u_fut_val, y_fut_val, g_val, sigma_val = DeePCcontroller.step(u_past, y_past, y_ref)
    action = u_fut_val[0]

    obs, _, done, _, _ = env.step(action)
    ####################################################################################################

    obs,_,done,_,_ = env.step(action)
    includeMeasNoise = False
    measNoiseVariance = 0.01
    if includeMeasNoise:
            obs += np.random.randn(p,)*measNoiseVariance
    if len(obs_indeces) == 0:
        obs_selected = obs
    else:
        obs_selected = obs[obs_indeces]
    u_sim = np.vstack((u_sim,action))
    y_sim = np.vstack((y_sim,obs_selected))

    if np.abs(obs[1]) > 0.4 or np.abs(obs[0]) > 0.4 or np.abs(obs[2]) > 0.4 or np.abs(obs[3]) > 0.4 :
        raise Exception('constraints violated')

    u_past = u_sim[-T_past:]
    y_past = y_sim[-T_past:]
    step += 1


T_sim_withInit = len(u_sim)

plt.subplot(2,1,1)
plt.plot(range(0,T_sim_withInit), y_sim, label=['x-pos (m)', 'angle (rad)', 'x-vel (m/s)', 'ang-vel (rad/s)'])
plt.legend(loc="upper left")
plt.xlabel('Time (s)', fontdict={'fontsize':15})
plt.title("State Evolution", fontdict={'fontsize':20})
plt.grid('on')
plt.tight_layout()

plt.subplot(2,1,2)
plt.plot(range(0,T_sim_withInit), u_sim)
plt.ylabel('Force (N)', fontdict={'fontsize':15})
plt.xlabel('Time (s)', fontdict={'fontsize':15})
plt.title("Actuation", fontdict={'fontsize':20})
plt.grid('on')
plt.tight_layout()

plt.show()

print(f'y_constraints: {y_constraints}')