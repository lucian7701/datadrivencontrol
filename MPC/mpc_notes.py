import numpy as np
import scipy as sp
import scipy.sparse as sparse
import osqp
import warnings


def __is_vector__(vec):
    if vec.ndim == 1:
        return True
    else:
        if vec.ndim == 2:
            if vec.shape[0] == 1 or vec.shape[1] == 0:
                return True
        else:
            return False
        return False


def __is_matrix__(mat):
    if mat.ndim == 2:
        return True
    else:
        return False


class MPCController:
    """

    Attributes
    Ad : 2D array_like. Size: (nx, nx)
         Discrete-time system matrix Ad.
    Bd : 2D array-like. Size: (nx, nu)
         Discrete-time system matrix Bd.
    Np : int
        Prediction horizon. Default value: 20.
    Nc : int
        Control horizon. It must be lower or equal to Np. If None, it is set equal to Np.
    x0 : 1D array_like. Size: (nx,)
         System state at time instant 0. If None, it is set to np.zeros(nx)
    uref : 1D array-like. Size: (nu, )
           System input reference. If None, it is set to np.zeros(nx)
    uminus1 : 1D array_like
             Input value assumed at time instant -1. If None, it is set to uref.
    Qx : 2D array_like
         State weight matrix. If None, it is set to eye(nx).
    QxN : 2D array_like
         State weight matrix for the last state. If None, it is set to eye(nx).
    Qu : 2D array_like
         Input weight matrix. If None, it is set to zeros((nu,nu)).
    QDu : 2D array_like
         Input delta weight matrix. If None, it is set to zeros((nu,nu)).
    xmin : 1D array_like
           State minimum value. If None, it is set to -np.inf*ones(nx).
    xmax : 1D array_like
           State maximum value. If None, it is set to np.inf*ones(nx).
    umin : 1D array_like
           Input minimum value. If None, it is set to -np.inf*ones(nx).
    umax : 1D array_like
           Input maximum value. If None, it is set to np.inf*ones(nx).
    Dumin : 1D array_like
           Input variation minimum value. If None, it is set to np.inf*ones(nx).
    Dumax : 1D array_like
           Input variation maximum value. If None, it is set to np.inf*ones(nx).
    eps_feas : float
               Scale factor for the matrix Q_eps. Q_eps = eps_feas*eye(nx).
    eps_rel : float
              Relative tolerance of the QP solver. Default value: 1e-3.
    eps_abs : float
              Absolute tolerance of the QP solver. Default value: 1e-3.
    """

    def __init__(self, Ad, Bd, Np=20, Nc=None,
                 x0=None, xref=None, uref=None, uminus1=None,
                 Qx=None, QxN=None, Qu=None, QDu=None,
                 xmin=None, xmax=None, umin=None, umax=None, Dumin=None, Dumax=None,
                 eps_feas=1e6, eps_rel=1e-3, eps_abs=1e-3):

        # Checks

        # State matrix checks
        if __is_matrix__(Ad) and (Ad.shape[0] == Ad.shape[1]):
            self.Ad = Ad
            self.nx = Ad.shape[0]  # number of states
        else:
            raise ValueError("Ad should be a square matrix of dimension (nx,nx)!")

        # Control matrix check
        if __is_matrix__(Bd) and Bd.shape[0] == self.nx:
            self.Bd = Bd
            self.nu = Bd.shape[1]  # number of inputs
        else:
            raise ValueError("Bd should be a matrix of dimension (nx, nu)!")

        # Prediction horizon check
        if Np > 1:
            self.Np = Np  # assert
        else:
            raise ValueError("Np should be > 1!")

        # Control horizon check
        if Nc is not None:
            if Nc <= Np:
                self.Nc = Nc
            else:
                raise ValueError("Nc should be <= Np!")
        else:
            self.Nc = self.Np

        # x0 check
        if x0 is not None:
            if __is_vector__(x0) and x0.size == self.nx:
                self.x0 = x0.ravel()
            else:
                raise ValueError("x0 should be an array of dimension (nx,)!")
        else:
            self.x0 = np.zeros(self.nx)

        # state reference check
        if xref is not None:
            if __is_vector__(xref) and xref.size == self.nx:
                self.xref = xref.ravel()
            elif __is_matrix__(xref) and xref.shape[1] == self.nx and xref.shape[0] >= self.Np:
                self.xref = xref
            else:
                raise ValueError("xref should be either a vector of shape (nx,) or a matrix of shape (Np+1, nx)!")
        else:
            self.xref = np.zeros(self.nx)

        # control reference check
        if uref is not None:
            if __is_vector__(uref) and uref.size == self.nu:
                self.uref = uref.ravel()  # assert...
            else:
                raise ValueError("uref should be a vector of shape (nu,)!")
        else:
            # This will be the majority of the case - i.e. want to just keep control as low as possible
            self.uref = np.zeros(self.nu)

        # u minus1/ previous input check
        if uminus1 is not None:
            if __is_vector__(uminus1) and uminus1.size == self.nu:
                self.uminus1 = uminus1
            else:
                raise ValueError("uminus1 should be a vector of shape (nu,)!")
        else:
            self.uminus1 = self.uref

        # Weights handling

        # State cost check
        if Qx is not None:
            if __is_matrix__(Qx) and Qx.shape[0] == self.nx and Qx.shape[1] == self.nx:
                self.Qx = Qx
            else:
                raise ValueError("Qx should be a matrix of shape (nx, nx)!")
        else:
            self.Qx = np.zeros((self.nx, self.nx))  # sparse

        # Terminal state cost check
        if QxN is not None:
            if __is_matrix__(QxN) and QxN.shape[0] == self.nx and Qx.shape[1] == self.nx:
                self.QxN = QxN
            else:
                raise ValueError("QxN should be a square matrix of shape (nx, nx)!")
        else:
            self.QxN = self.Qx  # sparse

        # Control cost check
        if Qu is not None:
            if __is_matrix__(Qu) and Qu.shape[0] == self.nu and Qu.shape[1] == self.nu:
                self.Qu = Qu
            else:
                raise ValueError("Qu should be a square matrix of shape (nu, nu)!")
        else:
            self.Qu = np.zeros((self.nu, self.nu))

        # Change of control cost check
        if QDu is not None:
            if __is_matrix__(QDu) and QDu.shape[0] == self.nu and QDu.shape[1] == self.nu:
                self.QDu = QDu
            else:
                raise ValueError("QDu should be a square matrix of shape (nu, nu)!")
        else:
            self.QDu = np.zeros((self.nu, self.nu))

        # Constraint handling

        # xmin
        if xmin is not None:
            if __is_vector__(xmin) and xmin.size == self.nx:
                self.xmin = xmin.ravel()
            else:
                raise ValueError("xmin should be a vector of shape (nx,)!")
        else:
            self.xmin = -np.ones(self.nx) * np.inf

        # xmax
        if xmax is not None:
            if __is_vector__(xmax) and xmax.size == self.nx:
                self.xmax = xmax
            else:
                raise ValueError("xmax should be a vector of shape (nx,)!")
        else:
            self.xmax = np.ones(self.nx) * np.inf

        # umin
        if umin is not None:
            if __is_vector__(umin) and umin.size == self.nu:
                self.umin = umin
            else:
                raise ValueError("umin should be a vector of shape (nu,)!")
        else:
            self.umin = -np.ones(self.nu) * np.inf

        # umax
        if umax is not None:
            if __is_vector__(umax) and umax.size == self.nu:
                self.umax = umax
            else:
                raise ValueError("umax should be a vector of shape (nu,)!")
        else:
            self.umax = np.ones(self.nu) * np.inf

        # Dumin
        if Dumin is not None:
            if __is_vector__(Dumin) and Dumin.size == self.nu:
                self.Dumin = Dumin
            else:
                raise ValueError("Dumin should be a vector of shape (nu,)!")
        else:
            self.Dumin = -np.ones(self.nu) * np.inf

        # Dumax
        if Dumax is not None:
            if __is_vector__(Dumax) and Dumax.size == self.nu:
                self.Dumax = Dumax
            else:
                raise ValueError("Dumax should be a vector of shape (nu,)!")
        else:
            self.Dumax = np.ones(self.nu) * np.inf

        # Scale the penalty for slack variables
        self.eps_feas = eps_feas
        # Diagonal penalty matrix for slack variables
        self.Qeps = eps_feas * sparse.eye(self.nx)

        self.eps_rel = eps_rel  # relative tolerance of QP solver
        self.eps_abs = eps_abs  # absolute tolerance of QP solver
        self.u_failure = self.uref  # value provided when the MPC solver fails.

        # Hidden settings (for debug purpose)
        self.raise_error = False  # Raise an error when MPC optimization fails
        self.JX_ON = True  # Cost function terms in X active
        self.JU_ON = True  # Cost function terms in U active
        self.JDU_ON = False  # Cost function terms in Delta U active
        self.SOFT_ON = False  # Soft constraints active
        self.COMPUTE_J_CNST = False  # Compute the constant term of the MPC QP problem

        # QP problem instance
        self.prob = osqp.OSQP()

        # Variables initialized by the setup() method
        self.res = None  # result
        self.P = None
        self.q = None
        self.A = None
        self.l = None
        self.u = None
        self.x0_rh = None
        self.uminus1_rh = None
        self.J_CNST = None  # Constant term of the cost function

    def setup(self, solve=True):
        """ Set up the QP problem.

            Parameters
            ----------
            solve : bool
                   If True, also solve the QP problem.

        """

        self.x0_rh = np.copy(self.x0)  # receding horizon initial state
        self.uminus1_rh = np.copy(self.uminus1)  # previous input value
        self._compute_QP_matrices_()
        self.prob.setup(self.P, self.q, self.A, self.l, self.u, warm_start=True, verbose=False, eps_abs=self.eps_rel,
                        eps_rel=self.eps_abs)

        if solve:
            self.solve()

    def _compute_QP_matrices_(self):
        Np = self.Np
        Nc = self.Nc
        nx = self.nx
        nu = self.nu
        Qx = self.Qx
        QxN = self.QxN
        Qu = self.Qu
        QDu = self.QDu
        xref = self.xref
        uref = self.uref
        uminus1 = self.uminus1
        Ad = self.Ad
        Bd = self.Bd
        x0 = self.x0
        xmin = self.xmin
        xmax = self.xmax
        umin = self.umin
        umax = self.umax
        Dumin = self.Dumin
        Dumax = self.Dumax
        Qeps = self.Qeps

        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
        # - quadratic objective

        # STATE COST

        # start by initialising a very large matrix for overall quadratic cost
        P_X = sparse.csc_matrix(((Np + 1) * nx, (Np + 1) * nx))  # Quadratic terms

        # initialise a matrix for overall linear cost
        q_X = np.zeros((Np + 1) * nx)  # Linear terms

        # Zero cost constant
        self.J_CNST = 0.0  # constant terms

        if self.JX_ON:
            P_X += sparse.block_diag([sparse.kron(sparse.eye(Np), Qx),  # x0...x_N-1
                                      QxN])  # xN
            # P_X has all the state cost matrices for every time step in prediction horizon laid out

            # if xref is a matrix of changing reference inputs
            if xref.ndim == 2 and xref.shape[0] >= Np + 1:
                q_X += (-xref.reshape(1, -1) @ (P_X)).ravel()

                if self.COMPUTE_J_CNST:
                    # This is not necessary as constant terms make no difference to the optimisation.
                    self.J_CNST += -1 / 2 * q_X @ xref.ravel()
            # if xref is a constant vector
            else:
                q_X += np.hstack([np.kron(np.ones(Np), -Qx.dot(xref)),  # x0... x_N-1
                                  -QxN.dot(xref)])  # x_N
                if self.COMPUTE_J_CNST:
                    self.J_CNST += 1 / 2 * Np * (xref.dot(QxN.dot(xref))) + 1 / 2 * xref.dot(QxN.dot(xref))
        else:
            pass

        # CONTROL COST

        P_U = sparse.csc_matrix((Nc * nu, Nc * nu))
        q_U = np.zeros(Nc * nu)

        if self.JU_ON:
            # Constant costs
            self.J_CNST += 1 / 2 * Np * (uref.dot(Qu.dot(uref)))

            iU = np.ones(Nc)
            iU[Nc - 1] = (Np - Nc + 1)
            P_U += sparse.kron(sparse.diags(iU), Qu)
            q_U += np.kron(iU, -Qu.dot(uref))

        # Filling P and q for J_DU
        if self.JDU_ON:
            self.J_CNST += 1 / 2 * uminus1.dot((QDu).dot(uminus1))
            iDu = 2 * np.eye(Nc) - np.eye(Nc, k=1) - np.eye(Nc, k=-1)
            iDu[Nc - 1, Nc - 1] = 1
            P_U += sparse.kron(iDu, QDu)
            q_U += np.hstack([-QDu.dot(uminus1),  # u0
                              np.zeros((Nc - 1) * nu)])  # u1..uN-1
        else:
            pass

        if self.SOFT_ON:
            P_eps = sparse.kron(np.eye((Np + 1)), Qeps)
            q_eps = np.zeros((Np + 1) * nx)

        # LINEAR CONSTRAINTS

        # Linear dynamics x_k+1 = Ax_k + Bu_k

        Ax = sparse.kron(sparse.eye(Np + 1), -sparse.eye(nx)) + sparse.kron(sparse.eye(Np + 1, k=-1), Ad)
        # 1st - block diagonal where each block is -I_nx
        # 2nd - block matrix with Ad matrix on sub-diagonals
        # both matrices same size.

        iBu = sparse.vstack([sparse.csc_matrix((1, Nc)),
                             sparse.eye(Nc)])
        # Here, this stack a matrix of 0s of Nc columns, and identity matrix Nc x Nc
        # Resulting shape is (Nc+1) x Nc

        if self.Nc < self.Np:  # expand A matrix if Nc < Np
            iBu = sparse.vstack([iBu,
                                 sparse.hstack([sparse.csc_matrix((Np - Nc, Nc - 1)), np.ones((Np - Nc, 1))])
                                 ])
        Bu = sparse.kron(iBu, Bd)

        n_eps = (Np + 1) * nx
        Aeq_dyn = sparse.hstack([Ax, Bu])
        if self.SOFT_ON:
            Aeq_dyn = sparse.hstack(
                [Aeq_dyn, sparse.csc_matrix((Aeq_dyn.shape[0], n_eps))])  # For soft constraints slack variables

        leq_dyn = np.hstack([-x0, np.zeros(Np * nx)])
        ueq_dyn = leq_dyn  # for equality constraints -> upper bound  = lower bound!

        # Bounds on x

        Aineq_x = sparse.hstack([sparse.eye((Np + 1) * nx), sparse.csc_matrix(((Np + 1) * nx, Nc * nu))])
        # second part of this hstack represents the absence of constraints on the control inputs in this
        # part of the constraints matrix
        if self.SOFT_ON:
            Aineq_x = sparse.hstack([Aineq_x, sparse.eye(n_eps)])  # For soft constraints slack variables
        lineq_x = np.kron(np.ones(Np + 1), xmin)  # lower bound of inequalities, these are long 1d arrays (row)
        uineq_x = np.kron(np.ones(Np + 1), xmax)  # upper bound of inequalities

        # Bounds on u

        Aineq_u = sparse.hstack([sparse.csc_matrix((Nc * nu, (Np + 1) * nx)), sparse.eye(Nc * nu)])
        if self.SOFT_ON:
            Aineq_u = sparse.hstack(
                [Aineq_u, sparse.csc_matrix((Aineq_u.shape[0], n_eps))])  # For soft constraints slack variables
        lineq_u = np.kron(np.ones(Nc), umin)  # lower bound of inequalities
        uineq_u = np.kron(np.ones(Nc), umax)  # upper bound of inequalities

        # Bounds on \Delta u
        Aineq_du = sparse.vstack(
            [sparse.hstack([np.zeros((nu, (Np + 1) * nx)), sparse.eye(nu), np.zeros((nu, (Nc - 1) * nu))]),
             # for u0 - u-1
             sparse.hstack([np.zeros((Nc * nu, (Np + 1) * nx)), -sparse.eye(Nc * nu) + sparse.eye(Nc * nu, k=1)])
             # for uk - uk-1, k=1...Np
             ]
        )
        if self.SOFT_ON:
            Aineq_du = sparse.hstack([Aineq_du, sparse.csc_matrix((Aineq_du.shape[0], n_eps))])

        uineq_du = np.kron(np.ones(Nc + 1), Dumax)  # np.ones((Nc+1) * nu)*Dumax
        uineq_du[0:nu] += self.uminus1[0:nu]

        lineq_du = np.kron(np.ones(Nc + 1), Dumin)  # np.ones((Nc+1) * nu)*Dumin
        lineq_du[0:nu] += self.uminus1[0:nu]  # works for non-scalar u?

        # OSQP constraints

        A = sparse.vstack([Aeq_dyn, Aineq_x, Aineq_u, Aineq_du]).tocsc()
        l = np.hstack([leq_dyn, lineq_x, lineq_u, lineq_du])
        u = np.hstack([ueq_dyn, uineq_x, uineq_u, uineq_du])

        # assign all
        if self.SOFT_ON:
            self.P = sparse.block_diag([P_X, P_U, P_eps], format='csc')
            self.q = np.hstack([q_X, q_U, q_eps])
        else:
            self.P = sparse.block_diag([P_X, P_U], format='csc')
            self.q = np.hstack([q_X, q_U])

        self.A = A
        self.l = l
        self.u = u

        self.P_X = P_X

    def solve(self):
        """ Solve the QP problem. """

        self.res = self.prob.solve()

        # Check solver status
        if self.res.info.status != 'solved':
            warnings.warn('OSQP did not solve the problem!')
            if self.raise_error:
                raise ValueError('OSQP did not solve the problem!')

    def output(self, return_x_seq=False, return_u_seq=False, return_eps_seq=False, return_status=False,
               return_obj_val=False):
        """ Return the MPC controller output uMPC, i.e., the first element of the optimal input sequence and assign is to self.uminus1_rh.

        Parameters
        ----------
        return_x_seq : bool
                       If True, the method also returns the optimal sequence of states in the info dictionary
        return_u_seq : bool
                       If True, the method also returns the optimal sequence of inputs in the info dictionary
        return_eps_seq : bool
                       If True, the method also returns the optimal sequence of epsilon in the info dictionary
        return_status : bool
                       If True, the method also returns the optimizer status in the info dictionary
        return_obj_val : bool
                       If True, the method also returns the objective function value in the info dictionary

        Returns
        -------
        array_like (nu,)
            The first element of the optimal input sequence uMPC to be applied to the system.
        dict
            A dictionary with additional infos. It is returned only if one of the input flags return_* is set to True
        """

        Nc = self.Nc
        Np = self.Np
        nx = self.nx
        nu = self.nu

        # Extract first control input to the plant
        if self.res.info.status == 'solved':
            uMPC = self.res.x[(Np + 1) * nx:(Np + 1) * nx + nu]  # control inputs are immediately after the states.
        else:
            uMPC = self.u_failure

        # Return additional info?
        info = {}
        if return_x_seq:
            seq_X = self.res.x[0:(Np + 1) * nx]
            seq_X = seq_X.reshape(-1, nx)
            info['x_seq'] = seq_X

        if return_u_seq:
            seq_U = self.res.x[(Np + 1) * nx:(Np + 1) * nx + Nc * nu]
            seq_U = seq_U.reshape(-1, nu)
            info['u_seq'] = seq_U

        if return_eps_seq:
            seq_eps = self.res.x[(Np + 1) * nx + Nc * nu: (Np + 1) * nx + Nc * nu + (Np + 1) * nx]
            seq_eps = seq_eps.reshape(-1, nx)
            info['eps_seq'] = seq_eps

        if return_status:
            info['status'] = self.res.info.status

        if return_obj_val:
            obj_val = self.res.info.obj_val + self.J_CNST  # constant of the objective value
            info['obj_val'] = obj_val

        self.uminus1_rh = uMPC

        if len(info) == 0:
            return uMPC

        else:
            return uMPC, info

    def update(self, x, u=None, xref=None, solve=True):
        """ Update the QP problem.

        Parameters
        ----------
        x : array_like. Size: (nx,)
            The new value of x0.

        u : array_like. Size: (nu,)
            The new value of uminus1. If none, it is set to the previously computed u.

        xref : array_like. Size: (nx,)
            The new value of xref. If none, it is not changed

        solve : bool
               If True, also solve the QP problem.

        """

        self.x0_rh = x  # previous x0
        if u is not None:
            self.uminus1_rh = u  # otherwise it is just the uMPC updated from the previous step() call
        if xref is not None:
            self.xref = xref
        self._update_QP_matrices_()

    def _update_QP_matrices_(self):
        # My understanding is this is all about finding A, l, u, P and q
        x0_rh = self.x0_rh
        uminus1_rh = self.uminus1_rh
        Np = self.Np
        Nc = self.Nc
        nx = self.nx
        nu = self.nu
        Dumin = self.Dumin
        Dumax = self.Dumax
        QDu = self.QDu
        uref = self.uref
        Qeps = self.Qeps
        Qx = self.Qx
        QxN = self.QxN
        Qu = self.Qu
        xref = self.xref
        P_X = self.P_X

        # Update initial state constraint
        self.l[:nx] = -x0_rh    # change that first equality constraint.
        self.u[:nx] = -x0_rh    # ""

        # update DU constraints
        self.l[
        (Np + 1) * nx + (Np + 1) * nx + (Nc) * nu:(Np + 1) * nx + (Np + 1) * nx + (Nc) * nu + nu] = Dumin + uminus1_rh[
                                                                                                            0:nu]  # update constraint on \Delta u0: Dumin <= u0 - u_{-1}
        self.u[
        (Np + 1) * nx + (Np + 1) * nx + (Nc) * nu:(Np + 1) * nx + (Np + 1) * nx + (Nc) * nu + nu] = Dumax + uminus1_rh[
                                                                                                           0:nu]  # update constraint on \Delta u0: u0 - u_{-1} <= Dumax

        # Update linear term q
        q_X = np.zeros((Np + 1) * nx)
        self.J_CNST = 0.0
        if self.JX_ON:
            if xref.ndim == 2 and xref.shape[0] >= Np + 1:  # xref is a vector per time-instant! experimental feature
                # for idx_ref in range(Np):
                #    q_X[idx_ref * nx:(idx_ref + 1) * nx] += -Qx.dot(xref[idx_ref, :])
                # q_X[Np * nx:(Np + 1) * nx] += -QxN.dot(xref[Np, :])
                q_X += (-xref.reshape(1, -1) @ (
                    P_X)).ravel()  # way faster implementation of the same formula commented above

                if self.COMPUTE_J_CNST:
                    self.J_CNST += -1 / 2 * q_X @ xref.ravel()
            else:
                q_X += np.hstack([np.kron(np.ones(Np), -Qx.dot(xref)),  # x0... x_N-1
                                  -QxN.dot(xref)])  # x_N
                if self.COMPUTE_J_CNST:
                    self.J_CNST += 1 / 2 * Np * (xref.dot(QxN.dot(xref))) + 1 / 2 * xref.dot(QxN.dot(xref))
        else:
            pass

        q_U = np.zeros(Nc * nu)
        if self.JU_ON:
            self.J_CNST += 1 / 2 * Np * (uref.dot(Qu.dot(uref)))
            if self.Nc == self.Np:
                q_U += np.kron(np.ones(Nc), -Qu.dot(uref))
            else:  # Nc < Np. This formula is more general and could handle the case Nc = Np either. TODO: test
                iU = np.ones(Nc)
                iU[Nc - 1] = (Np - Nc + 1)
                q_U += np.kron(iU, -Qu.dot(uref))

        # Filling P and q for J_DU
        if self.JDU_ON:
            self.J_CNST += 1 / 2 * uminus1_rh.dot((QDu).dot(uminus1_rh))
            q_U += np.hstack([-QDu.dot(uminus1_rh),  # u0
                              np.zeros((Nc - 1) * nu)])  # u1..uN-1
        else:
            pass

        if self.SOFT_ON:
            q_eps = np.zeros((Np + 1) * nx)
            self.q = np.hstack([q_X, q_U, q_eps])
        else:
            self.q = np.hstack([q_X, q_U])

        self.prob.update(l=self.l, u=self.u, q=self.q)


