import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel, MultitaskKernel
from gpytorch.means import ZeroMean, MultitaskMean
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
from torch.autograd import grad


class MultitaskGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        
        # Define the mean and covariance modules
        self.mean_module = MultitaskMean(
            ZeroMean(), num_tasks=num_tasks
        )
        
        self.covar_module = MultitaskKernel(
            RBFKernel(),
            num_tasks=num_tasks,
            rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)


class GPModel(ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super(GPModel, self).__init__(train_x, train_y, likelihood)
                self.mean_module = ZeroMean()
                self.covar_module = ScaleKernel(RBFKernel())

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return MultivariateNormal(mean_x, covar_x)



class GPyTorchGP:
    def __init__(self, X, Y, training_iterations=100, learning_rate=0.1):

        # Standardize the inputs and outputs
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()

        X_scaled = self.scaler_X.fit_transform(X)
        Y_scaled = self.scaler_Y.fit_transform(Y)

        num_tasks = Y_scaled.shape[1]

        # Convert to torch tensors
        self.train_x = torch.tensor(X_scaled, dtype=torch.float32)
        self.train_y = torch.tensor(Y_scaled, dtype=torch.float32)

        # Store dimensions for easy access later
        self.N_gp = self.train_x.shape[0]  # Number of training data points
        self.Ny_gp = self.train_y.shape[1]  # Number of outputs (output dimensions)
        self.Nu_gp = X.shape[1] - Y.shape[1]  # Number of control inputs (assuming X includes both states and controls)

        # Initialize the Gaussian likelihood and the multitask GP model
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks)
        self.model = MultitaskGPModel(self.train_x, self.train_y, self.likelihood, num_tasks=num_tasks)

        # # Initialize the Gaussian likelihood and the GP model
        # self.likelihood = GaussianLikelihood()
        # self.model = GPModel(self.train_x, self.train_y, self.likelihood)


        # Train the model
        self.train(training_iterations, learning_rate)


    def train(self, training_iterations=100, learning_rate=0.1):
        self.model.train()
        self.likelihood.train()

        optimizer = Adam(self.model.parameters(), lr=learning_rate)
        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)


        for i in range(training_iterations):
            optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Iteration {i+1}/{training_iterations} - Loss: {loss.item()}")
            
    
    def compute_posterior(self, x):
        self.model.eval()
        with torch.no_grad():
            posterior = self.model(x)
            mean = posterior.mean
            covar = posterior.covariance_matrix
        return mean, covar  


    def compute_jacobian(self, x):
        # Function to compute the Jacobian of the GP mean function
        mean_function = lambda x: self.model(x).mean
        jac = torch.autograd.functional.jacobian(mean_function, x)
        return jac
    

    # Uncertainty propagation methods

    def compute_taylor_approximation(self, mean, covar, jac, input_covariance):
        # Compute the Taylor approximation of the covariance
        approx_covar = covar + jac @ input_covariance @ jac.transpose(-1, -2)
        return mean, approx_covar
    
    def compute_me_approximation(self, mean, covar):
        # Compute the mean equivalence approximation of the covariance
        approx_covar = covar
        return mean, approx_covar


    def predict(self, x, u, cov, approximation_method="taylor"):

        # Combine state and control input into a single input
        input_data = torch.cat((x, u), dim=-1)

        # Standardize the input
        input_data_scaled = torch.tensor(self.scaler_X.transform(input_data), dtype=torch.float32)

        # Compute the GP predictions and Jacobian
        mean, covar = self.compute_posterior(input_data_scaled)
        jac = self.compute_jacobian(input_data_scaled)

        # Choose approximation method
        if approximation_method == "taylor":
            mean, approx_covar = self.compute_taylor_approximation(mean, covar, jac, cov)
        elif approximation_method == "mean_equivalence":
            mean, approx_covar = self.compute_me_approximation(mean, covar, cov)
        else:
            raise ValueError(f"Unknown approximation method: {approximation_method}")

        # Denormalize the mean
        mean = torch.tensor(self.scaler_Y.inverse_transform(mean.numpy()), dtype=torch.float32)

        return mean, approx_covar
        

    def get_size(self):
        return self.N_gp, self.Ny_gp, self.Nu_gp


    def discrete_linearize(self, x0, u0, cov0):
        """ Linearize the GP around the operating point
            x[k+1] = Ax[k] + Bu[k]
        # Arguments:
            x0: State vector
            u0: Input vector
            cov0: Covariance
        """
        # Combine state and control input into a single input
        combined_input = torch.cat((x0, u0), dim=-1)
        
        # Standardize the combined input
        combined_input_scaled = torch.tensor(self.scaler_X.transform(combined_input), dtype=torch.float32)
        
        # Compute the Jacobian using the standardized input
        jacobian_matrix = self.compute_jacobian(combined_input_scaled)

        # Extract A and B matrices from the Jacobian
        state_dim = x0.size(-1)
        Ad = jacobian_matrix[:, :state_dim]  # Derivative w.r.t x (state)
        Bd = jacobian_matrix[:, state_dim:]  # Derivative w.r.t u (control input)

        return Ad, Bd



# # Example usage
# input_dim = 5
# output_dim = 1
# gp_model = GPyTorchGP(input_dim, output_dim, normalize=True)
# # Training data
# train_x = ...  # Your training input data here
# train_y = ...  # Your training output data here
# gp_model.train(train_x, train_y)

# # Test data
# test_x = ...  # Your test input data here
# mean, covar = gp_model.predict(test_x)
