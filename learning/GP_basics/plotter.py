import matplotlib.pyplot as plt
import numpy as np


class GaussianProcessSinglePlotter:
    """
    Plotter for visualizing sampled functions from a Gaussian Process.

    Attributes:
        ys (np.ndarray): 2D array where each row is a sample function.
        X (np.ndarray): 1D array of input points.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        title (str): Title of the plot.
        xlim (tuple): Tuple specifying the x-axis limits (min, max).
        ylim (tuple): Tuple specifying the y-axis limits (min, max).
    """
    

    def __init__(
            self,
            X1=None,
            y1=None,
            X2=None,
            ys=None,
            mu2=None,
            sigma2=None,
            actual_func=None,
            x_label=None,
            y_label=None,
            title=None,
            xlim=None,
            ylim=None,
            func_label=None
    ):
        self.X1 = X1  # training points
        self.y1 = y1  # training points
        self.X2 = X2  # posterior points/ points to predict
        self.ys = ys  # sampled function
        self.mu2 = mu2
        self.sigma2 = sigma2
        self.actual_func = actual_func
        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        self.xlim = xlim
        self.ylim = ylim
        self.func_label = func_label

       
        self.figure, self.ax = plt.subplots(figsize=(6, 4))
        self.ax.grid(True)
        self._configure_plot()

    def _configure_plot(self):
        """Set up the plot labels, title, and limits."""
        # Set labels and title
        self.ax.set_xlabel(self.x_label if self.x_label else 'x', fontsize=13)
        self.ax.set_ylabel(self.y_label if self.y_label else 'y', fontsize=13)
        self.ax.set_title(self.title, fontsize=15)

        # Set x and y limits if provided
        if self.xlim is not None:
            self.ax.set_xlim(self.xlim)
        if self.ylim is not None:
            self.ax.set_ylim(self.ylim)

    def plot_distribution(self):
        """Plot the distribution."""
        if self.X2 is None or self.mu2 is None or self.sigma2 is None:
            raise ValueError("X2, mu2, and sigma2 must be provided for plotting.")

        # Plot the true function
        if self.actual_func is not None:
            self.ax.plot(self.X2, self.actual_func(self.X2), 'b--', label=self.func_label)

        # Plot the confidence interval 95th percentile
        self.ax.fill_between(self.X2.flat, self.mu2 - 2.33 * self.sigma2, self.mu2 + 2.33 * self.sigma2, color='red',
                             alpha=0.15, label=r'$2 \sigma_{2|1}$')

        # Plot the mean function
        self.ax.plot(self.X2, self.mu2, 'r-', lw=1, label=r'$\mu_{2|1}$')

        # Plot the training data
        if self.X1 is not None and self.y1 is not None:
            self.ax.plot(self.X1, self.y1, 'ko', lw=1, label='$(x_1, y_1)$')

        # Optionally, add a legend
        self.ax.legend()

    def plot_sampled_functions(self):
        """Plot sampled functions from the Gaussian Process."""
        if self.ys is None or self.X2 is None:
            raise ValueError("Data (ys and X) must be provided for plotting.")

        # Plot each sampled function
        for i in range(self.ys.shape[0]):
            self.ax.plot(self.X2, self.ys[i], linestyle='-', label=f'Sample {i + 1}')

    def show_plot(self):
        
        plt.show()

