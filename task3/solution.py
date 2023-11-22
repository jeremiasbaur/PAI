"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
# import additional ...
from sklearn.gaussian_process.kernels import *
from sklearn.gaussian_process import GaussianProcessRegressor


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA


# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        self.f_kernel = Matern(nu = 2.5) #+ WhiteKernel(noise_level=0.15)
        self.v_kernel = np.sqrt(2)*RBF(length_scale=1) + 4 #+ WhiteKernel(noise_level=0.0001)
        self.gpf = GaussianProcessRegressor(kernel=self.f_kernel, random_state=0, normalize_y=False)
        self.gpv = GaussianProcessRegressor(kernel=self.v_kernel, random_state=0, normalize_y=False)
        self.tolerance = 4
        self.lamb = 10
        self.x_data = []#np.zeros(shape = (1,DOMAIN.shape(0)))
        self.f_data = []#np.zeros(shape = (1,1))
        self.v_data = []#np.zeros(shape = (1,1))
        pass

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.
        return self.optimize_acquisition_function() 

        #raise NotImplementedError

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        # TODO: Implement the acquisition function you want to optimize.

        #Perform thompson sampling as in Lecture 8, Slide 44
        #f = self.gpf.posterior_samples_f(x)
        #v = self.gpv.posterior_samples_f(x)
        f_mean, f_std = self.gpf.predict(x, return_std=True)
        v_mean, v_std = self.gpv.predict(x, return_std=True)

        f_vals = np.random.normal(loc = f_mean, scale = f_std)
        v_vals = np.random.normal(loc = v_mean, scale = v_std)
        
        return f_vals - self.lamb*max(v_vals - SAFETY_THRESHOLD, 0)

        #raise NotImplementedError

    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """

        # TODO: Add the observed data {x, f, v} to your model.
        #Unsure if this does what its supposed to
        self.x_data.append(x)
        self.f_data.append(f)
        self.v_data.append(v)

        x_input =  np.array(self.x_data).reshape(-1, 1)
        self.gpf.fit(np.array(x_input), np.array(self.f_data))
        self.gpv.fit(np.array(x_input), np.array(self.v_data))

        #self.gpf.fit(x_arr, f)
        #self.gpv.fit(x_arr, v)
        #raise NotImplementedError

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        #x_current = self.next_recommendation()

        # f_mean, f_std = self.gpf.predict(x_current, return_std=True)
        # v_mean, v_std = self.gpv.predict(x_current, return_std=True)
        # f_current = f_mean + np.random.normal(scale = 0.15)
        # v_current = v_mean + np.random.normal(scale = 0.0001)
        
        #Might have to change this
        #self.add_data_point(x_current, f_current, v_current)

        #self.gpf.fit(self.x_data, self.f_data)
        #self.gpv.fit(self.x_data, self.v_data)
        
        mask = [bool(v > SAFETY_THRESHOLD) for v in self.v_data]
        f_vals = np.array(self.f_data).reshape(-1)
        f_max = max(f_vals[mask])
        return np.where(f_vals == f_max)[0]
        #raise NotImplementedError 

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        pass


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()
    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)
    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        # assert x.shape == (1, DOMAIN.shape[0]), \
        #     f"The function next recommendation must return a numpy array of " \
        #     f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.normal()
        cost_val = v(x) + np.random.normal()
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()
