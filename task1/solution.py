import os
import typing
from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.neighbors import KNeighborsRegressor as KNNR
from sklearn.model_selection import train_test_split


# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = True
EVALUATION_GRID_POINTS = 150  # Number of grid points used in extended evaluation

# Cost function constants
COST_W_UNDERPREDICT = 50.0
COST_W_NORMAL = 1.0


class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.rng = np.random.default_rng(seed=0)
        
        noise_kernel = 0.1**2 * RBF(length_scale=0.1) + WhiteKernel(noise_level=0.1**2, noise_level_bounds=(1e-5, 1e5))

        self.kernel =  0.8**2 * RationalQuadratic(length_scale=2.0, alpha=2) + noise_kernel #5.0**2 * RBF(length_scale=[10.0,10.0]) #1.0*RBF([1.0,1.0])#DotProduct() + WhiteKernel()
        self.model = GaussianProcessRegressor(kernel=self.kernel, random_state=0, normalize_y=False)
        
        self.gp1=GaussianProcessRegressor(kernel=self.kernel, random_state=0, normalize_y=False)
        self.gp2=GaussianProcessRegressor(kernel=self.kernel, random_state=0, normalize_y=False)
        self.gp3=GaussianProcessRegressor(kernel=self.kernel, random_state=0, normalize_y=False)
        self.gp4=GaussianProcessRegressor(kernel=self.kernel, random_state=0, normalize_y=False)

        #self.model = KNNR(n_neighbors=11, weights='distance')

    def make_predictions(self, test_x_2D: np.ndarray, test_x_AREA: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of city_areas.
        :param test_x_2D: city_areas as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param test_x_AREA: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """

        gp_mean = np.zeros(test_x_2D.shape[0], dtype=float)
        gp_std = np.zeros(test_x_2D.shape[0], dtype=float)
        predictions = np.zeros(test_x_2D.shape[0], dtype=float)

        model = None
        for index, pair in enumerate(test_x_2D):
            if pair[0] < self.mid_x:
                if pair[1] < self.mid_y:
                    gp_mean[index], gp_std[index] = self.gp1.predict(pair.reshape(1,-1), True, False)
                    predictions[index] = gp_mean[index] + self.gp1_y_mean
                else:
                    gp_mean[index], gp_std[index] = self.gp4.predict(pair.reshape(1,-1), True, False)
                    predictions[index] = gp_mean[index] + self.gp4_y_mean
            else:
                if pair[1] < self.mid_y:
                    gp_mean[index], gp_std[index] = self.gp2.predict(pair.reshape(1,-1), True, False)
                    predictions[index] = gp_mean[index] + self.gp2_y_mean
                else:
                    gp_mean[index], gp_std[index] = self.gp3.predict(pair.reshape(1,-1), True, False)
                    predictions[index] = gp_mean[index] + self.gp3_y_mean

        
        #gp_mean, gp_std = self.model.predict(test_x_2D, True, False)
        #predictions = gp_mean + self.y_mean
        #predictions = self.model.predict(test_x_2D)
        
        mask = [bool(AREA_idx) for AREA_idx in test_x_AREA]
        predictions += mask*gp_std*1.2

        return predictions, gp_mean, gp_std

    def fitting_model(self, train_y: np.ndarray,train_x_2D: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_x_2D: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_y: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """
        self.min_x = np.min(train_x_2D[:,0])
        self.min_y = np.min(train_x_2D[:,1])
        self.max_x = np.max(train_x_2D[:,0])
        self.max_y = np.max(train_x_2D[:,1])
        self.mid_x = (self.max_x+self.min_x)/2
        self.mid_y = (self.max_y+self.min_y)/2

        gp1_index = []
        gp2_index = []
        gp3_index = []
        gp4_index = []
        for index, pair in enumerate(train_x_2D):
            if pair[0] < self.mid_x:
                if pair[1] < self.mid_y:
                    gp1_index.append(index)
                else:
                    gp4_index.append(index)
            else:
                if pair[1] < self.mid_y:
                    gp2_index.append(index)
                else:
                    gp3_index.append(index)

        gp1_train = np.array([train_x_2D[index,:] for index in gp1_index])
        gp2_train = np.array([train_x_2D[index,:] for index in gp2_index])
        gp3_train = np.array([train_x_2D[index,:] for index in gp3_index])
        gp4_train = np.array([train_x_2D[index,:] for index in gp4_index])

        gp1_y = np.array([train_y[index] for index in gp1_index])
        gp2_y = np.array([train_y[index] for index in gp2_index])
        gp3_y = np.array([train_y[index] for index in gp3_index])
        gp4_y = np.array([train_y[index] for index in gp4_index])

        #X_train, X_test, y_train, y_test = train_test_split(train_x_2D, train_y, test_size=0.3, random_state=42)
        
        self.gp1_y_mean = gp1_y.mean()
        self.gp2_y_mean = gp2_y.mean()
        self.gp3_y_mean = gp3_y.mean()
        self.gp4_y_mean = gp4_y.mean()

        print(f'Len gp1: {len(gp1_y)}, Len gp2: {len(gp2_y)}, Len gp3: {len(gp3_y)}, Len gp4: {len(gp4_y)}')

        #self.y_mean = y_train.mean()
        #self.model.fit(X_train, y_train-self.y_mean)

        print("Training GP1")        
        self.gp1.fit(gp1_train, gp1_y-self.gp1_y_mean)
        print("Training GP2")
        self.gp2.fit(gp2_train, gp2_y-self.gp2_y_mean)
        print("Training GP3")
        self.gp3.fit(gp3_train, gp3_y-self.gp3_y_mean)
        print("Training GP4")
        self.gp4.fit(gp4_train, gp4_y-self.gp4_y_mean)


    def make_custom_distribution(self, data, desired_distribution):
        pass
    #     for label, desired_percentage in desired_distribution.items():
    #         current_percentage = current_distribution.get(label, 0) / len(data)
    #         if current_percentage < desired_percentage:
    #             # Oversample the class
    #             oversampled_data = resample(
    #                 [sample for sample in data if sample[1] == label],
    #                 n_samples=int(len(data) * (desired_percentage - current_percentage)),
    #                 random_state=42,
    #             )
    #             new_data.extend(oversampled_data)
    #         elif current_percentage > desired_percentage:
    #             # Undersample the class
    #             undersampled_data = resample(
    #                 [sample for sample in data if sample[1] == label],
    #                 n_samples=int(len(data) * (current_percentage - desired_percentage)),
    #                 replace=False,
    #                 random_state=42,
    #             )
    #             new_data.extend(undersampled_data)
    #         else:
    #             # No need to modify this class
    #             new_data.extend([sample for sample in data if sample[1] == label])

    # # Shuffle the new dataset
    # np.random.shuffle(new_data)

# You don't have to change this function
def cost_function(ground_truth: np.ndarray, predictions: np.ndarray, AREA_idxs: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param ground_truth: Ground truth pollution levels as a 1d NumPy float array
    :param predictions: Predicted pollution levels as a 1d NumPy float array
    :param AREA_idxs: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
    :return: Total cost of all predictions as a single float
    """
    assert ground_truth.ndim == 1 and predictions.ndim == 1 and ground_truth.shape == predictions.shape

    # Unweighted cost
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL

    # Case i): underprediction
    mask = (predictions < ground_truth) & [bool(AREA_idx) for AREA_idx in AREA_idxs]
    weights[mask] = COST_W_UNDERPREDICT

    # Weigh the cost and return the average
    return np.mean(cost * weights)


# You don't have to change this function
def is_in_circle(coor, circle_coor):
    """
    Checks if a coordinate is inside a circle.
    :param coor: 2D coordinate
    :param circle_coor: 3D coordinate of the circle center and its radius
    :return: True if the coordinate is inside the circle, False otherwise
    """
    return (coor[0] - circle_coor[0])**2 + (coor[1] - circle_coor[1])**2 < circle_coor[2]**2

# You don't have to change this function 
def determine_city_area_idx(visualization_xs_2D):
    """
    Determines the city_area index for each coordinate in the visualization grid.
    :param visualization_xs_2D: 2D coordinates of the visualization grid
    :return: 1D array of city_area indexes
    """
    # Circles coordinates
    circles = np.array([[0.5488135, 0.71518937, 0.17167342],
                    [0.79915856, 0.46147936, 0.1567626 ],
                    [0.26455561, 0.77423369, 0.10298338],
                    [0.6976312,  0.06022547, 0.04015634],
                    [0.31542835, 0.36371077, 0.17985623],
                    [0.15896958, 0.11037514, 0.07244247],
                    [0.82099323, 0.09710128, 0.08136552],
                    [0.41426299, 0.0641475,  0.04442035],
                    [0.09394051, 0.5759465,  0.08729856],
                    [0.84640867, 0.69947928, 0.04568374],
                    [0.23789282, 0.934214,   0.04039037],
                    [0.82076712, 0.90884372, 0.07434012],
                    [0.09961493, 0.94530153, 0.04755969],
                    [0.88172021, 0.2724369,  0.04483477],
                    [0.9425836,  0.6339977,  0.04979664]])
    
    visualization_xs_AREA = np.zeros((visualization_xs_2D.shape[0],))

    for i,coor in enumerate(visualization_xs_2D):
        visualization_xs_AREA[i] = any([is_in_circle(coor, circ) for circ in circles])

    return visualization_xs_AREA

# You don't have to change this function
def perform_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_xs_2D = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)
    visualization_xs_AREA = determine_city_area_idx(visualization_xs_2D)
    
    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.make_predictions(visualization_xs_2D, visualization_xs_AREA)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0

    # Plot the actual predictions
    fig, ax = plt.subplots()
    ax.set_title('Extended visualization of task 1')
    im = ax.imshow(predictions, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax = ax)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()

def extract_city_area_information(train_x: np.ndarray, test_x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the city_area information from the training and test features.
    :param train_x: Training features
    :param test_x: Test features
    :return: Tuple of (training features' 2D coordinates, training features' city_area information,
        test features' 2D coordinates, test features' city_area information)
    """
    train_x_2D = np.zeros((train_x.shape[0], 2), dtype=float)
    train_x_AREA = np.zeros((train_x.shape[0],), dtype=bool)
    test_x_2D = np.zeros((test_x.shape[0], 2), dtype=float)
    test_x_AREA = np.zeros((test_x.shape[0],), dtype=bool)

    train_x_2D[:] = train_x[:,:2]
    train_x_AREA[:] = train_x[:,-1] 
    test_x_2D[:] = test_x[:,:2]
    test_x_AREA[:] = test_x[:,-1] 

    assert train_x_2D.shape[0] == train_x_AREA.shape[0] and test_x_2D.shape[0] == test_x_AREA.shape[0]
    assert train_x_2D.shape[1] == 2 and test_x_2D.shape[1] == 2
    assert train_x_AREA.ndim == 1 and test_x_AREA.ndim == 1

    return train_x_2D, train_x_AREA, test_x_2D, test_x_AREA

def scatter_plot(x1, y1, x2, y2, train_x_AREA, test_x_AREA):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    colors = ['blue' if value > 0.5 else 'red' for value in train_x_AREA]
        
    ax1.scatter(x1, y1, c=colors, label='Training data')
    ax1.set_title('Training data distribution')
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')
    ax1.legend()

    colors = ['blue' if value > 0.5 else 'red' for value in test_x_AREA]
    ax2.scatter(x2, y2, c=colors, label='Test data')
    ax2.set_title('Test data distribution')
    ax2.set_xlabel('X-axis')
    ax2.set_ylabel('Y-axis')
    ax2.legend()

    plt.tight_layout()
    plt.show()

# you don't have to change this function
def main():
    # Load the training dateset and test features
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Extract the city_area information
    train_x_2D, train_x_AREA, test_x_2D, test_x_AREA = extract_city_area_information(train_x, test_x)
    # Fit the model
    print('Fitting model')
    model = Model()
    if True:
        X_train, X_test, y_train, y_test, area_train, area_test = train_test_split(train_x_2D, train_y, train_x_AREA, test_size=0.2, random_state=42)
        model.fitting_model(y_train,X_train)

        # Predict on the test features
        print('Predicting on test features')
        #predictions = model.make_predictions(test_x_2D, test_x_AREA)
        #print(predictions)
        
        #for index, pair in enumerate(X_train):
        #   print(index, pair.reshape(-1,1), model.gp1.predict(pair.reshape(1,-1)))

        predictions2 = model.make_predictions(X_test, area_test)
        print(cost_function(y_test,predictions2[0], area_test))

        if EXTENDED_EVALUATION:
            perform_extended_evaluation(model, output_dir='.')
    else:
        count_res = np.count_nonzero(train_x_AREA == 1)
        count_non_res = train_x_AREA.shape[0]-count_res
        print(count_res, count_non_res)
        scatter_plot(train_x_2D[:,0],train_x_2D[:,1], test_x_2D[:,0],test_x_2D[:,1], train_x_AREA, test_x_AREA)


if __name__ == "__main__":
    main()
