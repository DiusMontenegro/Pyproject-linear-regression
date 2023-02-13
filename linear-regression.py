import random
import matplotlib.pyplot as plt
import numpy as np

# Class for generating data points
class DataGenerator:
    def __init__(self, num_points):
        # Number of data points to generate
        self.num_points = num_points

    def generate_data(self):
        # Generates x values in the range of 0 to 10, spaced self.num_points apart
        x = np.linspace(0, 10, self.num_points)
        # Generates y values as 2 * x + 1 with some random noise
        y = 2 * x + 1 + np.random.normal(0, 1, self.num_points)
        return x, y