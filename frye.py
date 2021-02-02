from badapted.model import Model
from badapted.choice_functions import CumulativeNormalChoiceFunc
from badapted.designs import FryeEtAlGenerator
from badapted.parameter_recovery import simulated_experiment_trial_loop

import scipy
from scipy.stats import norm, halfnorm, uniform
import numpy as np
import pandas as pd
import time
import pickle
from tqdm import trange

import multiprocessing
from joblib import Parallel, delayed


class MyCustomModel(Model):

    def __init__(self, n_particles=1000,
                 prior={'logk': norm(loc=-4.25, scale=1.5),
                        'α': halfnorm(loc=0, scale=2)}):

        self.n_particles = n_particles
        self.prior = prior
        self.θ_fixed = {'ϵ': 0.01}
        # Annoying, this is why they invented probabilistic programming
        true_alpha = np.abs(scipy.random.normal(loc=0., scale=2.))
        true_logk = scipy.random.normal(loc=-4.25, scale=1.5)
        self.θ_true = pd.DataFrame([{'α': true_alpha, 'logk': true_logk}])

        self.choiceFunction = CumulativeNormalChoiceFunc

    def predictive_y(self, θ, data, display=False):

        k = np.exp(θ['logk'].values)
        VA = data['RA'].values * 1 / (1 + k * data['DA'].values)
        VB = data['RB'].values * 1 / (1 + k * data['DB'].values)
        decision_variable = VB - VA

        # Step 2 - apply choice function
        p_chose_B = self.choiceFunction(decision_variable, θ, self.θ_fixed)
        return p_chose_B


def run_exp():
    # Create a design generator using that design space
    design_generator = FryeEtAlGenerator(
        d_a=0., d_b_space=[7, 30, 90, 365], r_b=100., max_trials=20, trials_per_delay=5
    )

    # Create a model object
    model = MyCustomModel()

    # Run a simulated experiment
    model, design_generator= simulated_experiment_trial_loop(design_generator, model)

    return design_generator.data, model.θ_true


if __name__ == '__main__':

    t = time.time()
    processed_list = Parallel(n_jobs=40)(delayed(run_exp)() for i in trange(10000))
    print("Time", time.time() - t)

    with open('frye_results_T20.pickle', 'wb') as f:
        pickle.dump(processed_list, f)
