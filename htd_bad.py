from badapted.model import Model
from badapted.choice_functions import CumulativeNormalChoiceFunc
from badapted.designs import BayesianAdaptiveDesignGenerator
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


def build_my_design_space():
    delays = [1, 2, 3, 4, 5, 6, 7, 2 * 7, 3 * 7, 4 * 7,
              3 * 30, 4 * 30, 5 * 30, 6 * 30, 8 * 30, 9 * 30,
              1 * 365, 2 * 365, 3 * 365, 4 * 365, 5 * 365, 6 * 365, 7 * 365,
              8 * 365, 10 * 365, 15 * 365, 20 * 365, 25 * 365]
    rewards = list(range(1,100))
    data = []
    for i in range(len(delays)):
        for j in range(len(rewards)):
            optns = {'DA': 0, 'RB': 100, 'DB': delays[i], 'RA': rewards[j]}
            data.append(optns)
    designs = pd.DataFrame(data)
    return designs


class MyCustomModel(Model):

    def __init__(self, n_particles=20,
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
        if display:
            # print('VA', VA, 'VB', VB)
            pass
        decision_variable = VB - VA

        # Step 2 - apply choice function
        p_chose_B = self.choiceFunction(decision_variable, θ, self.θ_fixed)
        return p_chose_B


def run_exp(designs):
    # Create a design generator using that design space
    design_generator = BayesianAdaptiveDesignGenerator(designs, max_trials=20)

    # Create a model object
    model = MyCustomModel()

    # Run a simulated experiment
    model, design_generator= simulated_experiment_trial_loop(design_generator, model)

    return design_generator.data, model.θ_true


if __name__ == '__main__':

    # Build your design space
    designs = build_my_design_space()

    t = time.time()
    processed_list = Parallel(n_jobs=40)(delayed(run_exp)(designs) for i in trange(10000))
    print("Time", time.time() - t)

    with open('badapted_T20_run2.pickle', 'wb') as f:
        pickle.dump(processed_list, f)


