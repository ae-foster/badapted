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


def build_my_design_space():
    delays = [1, 2, 3, 4,5, 6, 7, 8, 9, 12,
              1 * 24, 2 * 24, 3 * 24, 4 * 24, 5 * 24, 6 * 24, 7 *24, 2 * 24 * 7 , 3 * 24 * 7, 4 *24 * 7,
              3 * 24 * 30, 4 * 24 * 30, 5 * 24 * 30, 6 * 24 * 30, 8 * 24 * 30, 9 *24*30,
              1 * 24 * 365, 2 * 24 * 365,3 * 24 * 365, 4 * 24 * 365, 5 * 24 * 365, 6 * 24 * 365, 7 * 24 * 365,
              8*24*365, 10*24*365, 15*24*365, 20*24*365, 25 *24*365]
    rewards = list(range(1,100))
    data = []
    for i in range(len(delays)):
        for j in range(len(rewards)):
            optns = {'DA': 0, 'RB': 100, 'DB': delays[i], 'RA': rewards[j]}
            data.append(optns)
    designs = pd.DataFrame(data)
    return designs


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
        if display:
            # print('VA', VA, 'VB', VB)
            pass
        decision_variable = VB - VA

        # Step 2 - apply choice function
        p_chose_B = self.choiceFunction(decision_variable, θ, self.θ_fixed)
        return p_chose_B


def run_exp(designs):
    # Create a design generator using that design space
    design_generator = FryeEtAlGenerator(d_a=0., d_b_space=[7 * 24, 30 * 24, 365 * 24], r_b=100., max_trials=15, trials_per_delay=5)

    # Create a model object
    model = MyCustomModel()

    # Run a simulated experiment
    model, design_generator= simulated_experiment_trial_loop(design_generator, model)

    return design_generator.data, model.θ_true


if __name__ == '__main__':

    # Build your design space
    designs = build_my_design_space()

    t = time.time()
    processed_list = Parallel(n_jobs=40)(delayed(run_exp)(designs) for i in trange(5000))
    print("Time", time.time() - t)

    with open('frye_results.pickle', 'wb') as f:
        pickle.dump(processed_list, f)

    
