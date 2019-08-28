'''
Provides base classes related to experimental designs to be used by _any_
domain specific use of this Bayesian Adaptive Design package.
'''


from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging
import time
import copy
import random
from scipy.stats import multivariate_normal
from badapted.optimisation import design_optimisation


class DesignGeneratorABC(ABC):
    '''
    The top level Abstract Base class for designs. It is not functional in itself,
    but provides the template for handling designs for given contexts, such as DARC.

    Core functionality is:
    a) It pumps out experimental designs with the get_next_design() method.
    b) It recieves the resulting (design, response) pair and stores a history of
       designs and responses.

    One reason why we have this DesignGeneratorABC class is because we want to go of
    in different directions. For example, we might want to create an adaptive class
    which inherits from this one. But we might also want to create a heuristic design
    generator which inherits from this one. Thus we have this DesignGeneratorABC class
    which just adds a few
    '''


    def __init__(self):
        self.trial = int(0)
        self.data = None


    @abstractmethod
    def get_next_design(self, model):
        ''' This method must be implemented in concrete classes. It should
        output either a Design (a named tuple we are using), or a None when
        there are no more designs left.
        '''
        pass


    def enter_trial_design_and_response(self, design, response):
        '''Call this function with the last design and it's corresponding response'''
        self.add_design_response_to_dataframe(design, response)
        self.trial += 1


    @abstractmethod
    def add_design_response_to_dataframe(self, design, response):
        '''
        This method must take in `design` and `reward` from the current trial
        and store this as a new row in self.data which is a pandas data frame.
        '''
        pass

    def get_last_response_chose_B(self):
        '''return True if the last response was for the option B'''
        if self.data.size == 0:
            # no previous responses
            return None

        if list(self.data.R)[-1] == 1:  # TODO: do this better?
            return True
        else:
            return False



    @staticmethod
    @abstractmethod
    def df_to_design_tuple(chosen_design_df):
        '''User must impliment this method. It takes in a design in the form of a
        single row of pandas dataframe, and it must return the chosen design as a
        named tuple'''
        pass


    # # TODO: remove this getter and just access the property directly
    # def get_df(self):
    #     '''return dataframe of data'''
    #     return self.data



class BayesianAdaptiveDesignGenerator(DesignGeneratorABC):
    '''
    This class selects the next design to run, based on a provided design
    space, a model, and a design/response history.
    '''

    def __init__(self, design_space,
                 max_trials=20,
                 allow_repeats=True,
                 penalty_function_option='default',
                 λ=2):
        super().__init__()

        self.all_possible_designs = design_space
        self.max_trials = max_trials
        self.allow_repeats = allow_repeats
        self.penalty_function_option = penalty_function_option
        # extract design variables as a list
        self.design_variables = list(design_space.columns.values)
        self.λ = λ  # penalty factor for _default_penalty_func()

    def get_next_design(self, model):

        if self.trial > self.max_trials - 1:
            return None
        start_time = time.time()
        logging.info(f'Getting design for trial {self.trial}')

        allowable_designs = copy.copy(self.all_possible_designs)
        logging.debug(f'{allowable_designs.shape[0]} designs initially')

        # Refine the design space
        allowable_designs = self._refine_design_space(model, allowable_designs)

        # Some checks to see if we have been way too aggressive
        # in refining the design space
        if allowable_designs.shape[0] == 0:
            logging.error(f'No ({allowable_designs.shape[0]}) designs left')

        if allowable_designs.shape[0] < 10:
            logging.warning(
                f'Very few ({allowable_designs.shape[0]}) designs left')

        # process penalty_function_option provided by user upon object
        # construction
        if self.penalty_function_option is 'default':
            # penalty_func = lambda d: self._default_penalty_func(d, λ=self.λ)
            def penalty_func(d): return self._default_penalty_func(d, λ=self.λ)
        elif self.penalty_function_option is None:
            penalty_func = None

        # Run the low level design optimisation code ~~~~~~~~~~~~~~~~~~~~~~~~~~
        chosen_design_df, _ = design_optimisation(allowable_designs,
                                                  model.predictive_y,
                                                  model.θ,
                                                  n_steps=50,
                                                  penalty_func=penalty_func)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        chosen_design_named_tuple = self.df_to_design_tuple(chosen_design_df)

        logging.debug(f'chosen design is: {chosen_design_named_tuple}')
        logging.info(
            f'get_next_design() took: {time.time()-start_time:1.3f} seconds')
        return chosen_design_named_tuple

    def _refine_design_space(self, model, allowable_designs):
        '''A series of operations to refine down the space of designs which we
        do design optimisations on.'''

        # Remove already run designs, if appropriate
        if not self.allow_repeats and self.trial > 1:
            # strip the responses off of the stored data
            designs_to_exclude = self.data.df.drop(columns=['R'])

            # there is no convenient set difference function for pandas, but
            # this achieves what we want
            allowable_designs = allowable_designs.loc[~allowable_designs.isin(
                designs_to_exclude.to_dict(orient="list")).all(axis=1), :]

        # Remove highly preductable designs
        allowable_designs = _remove_highly_predictable_designs(allowable_designs,
                                                               model)

        return allowable_designs

    def _get_sorted_unique_design_vals(self, design_variable):
        '''return sorted set of unique design values for the given design
        variable
        TODO: refactor this so it only gets called ONCE
        '''
        all_values = self.all_possible_designs[design_variable].values
        return np.sort(np.unique(all_values))

    def _convert_to_ranks(self, input_designs):
        '''Convert each design variable to a quantile in the set of allowed
        values for that variable, with the minimum and maximum
        taken to be 0 and 1 respectively.

        INPUT:
        - input_designs is a subset of rows of self.all_possible_designs
        OUTPUT:
        - design_ranks = ?????
        '''

        design_variable_names = list(input_designs.columns.values)

        # convert input_designs to martrix in the right dtype -----------------
        # input_designs = input_designs.to_numpy(dtype='float64')  # desired
        input_designs = input_designs.values  # for backward compatibility

        design_ranks = np.full_like(input_designs, np.nan)

        for n, design_variable in enumerate(design_variable_names):
            # Get set of instances this design variable can take
            all_designs_this_variable = self._get_sorted_unique_design_vals(design_variable)
            N_this = len(all_designs_this_variable)

            if N_this is 1:
                design_ranks[:, n] = 0.5
            else:
                # Interpolate input_designs so they are a value between 0 and 1
                # based on a linear regression from possible values of design
                # variables to their scaled ranks (where the later is just
                # (0:(N_this-1))'/(N_this-1))
                design_ranks[:, n] = np.interp(input_designs[:, n],
                                               all_designs_this_variable,
                                               np.linspace(0., 1., num=N_this))

        return design_ranks

    def _default_penalty_func(self, candidate_designs, base_sigma=1, λ=2):
        # get previous designs -----------------------------------
        # TODO: refactor so there is a simple getter for just the designs,
        # not the responses
        # This will get previous designs AND responses, then just get design
        # variables
        previous_designs = self.data[self.design_variables]
        # --------------------------------------------------------

        n_previous_designs = previous_designs.shape[0]
        if n_previous_designs is 0:
            # If there are no previous designs, we shouldn't apply any factors
            penalty_factors = np.ones(candidate_designs.shape[0])
            return penalty_factors

        # To keep problem self similarity, we apply the kernel in the space of
        # ranks.
        # Note: output to self._convert_to_ranks() is DataFrame, output is
        # numerical array
        candidate_designs = self._convert_to_ranks(candidate_designs)
        previous_designs = self._convert_to_ranks(previous_designs)

        # Though will be eliminated later, this is useful for doing the
        # rescaling of p without having to write everything twice.
        candidate_designs = np.concatenate((candidate_designs,
                                            previous_designs),
                                           axis=0)

        # Calculate density of each candidate point under a Gaussian
        # distribution centered on each previous design.
        nD, dD = candidate_designs.shape
        # Reduce standard deviation as we get more designs
        sigma = base_sigma/n_previous_designs
        if np.isscalar(sigma):
            sigma = sigma**2*np.eye(dD)
        else:
            sigma = np.diagflat(sigma**2)

        # Calculate the density of each candidate design using a Gaussian
        # centered at each previous point with covariance sigma
        densities = np.empty((nD, n_previous_designs)) # <----- IS THIS CORRECT SIZE?

        for n in range(n_previous_designs):
            densities[:, n] = multivariate_normal.pdf(candidate_designs,
                                                      mean=previous_designs[n, :],
                                                      cov=sigma)

        # p is a vector, one entry for each designs. It represents kernel
        # density estimate of previous design points by averaging over a
        # mixture of Gaussians centered at each previous design
        p = np.mean(densities, axis=1)
        p = p/np.max(p)  # Rescale so all p are between 0 and 1

        # Remove previous designs
        p = p[:-n_previous_designs]

        # Calculate final penalty factors
        penalty_factors = 1.0/(1.0+λ*p)

        # penalty_factors should be a 1D vector
        penalty_factors = np.squeeze(penalty_factors)
        return penalty_factors


def _choose_one_along_design_dimension(allowable_designs, design_dim_name):
    '''We are going to take one design dimension given by `design_dim_name` and randomly
    pick one of it's values and hold it constant by removing all others from the list of
    allowable_designs.
    The purpose of this is to promote variation along the chosen design dimension.
    Cutting down the set of allowable_designs which we do design optimisation on is a
    nice side-effect rather than a direct goal.
    '''
    unique_values = allowable_designs[design_dim_name].unique()
    chosen_value = random.choice(unique_values)
    # filter by chosen value of this dimension
    allowable_designs = allowable_designs.loc[allowable_designs[design_dim_name] == chosen_value]
    return allowable_designs


def _remove_highly_predictable_designs(allowable_designs, model):
    ''' Eliminate designs which are highly predictable as these will not be
    very informative '''

    # grab the design variables. We are going to add some additional columns
    # to the DataFrame in order to do the job, but we want to just grab the
    # design variables after that
    design_variables = allowable_designs.columns.values

    θ_point_estimate = model.get_θ_point_estimate()

    # TODO: CHECK WE CAN EPSILON TO 0
    p_chose_B = model.predictive_y(θ_point_estimate, allowable_designs)
    # add p_chose_B as a column to allowable_designs
    allowable_designs['p_chose_B'] = pd.Series(p_chose_B)

    # Decide which designs (rows) correspond to highly predictable responses
    # threshold = 0.05 means we drop designs with 0>P(y)<0.05 and 0.95<P(y)<1
    threshold = 0.05
    max_threshold = 0.25
    n_not_predictable = 201
    # n_designs_provided = allowable_designs.shape[0]

    while n_not_predictable > 200 and threshold < max_threshold:
        threshold *= 1.05
        highly_predictable, n_predictable, n_not_predictable = _calc_predictability(allowable_designs, threshold)

    # add this as a column in the design DataFrame
    allowable_designs['highly_predictable'] = pd.Series(highly_predictable)

    if n_not_predictable > 10:
        # drop the offending designs (rows)
        allowable_designs = allowable_designs.drop(
            allowable_designs[allowable_designs.p_chose_B < threshold].index)
        allowable_designs = allowable_designs.drop(
            allowable_designs[allowable_designs.p_chose_B > 1 - threshold].index)
        if n_not_predictable > 200:
            allowable_designs = allowable_designs.sample(n=200)
    else:
        # take the 10 designs closest to p_chose_B=0.5
        # NOTE: This is not exactly the same as Tom's implementation which examines
        # VB-VA (which is the design variable axis) and also VA+VB (orthogonal to
        # the design variable axis)
        logging.warning(
            'not many unpredictable designs, so taking the 10 closest to unpredictable')
        allowable_designs['badness'] = np.abs(
            0.5 - allowable_designs.p_chose_B)
        allowable_designs.sort_values(by=['badness'], inplace=True)
        allowable_designs = allowable_designs[:10]

    # Grab just the design variables... we don't want any intermediate columns
    # that we added above
    allowable_designs = allowable_designs[design_variables]

    logging.debug(
        f'{allowable_designs.shape[0]} designs after removing highly predicted designs')
    return allowable_designs


def _calc_predictability(allowable_designs, threshold):
    '''calculate how many designes are classified as predictable for a given
    threshold on 'p_chose_B' '''
    highly_predictable = (allowable_designs['p_chose_B'] < threshold) | (
        allowable_designs['p_chose_B'] > 1 - threshold)
    n_predictable = sum(highly_predictable)
    n_not_predictable = allowable_designs.shape[0] - n_predictable
    return (highly_predictable, n_predictable, n_not_predictable)
