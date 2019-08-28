'''
Provides base classes related to experimental designs to be used by _any_
domain specific use of this Bayesian Adaptive Design package.
'''


from abc import ABC, abstractmethod
import pandas as pd
import logging


class DesignGeneratorABC(ABC):
    '''
    The top level Abstract Base class for designs. It is not functional in itself,
    but provides the template for handling designs for given contexts, such as DARC.

    Core functionality is:
    a) It pumps out experimental designs with the get_next_design() method.
    b) It recieves the resulting (design, response) pair and stores a history of
       designs and responses.
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


    # TODO: remove this getter and just access the property directly
    def get_df(self):
        '''return dataframe of data'''
        return self.data
