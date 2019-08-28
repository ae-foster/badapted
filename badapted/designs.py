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

        # # generate empty dataframe
        # data_columns = ['RA', 'DA', 'PA', 'RB', 'DB', 'PB', 'R']
        # self.data = pd.DataFrame(columns=data_columns)


    @abstractmethod
    def get_next_design(self, model):
        ''' This method must be implemented in concrete classes. It should
        output either a Design (a named tuple we are using), or a None when
        there are no more designs left.
        '''
        pass


    def enter_trial_design_and_response(self, design, response):
        '''middle-man method'''

        # TODO: need to specify types here I think... then life might be
        # easier to decant the data out at another point
        # trial_df = design_to_df(design)
        # self.data = self.data.append(trial_df)

        trial_data = {'RA': design.ProspectA.reward,
                    'DA': design.ProspectA.delay,
                    'PA': design.ProspectA.prob,
                    'RB': design.ProspectB.reward,
                    'DB': design.ProspectB.delay,
                    'PB': design.ProspectB.prob,
                    'R': [int(response)]}
        self.data = self.data.append(pd.DataFrame(trial_data))
        # a bit clumsy but...
        self.data['R'] = self.data['R'].astype('int64')
        self.data = self.data.reset_index(drop=True)


        self.trial += 1

        # we potentially manually call model to update beliefs here. But so far
        # this is done manually in PsychoPy
        return


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
