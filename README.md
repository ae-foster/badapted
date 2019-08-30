# BADAPTED: Bayesian ADAPTive Experimental Design

[![PyPI](https://img.shields.io/pypi/v/badapted.svg?color=green)](https://pypi.org/project/badapted/)

**Status: Working code, but still under development 🔥**

Run efficient Bayesian adaptive experiments.

This code relates to the following pre-print. But, the pre-print is likely to appear in quite a different form when finally published.
> Vincent, B. T., & Rainforth, T. (2017, October 20). The DARC Toolbox: automated, flexible, and efficient delayed and risky choice experiments using Bayesian adaptive design. Retrieved from psyarxiv.com/yehjb


## Building your own adaptive experiment toolbox on top of `badapted`

Below we outline how the `badapted` package can be used to run adaptive experiments. On it's own, this `badapted` package will not do anything. It also requires a few classes (and probably some helper functions) that a developer must create for their particular experimental paradigm. This forms a 'toolbox' which will allow adaptive experiments to be run in a particular experimental domain.

The best (first) example of this is our [DARC Toolbox](https://github.com/drbenvincent/darc_toolbox) which allows adaptive experiments for Delayed And Risky Choice tasks.

But below we outline how to go about creating a new 'toolbox' for your experimental domain of interest.


### Step 1: define your design space

First we create a pandas dataframe called `designs` using a function we write to do this. Each column is a design variable. Each row is a particular design.

```python
def build_my_design_space(my_arguments):
    designs = # CREATE PANDAS DATAFRAME OF THE DESIGN SPACE HERE
    return designs
```

### Step 2: define a custom design generator

In order to generate your own design generator that uses Bayesian Adaptive Design (in your experimental domain) then you need to create a class which subclasses `badapted.BayesianAdaptiveDesignGenerator`. You will also need to implement some methods specific to your experimental domain, notably:
- `add_design_response_to_dataframe`
- `df_to_design_tuple`

For the moment, we will just provide the example we use in the DARC Toolbox. Firstly, our concrete design generator class is defined as:

```python
from badapted.designs import BayesianAdaptiveDesignGenerator

class BayesianAdaptiveDesignGeneratorDARC(DARCDesignGenerator, BayesianAdaptiveDesignGenerator):
    '''This will be the concrete class for doing Bayesian adaptive design
    in the DARC experiment domain.'''

    def __init__(self, design_space,
                 max_trials=20,
                 allow_repeats=True,
                 penalty_function_option='default',
                 λ=2):

        # call superclass constructors - note that the order of calling these is important
        BayesianAdaptiveDesignGenerator.__init__(self, design_space,
                 max_trials=max_trials,
                 allow_repeats=allow_repeats,
                 penalty_function_option=penalty_function_option,
                 λ=λ)

        DARCDesignGenerator.__init__(self)
```

Note that this has mulitple inheritance, so we also have a class `DARCDesignGenerator` which just includes DARC specific methods (`add_design_response_to_dataframe`, `df_to_design_tuple`). This is defined as:

```python
from badapted.designs import DesignGeneratorABC
from darc_toolbox import Prospect, Design


class DARCDesignGenerator(DesignGeneratorABC):
    '''This adds DARC specific functionality to the design generator'''

    def __init__(self):
        # super().__init__()
        DesignGeneratorABC.__init__(self)

        # generate empty dataframe
        data_columns = ['RA', 'DA', 'PA', 'RB', 'DB', 'PB', 'R']
        self.data = pd.DataFrame(columns=data_columns)

    def add_design_response_to_dataframe(self, design, response):
        '''
        This method must take in `design` and `reward` from the current trial
        and store this as a new row in self.data which is a pandas data frame.
        '''

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
        return

    @staticmethod
    def df_to_design_tuple(df):
        '''User must impliment this method. It takes in a design in the form of a
        single row of pandas dataframe, and it must return the chosen design as a
        named tuple.
        Convert 1-row pandas dataframe into named tuple'''
        RA = df.RA.values[0]
        DA = df.DA.values[0]
        PA = df.PA.values[0]
        RB = df.RB.values[0]
        DB = df.DB.values[0]
        PB = df.PB.values[0]
        chosen_design = Design(ProspectA=Prospect(reward=RA, delay=DA, prob=PA),
                            ProspectB=Prospect(reward=RB, delay=DB, prob=PB))
        return chosen_design
```

We only did this multiple inheritance because we wanted other (non Bayesian Adaptive) design generators which worked in the DARC domain, but did not have any of the Bayesian Adaptive Design components. In most situations just focussing on Bayesian Adaptive Design, you could just define the `add_design_response_to_dataframe`, `df_to_design_tuple` classes in your one single concrete design generator class.


### Step 3: define a model

You must provide a model class which inherits from `Model`. You must also provide the following methods:

- `__init__`
- `predictive_y`

Here is an example of a minimal implimentation of a user-defined model:

```python
from badapted.model import Model
from badapted.choice_functions import CumulativeNormalChoiceFunc, StandardCumulativeNormalChoiceFunc
from scipy.stats import norm, halfnorm, uniform
import numpy as np


class MyCustomModel(Model):
    '''My custom model which does XYZ.'''

    def __init__(self, n_particles,
                 prior={'logk': norm(loc=-4.5, scale=1),
                        'α': halfnorm(loc=0, scale=2)}):
        '''
        INPUTS
        - n_particles (integer).
        - prior (dictionary). The keys provide the parameter name. The values
        must be scipy.stats objects which define the prior distribution for
        this parameter.

        We provide choice functions in `badapted.choice_functions.py`. In this
        example, we define it in the __init__ but it is not necessary to happen
        here.
        '''
        self.n_particles = int(n_particles)
        self.prior = prior
        self.θ_fixed = {'ϵ': 0.01}
        self.choiceFunction = CumulativeNormalChoiceFunc

    def predictive_y(self, θ, data):
        '''
        INPUTS:
        - θ = parameters
        - data =

        OUTPUT:
        - p_chose_B (float) Must return a value between 0-1.
        '''

        # Step 1 - calculate decision variable
        k = np.exp(θ['logk'].values
        VA = data['RA'].values * 1 / (1 + k * data['DA'].values)
        VB = data['RB'].values * 1 / (1 + k * data['DB'].values)
        decision_variable = VB - VA

        # Step 2 - apply choice function
        p_chose_B = self.choiceFunction(decision_variable, θ, self.θ_fixed)
        return p_chose_B
```

### Step 4: build an experiment trial loop

This is pretty straight-forward and there doesn't need to be any major customisation here.

```python
def run_experiment(design_generator, model, max_trials):
    '''Run an adaptive experiment
    INPUTS:
    - design_generator: a class
    '''

    for trial in range(max_trials):
        design = design_generator.get_next_design(model)
        if design is None:
            break
        response = get_response(design)
        design_generator.enter_trial_design_and_response(design, response)
        model.update_beliefs(design_generator.data)

    return model
```

Note that the `response = get_response(design)` line is up to you to impliment. What you do here depends on whether you are simulating responses or getting real responses from PsychoPy etc. The `run_experiment` function is just an example of how the various parts of the code work together. When running _actual_ experiments using PsychoPy, it is best to refer to the demo psychopy files we provide in the [DARC Toolbox](https://github.com/drbenvincent/darc_toolbox) as examples to see how this is done.

### Step 5: setup and run the experiment

```python
designs = build_my_design_space(my_arguments)
design_generator = MyCustomDesignGenerator(designs, max_trials=max_trials)
model = MyCustomModel()

model = run_experiment(design_generator, model, max_trials)
```

Note that use of the `run_experiment` function is just a demonstration of the logic of how things fit together. As mentioned, please refer to PsychoPy example experiments in the [DARC Toolbox](https://github.com/drbenvincent/darc_toolbox) to see how this all comes together in a PsychoPy experiment.


## Toolboxes using `badapted`
- [DARC Toolbox](https://github.com/drbenvincent/darc_toolbox) for adpative Delayed and Risky Choice tasks.
- [Adaptive Psychophysics Toolbox](https://github.com/drbenvincent/adaptive_psychophysics_toolbox) for psychophysical experiments. _[Note: still under active development.]_
