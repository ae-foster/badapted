# BADAPTED: Bayesian ADAPTive Experimental Design

[![PyPI](https://img.shields.io/pypi/v/badapted.svg?color=green)](https://pypi.org/project/badapted/)


Run efficient Bayesian adaptive experiments.

This code relates to the following pre-print. But, the pre-print is likely to appear in quite a different form when finally published.
> Vincent, B. T., & Rainforth, T. (2017, October 20). The DARC Toolbox: automated, flexible, and efficient delayed and risky choice experiments using Bayesian adaptive design. Retrieved from psyarxiv.com/yehjb


## Building your own adaptive experiment toolbox on top of `badapted`

Below we outline how the `badapted` package can be used to run adaptive experiments. On it's own, this `badapted` package will not do anything. It also requires a few classes (and probably some helper functions) that a developer must create for their particular experimental paradigm. This forms a 'toolbox' which will allow adaptive experiments to be run in a particular experimental domain.

The best (first) example of this is our [DARC Toolbox](https://github.com/drbenvincent/darc_toolbox) which allows adaptive experiments for Delayed And Risky Choice tasks.

But below we outline how to go about creating a new 'toolbox' for your experimental domain of interest.


### Step 1: Define your design space

First we create a pandas dataframe called `designs` using a function we write to do this. Each column is a design variable. Each row is a particular design.

```python
def build_my_design_space(my_arguments):
    designs = # CREATE PANDAS DATAFRAME OF THE DESIGN SPACE HERE
    return designs
```

### Step 2: Define a model

You must provide a model class which inherits from `Model`. You must also provide the following methods:

- `__init__`
- `predictive_y`

Here is an example of a minimal implimentation of a user-defined model:

```python
from badapted.model import Model
from badapted.choice_functions import CumulativeNormalChoiceFunc
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
        '''
        self.n_particles = int(n_particles)
        self.prior = prior
        self.θ_fixed = {'ϵ': 0.01}
        self.choiceFunction = CumulativeNormalChoiceFunc

    def predictive_y(self, θ, data):
        '''
        INPUTS:
        - θ = pandas dataframe of parameters
        - data = pandas dataframe of designs

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

### Step 3: You are done! Now use the code

There are probably 2 main ways that someone might want to use the code:

#### Using the code to run actual experiments

The vast majority of people will want to use a custom experiment toolbox to run adpative experiments to collect data. We have currently focussed on using the adaptive methods in PsychoPy, but there is no real reason why it could not be used in any other Python-based experiment software.

Probably the best way to get your adaptive experiment toolkit up and running is to look at example PsychoPy experiments. See the section below "Toolboxes using `badapted`" to go and find examples.

#### Using the code to simulate adaptive experiments

If you are a computational modeller or similar, then you might want to test how identifiable some model parameters are for a given design space, for example. If so, we provide some useful functions that we have used the development of this software.

The code below shows an example of how to run a simulated experiment

This is pretty straight-forward and there doesn't need to be any major customisation here.

```python
from badapted.parameter_recovery import simulated_experiment_trial_loop


# Build your design space
designs = build_my_design_space(my_arguments)

# Create a design generator using that design space
design_generator = MyCustomDesignGenerator(designs, max_trials=max_trials)

# Create a model object
model = MyCustomModel()

# Run a simulated experiment
model, design_generator= simulated_experiment_trial_loop(design_generator, model, max_trials)
```

## Toolboxes using `badapted`
- [DARC Toolbox](https://github.com/drbenvincent/darc_toolbox) for adpative Delayed and Risky Choice tasks.
- [Adaptive Psychophysics Toolbox](https://github.com/drbenvincent/adaptive_psychophysics_toolbox) for psychophysical experiments. _[Note: still under active development.]_
