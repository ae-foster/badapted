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

### Step 3: Put it all together

This is pretty straight-forward and there doesn't need to be any major customisation here.

```python
def run_experiment(design_generator, model, max_trials):
    '''Bare-bones experiment'''

    for trial in range(max_trials):
        design = design_generator.get_next_design(model)
        if design is None:
            break
        response = get_response(design)
        design_generator.enter_trial_design_and_response(design, response)
        model.update_beliefs(design_generator.data)

    return model

designs = build_my_design_space(my_arguments)
design_generator = MyCustomDesignGenerator(designs, max_trials=max_trials)
model = MyCustomModel()
model = run_experiment(design_generator, model, max_trials)
```

Note that use of the `run_experiment` function is just a demonstration of the logic of how things fit together. As mentioned, please refer to PsychoPy example experiments in the [DARC Toolbox](https://github.com/drbenvincent/darc_toolbox) to see how this all comes together in a PsychoPy experiment.

The `response = get_response(design)` part is up to you to impliment. What you do here depends on whether you are simulating responses or getting real responses from PsychoPy etc. The `run_experiment` function is just an example of how the various parts of the code work together. When running _actual_ experiments using PsychoPy, it is best to refer to the demo psychopy files we provide in the [DARC Toolbox](https://github.com/drbenvincent/darc_toolbox) as examples to see how this is done.


## Toolboxes using `badapted`
- [DARC Toolbox](https://github.com/drbenvincent/darc_toolbox) for adpative Delayed and Risky Choice tasks.
- [Adaptive Psychophysics Toolbox](https://github.com/drbenvincent/adaptive_psychophysics_toolbox) for psychophysical experiments. _[Note: still under active development.]_
