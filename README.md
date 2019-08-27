# BADAPTED: Bayesian ADAPTive Experimental Design

[![PyPI](https://img.shields.io/pypi/v/badapted.svg?color=green)](https://pypi.org/project/badapted/)

**Status:  🔥 Under active development 🔥**

Run efficient Bayesian adaptive experiments.

This code relates to the following pre-print. But, the pre-print is likely to appear in quite a different form when finally published.
> Vincent, B. T., & Rainforth, T. (2017, October 20). The DARC Toolbox: automated, flexible, and efficient delayed and risky choice experiments using Bayesian adaptive design. Retrieved from psyarxiv.com/yehjb


## Guide to using `badapted`

Below we outline how the `badapted` package can be used to run adaptive experiments. On it's own, this `badapted` package will not do anything. It also requires a few classse that a developer must create for their particular experimental paradigm.

### Experiment trial loop

```python
def run_an_experiment(design_thing, model, max_trials):
    '''Run an adaptive experiment
    INPUTS:
    - design_thing: a class
    '''

    for trial in range(max_trials):

        design = design_thing.get_next_design(model)

        if design is None:
            break

        response = get_response(design)

        design_thing.enter_trial_design_and_response(design, response)

        model.update_beliefs(design_thing.get_df())

    return model
```

In order to run this we need some setup code first

```python
# SETUP CODE HERE !!!!!!!!!!!!!!!!!!!!!!!

model = run_an_experiment(design_thing, model, max_trials)
```

### Provide a model
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
        - prior (dictionary). The keys provide the parameter name. The values must be scipy.stats objects which define the prior distribution for this parameter.

        We provide choice functions in `badapted.choice_functions.py`. In this example, we define it in the __init__ but it is not necessary to happen here.
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

### Provide a design generator

Building your own _Bayesian adaptive_ design generator is not necessarily going to be easy. We recommend that you look at the example in the darc toolbox (https://github.com/drbenvincent/darc_toolbox) to get an idea what is going on.

However, it is much easier to create your own custom _heuristic_ adaptive design generator is much easier. Again, we suggest looking at examples from the darc toolbox (https://github.com/drbenvincent/darc_toolbox).

Nevertheless, the information below is an attempt to outline the core requirements of this class... You must provide a class which deals with designs that inherits from `DesignGeneratorABC`.  You must provide a single method `get_next_design()`.

Here is an exmaple of a minimal user-defined design generator.

```python
from badapted.designs import DesignGeneratorABC
import pandas as pd
import numpy as np

class MyCustomDesignGenerator(DesignGeneratorABC):
    '''
    A custom design generator.
    It must subclass `DesignGeneratorABC` from badapted.designs.py
    You must impliment the method `get_next_design`
    '''

    def __init__(self, max_trial=20):
        '''Do whatever setup you need to do here, such as creating class variables etc'''
        self.trial = 0
        self.max_trials = max_trial
        # call the superclass constructor
        super().__init__()

    def get_next_design(self, model):
        """Get the next design.

        INPUT:
        - `model` is an optional input.

        OUTPUT:
        - return the next design. This can be in whatever form you want, but it might be useful to define a namped tuple which is intuitive for your problem domain and return that.
        """

        if self.trial >= self.max_trials:
            return None

        # You can use this method from the base class if it is useful
        last_response_chose_B = self.get_last_response_chose_B()

        # *** YOUR CODE TO PROVIDE A DESIGN HERE ***

        return design
```
