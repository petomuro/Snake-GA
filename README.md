# Snake-GA
Snake GA - Genetic algorithm that solves the Snake game. GA was implemented by [PyGAD](https://pygad.readthedocs.io/en/latest/#) library available for Python, neural networks was created in [Keras](https://keras.io/) and game was created in [Pygame](https://www.pygame.org/news).

You have to installed [Python 3.7.9](https://www.python.org/downloads/release/python-379/) and other requirements which are necessary for run the program (requirements.txt).

![Example](https://github.com/petomuro/Snake-GA/blob/main/Game.PNG)

## Optimization
If you want to optimize params, you need to: 
  1. change `OPTIMIZATION = True`
  2. run `main.py`

![Example](https://github.com/petomuro/Snake-GA/blob/main/Training_final.png)

## Test model
When the optimization is complete, a file will be created in the weights folder. 

If you want to test trained neural network you need to:
  1. change `OPTIMIZATION = False`
  2. change in `neural_network.py` --> `self.network.load_weights('weights/model_name.h5')` --> e.g. `self.network.load_weights('weights/model20210526044312.h5')`
  3. run `main.py`

## Create graph
When the optimization is complete, a file will be created in the results folder.

If you want to create graph you need to: 
  1. change in `helper.py` --> `with open('results/results_name.csv') as results_file:` --> e.g. `with open('results/results20210520223339.csv') as results_file:`
  2. run `helper.py`

## Task lists
- [ ] Refactor source code
