# Snake-GA
Snake GA - Genetic algorithm that solves the Snake game. GA was implemented by [PyGAD](https://pygad.readthedocs.io/en/latest/#) library available for Python, neural networks was created in [Keras](https://keras.io/) and game was created in [Pygame](https://www.pygame.org/news).

All necessary requirements are stored in requirements.txt



# Optimization
If you want to optimize params, you need to change OPTIMIZATION = True

![Example](https://github.com/petomuro/Snake-GA/blob/main/Training_final.png)

# Test model
If you want to test trained neural network you need to change:
  1. OPTIMIZATION = False
  2. in neural_network.py self.network.load_weights('weights/model_name.h5') e.g. self.network.load_weights('weights/model20210526044312.h5')

# Run
You have to installed [Python](https://www.python.org/) language and other requirements which are necessary for run the program (requirements.txt).

Run main.py
