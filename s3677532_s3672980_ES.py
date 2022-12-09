# Implement an Evolution Strategy (ES) using IOHexperimenter
import sys, time
from absl import app
from nasbench import api
import numpy as np
import nas_ioh
from s3677532_s3672980_GA import create_random_specs, roulette_wheel_selection, crossover, mutate


np.random.seed(42)


INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'


def ES_algorithm():
    """Implement an Evolution Strategy (ES) using IOHexperimenter"""
    population_size = 50
    l = 26
    m_rate = 1/l
    mu = 50
    n = None

    # Initialize the population
    population = create_random_specs(population_size)

    # Evaluate: determine fitness for all individuals
    fitness_pop = np.array([nas_ioh.f(agent) for agent in population])
    
    for _ in range((nas_ioh.budget - population_size) % population_size):
    # for _ in range(nas_ioh.budget):

        # recombination: create new offspring by crossover
        population = recombine(population, fitness_pop)

        # Mutate: alter every gene with a probability = 1/l
        population = mutate(population, m_rate)
        
        # Selction: choose the best mu individuals from the population
        population = select(population, mu)
        
        # determine fitness for all individuals
        fitness_pop = [nas_ioh.f(agent) for agent in population]

    final_best_agent = population[np.argmax(fitness_pop)]
    
    # convert model spec back to bit string
    return final_best_agent, max(fitness_pop)


##################################### MAIN ######################################   

def main(argv):
    del argv
    for r in range(nas_ioh.runs):
        f_best = sys.float_info.min
        x, y = ES_algorithm()
        if y > f_best:
            f_best = y
            x_best = x
        print("run ", r, ", best x:", x_best,", f :",f_best)
        nas_ioh.f.reset()


if __name__ == '__main__':
    start = time.time()
    app.run(main)
    end = time.time()
    print("The program takes %s seconds" % (end-start))