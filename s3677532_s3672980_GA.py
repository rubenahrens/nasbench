# Implement a Genetic Algorithm (GA) using IOHexperimenter
import sys, time
from absl import app
from nasbench import api
import numpy as np
import nas_ioh


np.random.seed(42)


INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

#     Adjacency matrix           "Operations"
# "0101011010110101010101"    +    "12112"
# 𝑥𝑖 ∈ {0, 1}, 𝑖 ∈ [1..21]        𝑥𝑖 ∈ {0, 1, 2}, 𝑖 ∈ [22..26]


def gen_to_phen(x):

    x = [int(i) for i in x]
    
    connections_matrix = np.zeros((7,7))
    connections_matrix[np.triu_indices(7, 1)] = x[:21]
    
    operation_types = {0 : 'conv1x1-bn-relu', 1 : 'conv3x3-bn-relu', 2 : 'maxpool3x3'}

    spec = api.ModelSpec(
        matrix=connections_matrix,
        ops=['input'] + [operation_types[operation] for operation in x[21:]] + ['output']
    )

    return spec


def create_random_specs(population_size):
    
    population = []
    for _ in range(population_size):
        while True:
            connections_matrix = np.random.randint(2, size=(7, 7))
            connections_matrix = np.triu(connections_matrix, 1)

            connections_vector = connections_matrix[np.triu_indices(7, 1)]

            operations = [CONV1X1,CONV3X3,MAXPOOL3X3]
            choices = np.random.randint(3, size=5)
            ops = [INPUT] + [operations[i] for i in choices] + [OUTPUT]

            model_spec = api.ModelSpec(
                matrix=connections_matrix,
                ops=ops
            )

            if nas_ioh.nasbench.is_valid(model_spec):
                gen = np.append(connections_vector, choices)
                population.append(gen)
                break

    return population


def generate_population(population_size, chromosome_length):
    population = []
    for i in range(population_size):
        NAS_spec = []
        for j in range(chromosome_length):
            NAS_spec.append(np.random.randint(0, 2))
        population.append(NAS_spec)
    return population


# TODO: tournament selection
def roulette_wheel_selection(population, fitness_pop, mu):
    """Select mu parents from a population using roulette wheel selection."""
    
    norm_fitness = fitness_pop/np.sum(fitness_pop)
    selected_parents = np.random.choice(list(range(len(population))), p=norm_fitness, size=mu)

    return [population[i] for i in selected_parents]


def mutate(population, m_rate):
    """Mutate a population (spec) with a given mutation rate."""
    flip_bit = {0 : 1, 1 : 0}
    for i in range(len(population)):
        while True:
            agent_tmp = population[i].copy()
            for j in range(len(agent_tmp)):
                if np.random.random() < m_rate:
                    if j < 21:
                        agent_tmp[j] = flip_bit[agent_tmp[j]]
                    else:
                        choices = [0,1,2]
                        choices.remove(agent_tmp[j])
                        agent_tmp[j] = np.random.choice(choices)
            spec = gen_to_phen(agent_tmp)
            if nas_ioh.nasbench.is_valid(spec):
                population[i] = agent_tmp
                break
    return population

# ! GEWELDIG! We kunnen nu experimenteren met c_rate en drie crossover methodes & try jit
def crossover(population, n=1, c_rate=0.9):
    """Crossover two parents to create two children."""
    for i in range(0, len(population), 2):
        if np.random.random() < c_rate:
            child1 = population[i].copy()
            child2 = population[i+1].copy()
            while True:
                # uniform crossover
                if n == None:
                    for j in range(len(population[i])):
                        if np.random.random() < 0.5:
                            child1[j] = population[i+1][j]
                            child2[j] = population[i][j]
                # 1-point crossover
                elif n == 1:
                    point = np.random.randint(0, len(population[i]))
                    child1[:point] = population[i+1][:point]
                    child2[:point] = population[i][:point]
                # n-point crossover
                elif n > 1:
                    points = np.random.randint(0, len(population[i]), size=n)
                    points = sorted(points)
                    for j in range(0,len(points),1):
                        if j % 2 == 0:
                            if j == len(points) - 1:
                                point2 = len(points)
                            else:
                                point2 = points[j+1]
                            child1[points[j]:point2] = population[i+1][points[j]:point2]
                            child2[points[j]:point2] = population[i][points[j]:point2]
                spec1 = gen_to_phen(child1)
                if nas_ioh.nasbench.is_valid(spec1):
                    spec2 = gen_to_phen(child2)
                    if nas_ioh.nasbench.is_valid(spec2):
                        population[i], population[i+1] = child1, child2
                        break
    return population


def GA_algorithm():
    """Run the GA algorithm. """
    population_size = 50
    l = 26
    m_rate = 1/l
    c_rate = 0.75
    mu = 50
    n = 1

    # Initialize the population
    population = create_random_specs(population_size)

    # Evaluate: determine fitness for all individuals
    fitness_pop = np.array([nas_ioh.f(agent) for agent in population])

    # for _ in range((nas_ioh.budget - len(population)) // len(population)):
    for _ in range(nas_ioh.budget):

        # Mating selection: choose new parents proportional to fitness
        population = roulette_wheel_selection(population, fitness_pop, mu)

        # Crossover: mix parents genes inplace with a random chance c_rate
        population = crossover(population, n, c_rate)

        # Mutate: alter every gene with a probability = 1/l
        population = mutate(population, m_rate)
        
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
        x_best, f_best = GA_algorithm()
        print("run ", r, ", best x:", x_best,", f :",f_best)
        nas_ioh.f.reset()

if __name__ == '__main__':
    start = time.time()
    app.run(main)
    end = time.time()
    print("The program takes %s seconds" % (end-start))