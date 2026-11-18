from typing import Tuple 
import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
import ioh
from ioh import get_problem, logger, ProblemClass

budget = 5000

# To make your results reproducible (not required by the assignment), you could set the random seed by
# `np.random.seed(some integer, e.g., 42)`

def studentnumber1_studentnumber2_GA(problem: ioh.problem.PBO, population_size: int = 100, mutation_rate: float = 0.01, crossover_rate: float = 0.7) -> None:
    # initial_pop = ... make sure you randomly create the first population
    # create an initial population by randomly initializing population size number of binary strings the length of problem dimension
    initial_pop = np.random.randint(0, 2, size=(population_size, problem.meta_data.n_variables))

    fitnesses = np.zeros(population_size)   # initialize an array storing fitness evaluations
    for i, individual in enumerate(initial_pop):
        f = problem(individual)   # evaluate each individual from the initial population
        fitnesses[i] = f

    population = initial_pop
    # `problem.state.evaluations` counts the number of function evaluation automatically,
    # which is incremented by 1 whenever you call `problem(x)`.
    # You could also maintain a counter of function evaluations if you prefer.
    while problem.state.evaluations < budget:

        # apply elitism, save the best individual from current population
        best_idx = np.argmax(fitnesses)
        elite = population[best_idx].copy()
        elite_f = fitnesses[best_idx]

        # initialize new empty generation
        new_population = np.empty_like(population)
        new_fitnesses = np.zeros(population_size)

        # please implement the mutation, crossover, selection here
        i = 0
        while i < population_size:   # generate children until new population is full
            # implement the tournament selection
            # pick 3 random parents and select the best of them
            # do it for two parents
            options_1 = np.random.randint(0, population_size, size=3)
            winner_idx = options_1[np.argmax([fitnesses[idx] for idx in options_1])]
            parent_1 = population[winner_idx]

            options_2 = np.random.randint(0, population_size, size=3)
            winner_idx = options_2[np.argmax([fitnesses[idx] for idx in options_2])]
            parent_2 = population[winner_idx]

            # implement the uniform crossover
            # create a mask based on crossover rate to decide which parent's bit gets used
            crossover_mask = np.random.rand(problem.meta_data.n_variables) < crossover_rate

            # create two children
            child_1 = np.where(crossover_mask, parent_1, parent_2)
            child_2 = np.where(crossover_mask, parent_2, parent_1)

            # apply mutation to the children
            mutation_mask_1 = np.random.rand(problem.meta_data.n_variables) < mutation_rate
            child_1 = np.bitwise_xor(child_1, mutation_mask_1)  # flip the bit when mutation mask is 1
            
            mutation_mask_2 = np.random.rand(problem.meta_data.n_variables) < mutation_rate
            child_2 = np.bitwise_xor(child_2, mutation_mask_2)  # flip the bit when mutation mask is 1
            
            # this is how you evaluate one solution `x`
            # f = problem(x)

            # evaluate children
            new_population[i] = child_1
            new_fitnesses[i] = problem(child_1)
            i += 1

            if i < population_size:
                new_population[i] = child_2
                new_fitnesses[i] = problem(child_2)
                i += 1

        # replace the first individual of the new generation with the elite
        new_population[0] = elite
        new_fitnesses[0] = elite_f

        population = new_population
        fitnesses = new_fitnesses


    # no return value needed
        
        pass


def create_problem(dimension: int, fid: int) -> Tuple[ioh.problem.PBO, ioh.logger.Analyzer]:
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="genetic_algorithm",  # name of your algorithm
        algorithm_info="Practical assignment of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    # this how you run your algorithm with 20 repetitions/independent run
    # create the LABS problem and the data logger
    F18, _logger = create_problem(dimension=50, fid=18)
    for run in range(20): 
        studentnumber1_studentnumber2_GA(F18)
        F18.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder

    # create the N-Queens problem and the data logger
    F23, _logger = create_problem(dimension=49, fid=23)
    for run in range(20): 
        studentnumber1_studentnumber2_GA(F23)
        F23.reset()
    _logger.close()

