from typing import List

import numpy as np
import GA as ga_module
from GA import create_problem, studentnumber1_studentnumber2_GA

budget = 100_000
base_seed = 42

hyperparameter_space = {
    "population_size": [20, 50, 100],
    "mutation_rate": [0.01, 0.02, 0.05],
    "crossover_rate": [0.5, 0.7, 0.9],
}


def tune_hyperparameters() -> List:
    best_score = float("inf")
    best_params = None
    F18, _logger18 = create_problem(dimension=50, fid=18)
    F23, _logger23 = create_problem(dimension=49, fid=23)

    n_settings = (len(hyperparameter_space["population_size"])
                  * len(hyperparameter_space["mutation_rate"])
                  * len(hyperparameter_space["crossover_rate"]))
    evals_per_run = budget // (n_settings * 2)
    used = 0
    records = []

    for pop_size in hyperparameter_space["population_size"]:
        for mutation_rate in hyperparameter_space["mutation_rate"]:
            for crossover_rate in hyperparameter_space["crossover_rate"]:
                ga_module.budget = evals_per_run - pop_size
                results = []
                for problem, seed in ((F18, base_seed), (F23, base_seed + 100)):
                    studentnumber1_studentnumber2_GA(problem, population_size=pop_size,
                                                     mutation_rate=mutation_rate,
                                                     crossover_rate=crossover_rate, seed=seed)
                    used += problem.state.evaluations
                    results.append(problem.state.current_best.y)
                    problem.reset()
                records.append(((pop_size, mutation_rate, crossover_rate), results))

    ranks = np.zeros(len(records))
    for j in range(2):
        f = np.array([r[1][j] for r in records])
        order = np.argsort(-f)
        r = np.empty(len(records))
        r[order] = np.arange(len(records))
        ranks += r / 2
    for (params, _), score in zip(records, ranks):
        if score < best_score:
            best_score = score
            best_params = list(params)

    _logger18.close()
    _logger23.close()
    ga_module.budget = 5000
    print(f"Evaluations used: {used}/{budget}, best average rank: {best_score}")
    return best_params


if __name__ == "__main__":
    population_size, mutation_rate, crossover_rate = tune_hyperparameters()
    print(population_size)
    print(mutation_rate)
    print(crossover_rate)
