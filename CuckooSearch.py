import numpy as np
import math

values = np.array([60, 100, 120])       
weights = np.array([10, 20, 30])       
capacity = 50                          

n_items = len(values)
n_nests = 15                          
n_iterations = 100                   
pa = 0.25      

def fitness(solution):
    total_weight = np.sum(solution * weights)
    total_value = np.sum(solution * values)
    if total_weight > capacity:
        return 0  
    else:
        return total_value

def initialize_nests(n_nests, n_items):
    return np.random.randint(0, 2, size=(n_nests, n_items))

def levy_flight(nest, alpha=0.01):
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(np.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)

    u = np.random.normal(0, sigma, size=nest.shape)
    v = np.random.normal(0, 1, size=nest.shape)
    step = u / (np.abs(v) ** (1 / beta))

    stepsize = alpha * step * (nest - np.mean(nest))

    new_nest = np.copy(nest)
    flip_prob = np.abs(stepsize)
    for i in range(len(nest)):
        if np.random.rand() < flip_prob[i]:
            new_nest[i] = 1 - new_nest[i] 
    return new_nest

def get_best_nest(nests):
    fitnesses = np.array([fitness(n) for n in nests])
    best_idx = np.argmax(fitnesses)
    return nests[best_idx], fitnesses[best_idx]

def cuckoo_search():
    nests = initialize_nests(n_nests, n_items)
    best_nest, best_fitness = get_best_nest(nests)

    for iteration in range(n_iterations):

        new_nests = []
        for nest in nests:
            new_nest = levy_flight(nest)

            if fitness(new_nest) > fitness(nest):
                new_nests.append(new_nest)
            else:
                new_nests.append(nest)
        nests = np.array(new_nests)

        n_abandon = int(pa * n_nests)
        abandon_indices = np.random.choice(n_nests, n_abandon, replace=False)
        for idx in abandon_indices:
            nests[idx] = np.random.randint(0, 2, n_items)

        current_best_nest, current_best_fitness = get_best_nest(nests)
        if current_best_fitness > best_fitness:
            best_nest, best_fitness = current_best_nest, current_best_fitness

        if (iteration + 1) % 10 == 0 or iteration == 0:
            print(f"Iter {iteration+1}/{n_iterations} - Best fitness: {best_fitness}")

    return best_nest, best_fitness

if __name__ == "__main__":
    solution, fitness_value = cuckoo_search()
    print("\nBest solution found:")
    print("Selected items:", solution)
    print("Total value:", fitness_value)
    print("Total weight:", np.sum(solution * weights))
