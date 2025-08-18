import random
import math

cities = [
    (60, 200), (180, 200), (80, 180), (140, 180),
    (20, 160), (100, 160), (200, 160), (140, 140),
    (40, 120), (100, 120)
]

NUM_CITIES = len(cities)
POPULATION_SIZE = 100
GENERATIONS = 500
MUTATION_RATE = 0.01
TOURNAMENT_SIZE = 5

def distance(city1, city2):
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

def total_distance(route):
    dist = 0
    for i in range(NUM_CITIES):
        dist += distance(cities[route[i]], cities[route[(i + 1) % NUM_CITIES]])
    return dist

def fitness(route):
    return 1 / total_distance(route)

def create_route():
    route = list(range(NUM_CITIES))
    random.shuffle(route)
    return route

def initial_population():
    return [create_route() for _ in range(POPULATION_SIZE)]

def tournament_selection(population, fitnesses):
    tournament = random.sample(list(zip(population, fitnesses)), TOURNAMENT_SIZE)
    tournament.sort(key=lambda x: x[1], reverse=True)
    return tournament[0][0]

def ordered_crossover(parent1, parent2):
    start, end = sorted(random.sample(range(NUM_CITIES), 2))
    child = [None] * NUM_CITIES

    child[start:end + 1] = parent1[start:end + 1]

    pointer = 0
    for gene in parent2:
        if gene not in child:
            while child[pointer] is not None:
                pointer += 1
            child[pointer] = gene
    return child

def mutate(route):
    for i in range(NUM_CITIES):
        if random.random() < MUTATION_RATE:
            j = random.randint(0, NUM_CITIES - 1)
            route[i], route[j] = route[j], route[i]

def genetic_algorithm():
    population = initial_population()
    fitnesses = [fitness(route) for route in population]

    best_route = population[fitnesses.index(max(fitnesses))]
    best_fitness = max(fitnesses)

    for generation in range(GENERATIONS):
        new_population = []

        new_population.append(best_route)

        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            child = ordered_crossover(parent1, parent2)
            mutate(child)
            new_population.append(child)

        population = new_population
        fitnesses = [fitness(route) for route in population]

        current_best_fitness = max(fitnesses)
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_route = population[fitnesses.index(best_fitness)]
            
        if generation % 50 == 0 or generation == GENERATIONS - 1:
         print(f"Gen {generation+1}: Best Distance = {1/best_fitness:.2f}")

    print("\nBest route found:")
    print(best_route)
    print(f"Total distance: {1 / best_fitness:.2f}")

if __name__ == "__main__":
    genetic_algorithm()


Output: 
Gen 1: Best Distance = 560.08
Gen 51: Best Distance = 459.99
Gen 101: Best Distance = 459.99
Gen 151: Best Distance = 459.99
Gen 201: Best Distance = 459.99
Gen 251: Best Distance = 459.99
Gen 301: Best Distance = 459.99
Gen 351: Best Distance = 459.99
Gen 401: Best Distance = 459.99
Gen 451: Best Distance = 459.99
Gen 500: Best Distance = 459.99

Best route found:
[9, 7, 6, 1, 3, 5, 2, 0, 4, 8]
Total distance: 459.99
