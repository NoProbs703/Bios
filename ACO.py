import math
import random
import numpy as np

random.seed(42)
np.random.seed(42)


def euclidean_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def compute_distance_matrix(cities):
    n = len(cities)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean_distance(cities[i], cities[j])
            dist[i, j] = dist[j, i] = d
    return dist


def nearest_neighbor_tour(dist):
    n = len(dist)
    tour = [0]
    unvisited = set(range(1, n))
    while unvisited:
        last = tour[-1]
        next_city = min(unvisited, key=lambda x: dist[last, x])
        tour.append(next_city)
        unvisited.remove(next_city)
    return tour


def tour_length(tour, dist):
    L = 0.0
    for i in range(len(tour)):
        L += dist[tour[i], tour[(i + 1) % len(tour)]]
    return L


def aco_tsp(
    cities,
    n_ants=None,
    n_iter=200,
    alpha=1.0,
    beta=5.0,
    rho=0.5,
    Q=100.0,
    initial_pheromone=None,
    verbose=False,
):
    """
    Ant Colony Optimization for symmetric TSP.
    Returns: best_tour (list of city indices), best_length (float), history (list of best lengths)
    Parameters:
      cities: list of (x,y)
      n_ants: number of ants (defaults to number of cities)
      n_iter: iterations
      alpha: pheromone importance
      beta: heuristic (visibility) importance
      rho: evaporation rate
      Q: pheromone deposit factor
      initial_pheromone: if None, set from nearest-neighbour tour
    """
    n = len(cities)
    if n_ants is None:
        n_ants = n

    dist = compute_distance_matrix(cities)
    # visibility (eta) = 1/d; avoid division by zero on diagonal
    with np.errstate(divide="ignore"):
        visibility = 1.0 / (dist + np.eye(n))
    visibility[np.isinf(visibility)] = 0.0
    np.fill_diagonal(visibility, 0.0)

    if initial_pheromone is None:
        nn_tour = nearest_neighbor_tour(dist)
        initial_pheromone = n / tour_length(nn_tour, dist)

    pheromone = np.full((n, n), initial_pheromone, dtype=float)

    best_tour = None
    best_length = float("inf")
    history = []

    for iteration in range(n_iter):
        all_tours = []
        all_lengths = []

        for ant in range(n_ants):
            start = random.randrange(n)
            visited = [start]
            unvisited = set(range(n))
            unvisited.remove(start)
            current = start

            while unvisited:
                # compute transition probabilities
                denom = 0.0
                probs = []
                for j in unvisited:
                    tau = pheromone[current, j] ** alpha
                    eta = visibility[current, j] ** beta
                    value = tau * eta
                    probs.append((j, value))
                    denom += value

                if denom == 0.0:
                    next_city = random.choice(list(unvisited))
                else:
                    r = random.random() * denom
                    cumulative = 0.0
                    next_city = None
                    for j, val in probs:
                        cumulative += val
                        if r <= cumulative:
                            next_city = j
                            break
                    if next_city is None:
                        next_city = probs[-1][0]

                visited.append(next_city)
                unvisited.remove(next_city)
                current = next_city

            L = tour_length(visited, dist)
            all_tours.append(visited)
            all_lengths.append(L)

            if L < best_length:
                best_length = L
                best_tour = visited.copy()

        # evaporation
        pheromone *= (1.0 - rho)

        # deposit pheromone
        for tour, L in zip(all_tours, all_lengths):
            deposit = Q / L
            for i in range(n):
                a = tour[i]
                b = tour[(i + 1) % n]
                pheromone[a, b] += deposit
                pheromone[b, a] += deposit

        # prevent numeric underflow / stagnation
        np.clip(pheromone, 1e-12, None, out=pheromone)

        history.append(best_length)
        if verbose and ((iteration + 1) % max(1, (n_iter // 10)) == 0 or iteration == 0):
            print(f"Iter {iteration+1}/{n_iter} best_length = {best_length:.6f}")

    return best_tour, best_length, history


if __name__ == "__main__":
    # Example: random cities
    N_CITIES = 20
    cities = [(random.random() * 100, random.random() * 100) for _ in range(N_CITIES)]

    best_tour, best_len, history = aco_tsp(
        cities,
        n_ants=30,
        n_iter=200,
        alpha=1.0,
        beta=5.0,
        rho=0.5,
        Q=100.0,
        verbose=True,
    )

    print("Best tour length:", best_len)
    print("Best tour sequence:", best_tour)
