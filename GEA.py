import numpy as np
import random
import operator

FUNCTIONS = {
    'add': operator.add,
    'sub': operator.sub,
    'mul': operator.mul,
    'div': lambda x, y: x / y if y != 0 else 1
}

FUNCTION_SET = list(FUNCTIONS.keys())

def generate_data(samples=500, features=5):
    X = np.random.uniform(-1, 1, size=(samples, features))
    y = (X[:, 0] * X[:, 1] - X[:, 2] + 0.5 > 0).astype(int)
    return X, y

X, y = generate_data()
TERMINAL_SET = ['x' + str(i) for i in range(X.shape[1])]

split = int(0.7 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

POP_SIZE = 50
CHROMOSOME_LENGTH = 15
NUM_GENERATIONS = 30
MUTATION_RATE = 0.1

def random_gene():
    return random.choice(FUNCTION_SET + TERMINAL_SET)

def generate_chromosome():
    return [random_gene() for _ in range(CHROMOSOME_LENGTH)]

def evaluate_chromosome(chromosome, input_vars):
    stack = []
    for gene in chromosome:
        if gene in FUNCTION_SET:
            if len(stack) < 2:
                stack.append(1)
                continue
            b = stack.pop()
            a = stack.pop()
            stack.append(FUNCTIONS[gene](a, b))
        else:
            idx = int(gene[1:])
            stack.append(input_vars[idx])
    return stack[0] if stack else 0

def accuracy(y_true, y_pred):
    correct = sum(int(yt == yp) for yt, yp in zip(y_true, y_pred))
    return correct / len(y_true)

def fitness(chromosome):
    preds = []
    for xi in X_train:
        val = evaluate_chromosome(chromosome, xi)
        preds.append(1 if val > 0 else 0)
    return accuracy(y_train, preds)

def mutate(chromosome):
    return [random_gene() if random.random() < MUTATION_RATE else gene for gene in chromosome]

population = [generate_chromosome() for _ in range(POP_SIZE)]

for gen in range(NUM_GENERATIONS):
    population = sorted(population, key=fitness, reverse=True)
    next_gen = population[:5]  

    while len(next_gen) < POP_SIZE:
        parent = random.choice(population[:25])
        child = mutate(parent)
        next_gen.append(child)

    population = next_gen

best_chromosome = population[0]
y_pred = [1 if evaluate_chromosome(best_chromosome, xi) > 0 else 0 for xi in X_test]
test_acc = accuracy(y_test, y_pred)

print("Best Chromosome:", best_chromosome)
print("Test Accuracy:", test_acc * 100, "%")
