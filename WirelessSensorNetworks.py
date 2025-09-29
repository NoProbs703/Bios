import numpy as np

# Parameters
num_nodes = 50  # Number of sensor nodes
num_ch = 5      # Number of cluster heads
area_size = 100 # Size of the square deployment area (0 to 100)

# Random sensor node positions
nodes = np.random.uniform(0, area_size, (num_nodes, 2))

# Objective: minimize sum of distance of each node to nearest cluster head
def objective(cluster_heads_flat):
    cluster_heads = cluster_heads_flat.reshape((num_ch, 2))
    dist = np.linalg.norm(nodes[:, None, :] - cluster_heads[None, :, :], axis=2)  # shape (num_nodes, num_ch)
    min_dist = np.min(dist, axis=1)
    return np.sum(min_dist)

# Grey Wolf Optimizer for cluster head selection
class GreyWolfOptimizer:
    def __init__(self, obj_func, lb, ub, dim, num_wolves=10, max_iter=50):
        self.obj_func = obj_func
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.num_wolves = num_wolves
        self.max_iter = max_iter
        
        # Initialize wolves (solutions)
        self.positions = np.random.uniform(self.lb, self.ub, (self.num_wolves, self.dim))
        self.alpha_pos = np.zeros(self.dim)
        self.beta_pos = np.zeros(self.dim)
        self.delta_pos = np.zeros(self.dim)
        self.alpha_score = np.inf
        self.beta_score = np.inf
        self.delta_score = np.inf
    
    def optimize(self):
        for iteration in range(self.max_iter):
            for i in range(self.num_wolves):
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
                fitness = self.obj_func(self.positions[i])
                
                if fitness < self.alpha_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()
                    self.beta_score = self.alpha_score
                    self.beta_pos = self.alpha_pos.copy()
                    self.alpha_score = fitness
                    self.alpha_pos = self.positions[i].copy()
                elif fitness < self.beta_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()
                    self.beta_score = fitness
                    self.beta_pos = self.positions[i].copy()
                elif fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = self.positions[i].copy()
            
            a = 2 - iteration * (2 / self.max_iter)
            
            for i in range(self.num_wolves):
                for j in range(self.dim):
                    r1 = np.random.random()
                    r2 = np.random.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha_pos[j] - self.positions[i][j])
                    X1 = self.alpha_pos[j] - A1 * D_alpha

                    r1 = np.random.random()
                    r2 = np.random.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.beta_pos[j] - self.positions[i][j])
                    X2 = self.beta_pos[j] - A2 * D_beta

                    r1 = np.random.random()
                    r2 = np.random.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.delta_pos[j] - self.positions[i][j])
                    X3 = self.delta_pos[j] - A3 * D_delta

                    self.positions[i][j] = (X1 + X2 + X3) / 3
            
            # Print only at iteration 1 and every 10th iteration
            if iteration == 0 or (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1} | Best Fitness: {self.alpha_score:.4f}")
        
        return self.alpha_pos, self.alpha_score

# Set bounds for cluster heads coordinates (x and y between 0 and area_size)
dim = num_ch * 2
lb = np.zeros(dim)
ub = np.ones(dim) * area_size

# Run GWO
gwo = GreyWolfOptimizer(objective, lb, ub, dim, num_wolves=20, max_iter=100)
best_solution, best_fitness = gwo.optimize()

# Extract cluster heads positions
cluster_heads = best_solution.reshape((num_ch, 2))

print("\nOptimized Cluster Head Positions (x, y):")
for idx, ch in enumerate(cluster_heads):
    print(f"CH{idx + 1}: ({ch[0]:.2f}, {ch[1]:.2f})")

print(f"\nBest fitness (sum of distances): {best_fitness:.4f}")

# ASCII plot function for rough visualization
def ascii_plot(nodes, cluster_heads, size=10):  # changed default size to 10
    grid = [['.' for _ in range(size)] for _ in range(size)]

    def scale_point(pt):
        x = int(pt[0] / area_size * (size-1))
        y = int(pt[1] / area_size * (size-1))
        return x, y
    
    for node in nodes:
        x, y = scale_point(node)
        grid[y][x] = 'n'  # node
    
    for ch in cluster_heads:
        x, y = scale_point(ch)
        grid[y][x] = 'C'  # cluster head

    for row in reversed(grid):  # print y=0 at bottom
        print(''.join(row))

print("\nNetwork topology (C=Cluster Head, n=Node):")
ascii_plot(nodes, cluster_heads, size=10)
