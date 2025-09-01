import numpy as np

iris_data = np.array([
    [5.1,3.5,1.4,0.2,0],
    [4.9,3.0,1.4,0.2,0],
    [4.7,3.2,1.3,0.2,0],
    [6.0,2.2,4.0,1.0,1],
    [5.9,3.2,4.8,1.8,2],
    [6.3,2.7,4.9,1.8,2],
    [5.1,2.5,3.0,1.1,1],
    [5.7,2.8,4.1,1.3,1],
    [6.5,3.0,5.8,2.2,2],
    [6.3,2.9,5.6,1.8,2],
    [5.0,3.5,1.3,0.3,0],
    [6.7,3.0,5.2,2.3,2]
])

X = iris_data[:, :-1]
y = iris_data[:, -1].astype(int)
num_features = X.shape[1]
num_classes = len(set(y))
num_samples = X.shape[0]

def get_folds(X, y, k=3):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    fold_size = len(X) // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else len(X)
        val_idx = indices[start:end]
        train_idx = np.setdiff1d(indices, val_idx)
        folds.append((train_idx, val_idx))
    return folds

def simple_decision_stump(X, y):
    best_acc = 0
    best_feature = 0
    best_threshold = 0
    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            preds = X[:, feature] > threshold
            preds = preds.astype(int)
            majority = int(np.round(np.mean(y)))
            acc = np.mean(preds == y)
            acc_inv = np.mean(1 - preds == y)
            if acc > best_acc:
                best_acc = acc
                best_feature = feature
                best_threshold = threshold
            if acc_inv > best_acc:
                best_acc = acc_inv
                best_feature = feature
                best_threshold = threshold
    return best_acc

def fitness_function(binary_position):
    if np.sum(binary_position) == 0:
        return 1.0 
    
    selected_indices = np.where(binary_position == 1)[0]
    X_sel = X[:, selected_indices]
    folds = get_folds(X_sel, y, k=3)
    accs = []
    for train_idx, val_idx in folds:
        train_X, val_X = X_sel[train_idx], X_sel[val_idx]
        train_y, val_y = y[train_idx], y[val_idx]
        acc = simple_decision_stump(train_X, train_y)  
        accs.append(acc)
    return 1 - np.mean(accs) 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

num_particles = 20
max_iter = 30
w, c1, c2 = 0.7, 1.5, 1.5

positions = np.random.randint(0, 2, (num_particles, num_features))
velocities = np.random.uniform(-1, 1, (num_particles, num_features))
pbest_positions = np.copy(positions)
pbest_scores = np.array([fitness_function(p) for p in positions])
gbest_index = np.argmin(pbest_scores)
gbest_position = pbest_positions[gbest_index]

for t in range(max_iter):
    for i in range(num_particles):
        r1 = np.random.rand(num_features)
        r2 = np.random.rand(num_features)

        velocities[i] = (
            w * velocities[i]
            + c1 * r1 * (pbest_positions[i] - positions[i])
            + c2 * r2 * (gbest_position - positions[i])
        )
        probs = sigmoid(velocities[i])
        positions[i] = np.where(np.random.rand(num_features) < probs, 1, 0)

        score = fitness_function(positions[i])
        if score < pbest_scores[i]:
            pbest_scores[i] = score
            pbest_positions[i] = positions[i]

    gbest_index = np.argmin(pbest_scores)
    gbest_position = pbest_positions[gbest_index]

    if t % 5 == 0 or t == max_iter - 1:
        acc = 1 - pbest_scores[gbest_index]
        print(f"Iteration {t+1}/{max_iter} - Best Accuracy: {acc:.4f} - Features: {np.sum(gbest_position)}")

selected = np.where(gbest_position == 1)[0]
print("\nSelected feature indices:", selected)
print("Number of selected features:", len(selected))
print("Final estimated accuracy:", round(1 - pbest_scores[gbest_index], 4))
