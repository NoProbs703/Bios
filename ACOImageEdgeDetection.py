import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

IMAGE_PATH = r"C:\Users\student\Pictures\NatureFlower.jpg"   
IMAGE_SIZE = (256, 256)

NUM_ANTS = 500
NUM_ITERATIONS = 100
ANT_LIFE = 20

ALPHA = 1.0    
BETA = 2.0     
RHO = 0.1     
EDGE_THRESHOLD = 0.4

if not os.path.exists(IMAGE_PATH):
    raise ValueError(f"Image not found: {IMAGE_PATH}")

img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, IMAGE_SIZE)
img = img.astype(np.float32) / 255.0

height, width = img.shape

gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
heuristic = np.sqrt(gx**2 + gy**2)
heuristic += 1e-6  

pheromone = np.ones((height, width), dtype=np.float32) * 0.1

for iteration in range(NUM_ITERATIONS):
    pheromone *= (1.0 - RHO)

    for ant in range(NUM_ANTS):
        x = np.random.randint(1, height - 1)
        y = np.random.randint(1, width - 1)

        for step in range(ANT_LIFE):
            neighbors = []
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if i == 0 and j == 0:
                        continue
                    nx, ny = x + i, y + j
                    if 0 <= nx < height and 0 <= ny < width:
                        neighbors.append((nx, ny))

            if not neighbors:
                break  

            probs = np.zeros(len(neighbors), dtype=np.float32)
            for i, (nx, ny) in enumerate(neighbors):
                probs[i] = (pheromone[nx, ny] ** ALPHA) * (heuristic[nx, ny] ** BETA)

            prob_sum = probs.sum()
            if prob_sum == 0:
                break

            probs /= prob_sum
            idx = np.random.choice(len(neighbors), p=probs)

            x, y = neighbors[idx]
            pheromone[x, y] += heuristic[x, y]

pheromone_norm = pheromone / pheromone.max()
edges = (pheromone_norm > EDGE_THRESHOLD).astype(np.uint8) * 255

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Pheromone Map")
plt.imshow(pheromone_norm, cmap="hot")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("ACO Edge Detection")
plt.imshow(edges, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
