import numpy as np
import time
import os

class TrafficPCA:
    def __init__(self, road_length=50, car_density=0.3, v_max=5, p_slow=0.2, seed=None):
        self.road_length = road_length
        self.v_max = v_max
        self.p_slow = p_slow
        np.random.seed(seed)

        # Initialize road with -1 (empty); otherwise store car's speed
        self.road = -np.ones(road_length, dtype=int)
        num_cars = int(car_density * road_length)
        car_positions = np.random.choice(road_length, size=num_cars, replace=False)

        for pos in car_positions:
            self.road[pos] = np.random.randint(0, v_max + 1)

    def step(self):
        new_road = -np.ones_like(self.road)
        for i in range(self.road_length):
            speed = self.road[i]
            if speed >= 0:
                speed = min(speed + 1, self.v_max)

                distance = 1
                while distance <= speed and self.road[(i + distance) % self.road_length] == -1:
                    distance += 1
                gap = distance - 1
                speed = min(speed, gap)

                if speed > 0 and np.random.rand() < self.p_slow:
                    speed -= 1

                new_pos = (i + speed) % self.road_length
                new_road[new_pos] = speed
        self.road = new_road

    def display(self):
        line = ''.join('c' if speed >= 0 else '.' for speed in self.road)
        print(line)

    def run_simulation(self, steps=50, delay=0.1):
        for t in range(steps):
            os.system('cls' if os.name == 'nt' else 'clear')  # Clear screen for animation effect
            print(f"Step {t+1}")
            self.display()
            self.step()
            time.sleep(delay)


if __name__ == "__main__":
    sim = TrafficPCA(
        road_length=10,
        car_density=0.3,
        v_max=5,
        p_slow=0.2,
        seed=42
    )

    sim.run_simulation(steps=10, delay=0.1)
