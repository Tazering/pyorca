import numpy as np
import matplotlib.pyplot as plt
from pyorca import Agent, orca

# Define initial positions and goals
initial_positions = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)]
goals = [(5.0, 5.0), (0.0, 5.0), (0.0, 0.0), (5.0, 0.0)]

# Simulation parameters
dt = 0.1  # Time step (seconds)
tau = 1.0  # Time horizon for ORCA (seconds)
max_steps = 100  # Max simulation steps
radius = 0.4  # Agent radius (meters)
max_speed = 2.0  # Max speed (m/s)

# Initialize agents with goals
agents = [Agent(pos, (0.0, 0.0), radius, max_speed, goal=goal) 
          for pos, goal in zip(initial_positions, goals)]

# Store paths for visualization
paths = [[pos] for pos in initial_positions]

# Simulation loop
for step in range(max_steps):
    # Apply ORCA to compute collision-free velocities
    for i, agent in enumerate(agents):
        others = agents[:i] + agents[i+1:]
        new_vel, _ = orca(agent, others, tau, dt)
        agent.velocity = new_vel

    # Update positions
    all_done = True
    for i, agent in enumerate(agents):
        new_pos = agent.position + agent.velocity * dt
        agent.position = new_pos
        paths[i].append(tuple(new_pos))  # Convert to tuple for plotting
        dist_to_goal = np.sqrt(np.sum((new_pos - goals[i])**2))
        if dist_to_goal > 0.1:  # Continue if not at goal
            all_done = False

    # Stop if all agents are near their goals
    if all_done:
        print(f"Simulation completed in {step+1} steps")
        break

# Visualization
plt.figure(figsize=(8, 8))
colors = ['b', 'g', 'r', 'c']
for i, path in enumerate(paths):
    x, y = zip(*path)
    plt.plot(x, y, color=colors[i], label=f'Agent {i+1}')
    plt.plot(initial_positions[i][0], initial_positions[i][1], 'o', color=colors[i])
    plt.plot(goals[i][0], goals[i][1], 'x', color=colors[i])
plt.grid(True)
plt.xlim(-1, 6)
plt.ylim(-1, 6)
plt.legend()
plt.title("pyorca with Goals: Four Agents Crossing a Square")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.show()

# Check for collisions
min_dist = float('inf')
for i in range(len(agents)):
    for j in range(i+1, len(agents)):
        for t in range(len(paths[i])):
            dx = paths[i][t][0] - paths[j][t][0]
            dy = paths[i][t][1] - paths[j][t][1]
            dist = np.sqrt(dx**2 + dy**2)
            min_dist = min(min_dist, dist)
print(f"Minimum distance between agents: {min_dist:.2f}m")

# Check goal proximity
for i, path in enumerate(paths):
    final_pos = np.array(path[-1])
    goal = np.array(goals[i])
    dist = np.sqrt(np.sum((final_pos - goal)**2))
    print(f"Intelligent Agent {i+1} final distance to goal: {dist:.2f}m")