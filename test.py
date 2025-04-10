from __future__ import division

import numpy as np
from pyorca import Agent, orca, normalized, perp
import pygame

# Define initial positions and goals for four agents crossing a square
# initial_positions = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)]
# goals = [(5.0, 5.0), (0.0, 5.0), (0.0, 0.0), (5.0, 0.0)]
initial_positions = [(0.0, 0.0), (10.2, 0.1), (10.0, 10.0), (0.1, 9.8)]  # Small perturbations
goals = [(10.0, 10.0), (0.0, 10.0), (0.0, 0.0), (10.0, 0.0)]  # Keep goals as-is for now

# Simulation parameters
RADIUS = 0.2  # Reduced from 0.4 to allow closer approaches
SPEED = 2.0   # Max speed (m/s)

# Initialize agents with goals
agents = []
for i, (pos, goal) in enumerate(zip(initial_positions, goals)):
    agent = Agent(pos, (0., 0.), RADIUS, SPEED, goal=goal)
    print(f"Agent {i+1} initialized at {pos} with goal {goal}")
    agents.append(agent)

# Colors for each agent
colors = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
]

# Pygame setup
pygame.init()
dim = (640, 480)
screen = pygame.display.set_mode(dim)
pygame.display.set_caption("pyorca with Goals: Four Agents Crossing a Square")

O = np.array(dim) / 2  # Screen position of origin
scale = 44  # Drawing scale (meters to pixels)
sim_center = np.array([5.1, 5.0])  # Center of the simulation space
window_center = np.array(dim) / 2  # Center of the window (320, 240)
O = window_center - sim_center * scale  # Origin to center the simulation

clock = pygame.time.Clock()
FPS = 20
dt = 1 / FPS
tau = 5.0  # Increased from 2.0 for better planning

# Drawing functions
def draw_agent(agent, color):
    pos = np.rint(agent.position * scale + O).astype(int)
    pygame.draw.circle(screen, color, pos, int(round(agent.radius * scale)), 0)

def draw_velocity(agent, color):
    start = np.rint(agent.position * scale + O).astype(int)
    end = np.rint((agent.position + agent.velocity) * scale + O).astype(int)
    pygame.draw.line(screen, color, start, end, 1)

# Main loop
running = True
accum = 0
paths = [[] for _ in agents]


while running:
    accum += clock.tick(FPS)

    # Update simulation at fixed time steps
    while accum >= dt * 1000:
        accum -= dt * 1000

        # Compute new velocities using ORCA
        new_vels = [None] * len(agents)
        for i, agent in enumerate(agents):
            candidates = agents[:i] + agents[i+1:]
            new_vels[i], _ = orca(agent, candidates, tau, dt)

        # Apply velocities and update positions
        all_done = True
        for i, agent in enumerate(agents):
            agent.velocity = new_vels[i]
            print(f"Agent {i+1} velocity after ORCA: {agent.velocity}, pref_velocity: {agent.pref_velocity}")
            agent.position += agent.velocity * dt
            paths[i].append(agent.position.copy())  # Store a copy of the position
            print(f"Agent {i+1} position: {agent.position}, goal: {agent.goal}")
            dist_to_goal = np.sqrt(np.sum((agent.position - agent.goal)**2))
            if dist_to_goal > 0.1:
                all_done = False

        # Stop if all agents are near their goals
        if all_done:
            print("All agents reached their goals!")
            running = False

    # Render
    screen.fill(pygame.Color(0, 0, 0))  # Black background

    for path, color in zip(paths, colors):
        if len(path) > 1:
            for i in range(len(path) - 1):
                start = np.rint(path[i] * scale + O).astype(int)
                end = np.rint(path[i+1] * scale + O).astype(int)
                pygame.draw.line(screen, color, start, end, 1)

    # Draw agents and their velocities
    for agent, color in zip(agents, colors):
        draw_agent(agent, color)
        draw_velocity(agent, (255, 255, 255))  # White velocity lines

    # Draw goals
    for goal, color in zip(goals, colors):
        goal_pos = np.rint(np.array(goal) * scale + O).astype(int)
        pygame.draw.circle(screen, color, goal_pos, 3, 1)  # Small circle for goal

    pygame.display.flip()

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()