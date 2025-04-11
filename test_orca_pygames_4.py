import pygame
import sys
import numpy as np
import cv2  # Import OpenCV
from ORCA import Orca

pygame.init()
WIDTH = 800
HEIGHT = 600
CELL_SIZE = 50
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("ORCA Simulation with Four Agents")

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

orca = Orca(dt=0.05, tau=5.0)

# Red Agent: Left to Right
red_agent = orca.add_agent(
    position=[2.0, 5.0],
    velocity=[0.0, 0.0],
    radius=0.3,
    max_speed=0.3,
    goal=[12.0, 5.0]
)
orca.set_waypoints(red_agent, [])

# Green Agent: Right to Left
green_agent = orca.add_agent(
    position=[12.0, 5.0],
    velocity=[0.0, 0.0],
    radius=0.3,
    max_speed=0.3,
    goal=[2.0, 5.0]
)
orca.set_waypoints(green_agent, [])

# Blue Agent: Bottom to Top
blue_agent = orca.add_agent(
    position=[7.0, 2.0],
    velocity=[0.0, 0.0],
    radius=0.3,
    max_speed=0.3,
    goal=[7.0, 8.0]
)
orca.set_waypoints(blue_agent, [])

# Yellow Agent: Top to Bottom
yellow_agent = orca.add_agent(
    position=[7.0, 8.0],
    velocity=[0.0, 0.0],
    radius=0.3,
    max_speed=0.3,
    goal=[7.0, 2.0]
)
orca.set_waypoints(yellow_agent, [])

# Set up OpenCV video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/home/tyler/dev/MRN_project/ORCA/recordings/ORCA_4agents.mp4', fourcc, 60.0, (WIDTH, HEIGHT))

clock = pygame.time.Clock()
running = True
frame_count = 0
max_frames = 1200  # Record for 300 frames (5 seconds at 60 FPS)

while running and frame_count < max_frames:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Predict positions one step ahead for collision detection
    agents = orca.get_agents()
    future_positions = []
    for agent in agents:
        future_pos = agent.position + agent.velocity * orca.dt
        future_positions.append(future_pos)

    # Check for collisions
    collision_detected = False
    for i, agent in enumerate(agents):
        agent.collision_detected = False
        for j, collider in enumerate(agents):
            if i == j:
                continue
            dist = np.sqrt(np.sum((future_positions[i] - future_positions[j])**2))
            if dist < (agent.radius + collider.radius):
                agent.collision_detected = True
                collision_detected = True

    # Update ORCA simulation
    orca.step()

    # Get agent positions
    red_pos = agents[0].position
    green_pos = agents[1].position
    blue_pos = agents[2].position
    yellow_pos = agents[3].position

    # Print positions, headings, and collision status
    print(f"Red Agent: Pos={red_pos}, Heading={agents[0].desired_heading:.2f}, Speed={agents[0].current_speed:.2f}, Collision={agents[0].collision_detected}")
    print(f"Green Agent: Pos={green_pos}, Heading={agents[1].desired_heading:.2f}, Speed={agents[1].current_speed:.2f}, Collision={agents[1].collision_detected}")
    print(f"Blue Agent: Pos={blue_pos}, Heading={agents[2].desired_heading:.2f}, Speed={agents[2].current_speed:.2f}, Collision={agents[2].collision_detected}")
    print(f"Yellow Agent: Pos={yellow_pos}, Heading={agents[3].desired_heading:.2f}, Speed={agents[3].current_speed:.2f}, Collision={agents[3].collision_detected}")

    # Clear the screen
    screen.fill(WHITE)

    # Draw grid
    for x in range(0, WIDTH, CELL_SIZE):
        pygame.draw.line(screen, (200, 200, 200), (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, (200, 200, 200), (0, y), (WIDTH, y))

    # Draw agents
    pygame.draw.circle(screen, RED, (int(red_pos[0] * CELL_SIZE), int(red_pos[1] * CELL_SIZE)), int(red_agent.radius * CELL_SIZE))
    pygame.draw.circle(screen, GREEN, (int(green_pos[0] * CELL_SIZE), int(green_pos[1] * CELL_SIZE)), int(green_agent.radius * CELL_SIZE))
    pygame.draw.circle(screen, BLUE, (int(blue_pos[0] * CELL_SIZE), int(blue_pos[1] * CELL_SIZE)), int(blue_agent.radius * CELL_SIZE))
    pygame.draw.circle(screen, YELLOW, (int(yellow_pos[0] * CELL_SIZE), int(yellow_pos[1] * CELL_SIZE)), int(yellow_agent.radius * CELL_SIZE))

    # Draw goals
    pygame.draw.circle(screen, BLUE, (int(red_agent.goal[0] * CELL_SIZE), int(red_agent.goal[1] * CELL_SIZE)), 5)
    pygame.draw.circle(screen, BLUE, (int(green_agent.goal[0] * CELL_SIZE), int(green_agent.goal[1] * CELL_SIZE)), 5)
    pygame.draw.circle(screen, BLUE, (int(blue_agent.goal[0] * CELL_SIZE), int(blue_agent.goal[1] * CELL_SIZE)), 5)
    pygame.draw.circle(screen, BLUE, (int(yellow_agent.goal[0] * CELL_SIZE), int(yellow_agent.goal[1] * CELL_SIZE)), 5)

    # Capture the Pygame surface as a NumPy array
    frame = pygame.surfarray.array3d(screen)
    # Convert from Pygame's (width, height, channels) to OpenCV's (height, width, channels)
    frame = np.transpose(frame, (1, 0, 2))
    # Convert RGB to BGR (OpenCV uses BGR)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # Write the frame to the video file
    out.write(frame)

    pygame.display.flip()
    clock.tick(60)
    frame_count += 1

# Release the video writer and clean up
out.release()
pygame.quit()
sys.exit()