import numpy as np
from numpy import array, sqrt, copysign, dot
from numpy.linalg import det
from pyorca import Agent, orca
from halfplaneintersect import Line, halfplane_optimize, InfeasibleError

class Orca:
    def __init__(self, dt=0.05, tau=5.0, waypoint_interval=0.5):
        self.dt = dt
        self.tau = tau
        self.waypoint_interval = waypoint_interval
        self.agents = []

    def add_agent(self, position, velocity, radius, max_speed, goal=None):
        # Use pyorca.Agent to create a new agent
        agent = Agent(position, velocity, radius, max_speed, goal)
        self.agents.append(agent)
        return agent

    def remove_agent(self, agent):
        if agent in self.agents:
            self.agents.remove(agent)

    def get_agents(self):
        return self.agents

    def set_waypoints(self, agent, waypoints):
        # Since pyorca doesn't support waypoints natively, we'll handle simple goal-setting
        # For now, we'll treat the last waypoint as the goal
        if waypoints:
            agent.goal = np.array(waypoints[-1], dtype=float)
        else:
            agent.goal = np.array(agent.goal, dtype=float)  # Ensure goal is a numpy array

    def update_goals(self):
        # In pyorca, the orca function updates pref_velocity based on the goal
        # No additional goal updates needed here, but we can add waypoint logic later if needed
        pass


    def norm_sq(self, x):
        return np.dot(x, x)

    def normalized(self, x):
        l = self.norm_sq(x)
        assert l > 0, (x, l)
        return x / np.sqrt(l)

    def perp(self, v):
        return np.array([-v[1], v[0]])

    # from the github
    def get_avoidance_velocity(self, agent, collider):
        """Get the smallest relative change in velocity between agent and collider
        that will get them onto the boundary of each other's velocity obstacle
        (VO), and thus avert collision."""

        x = -(agent.position - collider.position)
        v = agent.velocity - collider.velocity
        r = agent.radius + collider.radius

        x_len_sq = self.norm_sq(x)

        if x_len_sq >= r * r:
            adjusted_center = x/self.tau * (1 - (r*r)/x_len_sq)
            if dot(v - adjusted_center, adjusted_center) < 0:
                w = v - x/self.tau
                u = self.normalized(w) * r/self.tau - w
                n = self.normalized(w)
            else:
                leg_len = sqrt(x_len_sq - r*r)
                sine = copysign(r, det((v, x)))
                rot = array(
                    ((leg_len, sine),
                    (-sine, leg_len)))
                rotated_x = rot.dot(x) / x_len_sq
                n = self.perp(rotated_x)
                if sine < 0:
                    n = -n
                u = rotated_x * dot(v, rotated_x) - v
        else:
            w = v - x/self.dt
            u = self.normalized(w) * r/self.dt - w
            n = self.normalized(w)
        return u, n


    # def compute_velocity(self, agent):
    #     dx = agent.goal - agent.position
    #     dist = np.sqrt(self.norm_sq(dx))

    #     # Compute preferred velocity toward goal
    #     if dist > 0.1:
    #         speed = min(agent.max_speed, dist / self.dt)
    #         agent.pref_velocity = (speed * dx / dist)
    #         print(f"Agent at {agent.position} heading to {agent.goal}, pref_velocity: {agent.pref_velocity}")

    #     else:
    #         agent.pref_velocity = np.array([0.0, 0.0])
    #         print(f"Agent at {agent.position} reached goal {agent.goal}")

    #     # Compute ORCA lines for collision avoidance
    #     lines = []
    #     for collider in self.agents:
    #         if collider is agent:
    #             continue
    #         dv, n = self.get_avoidance_velocity(agent, collider)
    #         line = Line(agent.velocity + dv/2, n)
    #         lines.append(line)

    #     try:
    #         new_velocity = halfplane_optimize(lines, agent.pref_velocity)
    #     except InfeasibleError:
    #         speed = np.sqrt(self.norm_sq(agent.pref_velocity))
    #         if speed > 0:
    #             new_velocity = (agent.pref_velocity / speed) * min(speed * 0.5, agent.max_speed)
    #         else:
    #             new_velocity = np.array([0.0, 0.0])

    #     new_speed = np.sqrt(np.sum(new_velocity**2))
    #     new_heading = np.arctan2(new_velocity[1], new_velocity[0]) if new_speed > 0.01 else getattr(agent, "desired_heading", 0.0)
        
    #     agent.desired_heading = new_heading
        
    #     return new_speed, new_heading


    # def modify_velocity(self, agent):
    #     if not hasattr(agent, "current_speed"):
    #         agent.current_speed = agent.max_speed
        
    #     if not hasattr(agent, "collision_detected"):
    #         agent.collision_detected = False
        
    #     if not agent.collision_detected:
    #         agent.current_speed = agent.max_speed
    #         return agent.current_speed

    #     mean_delta = -0.1
    #     std_dev = 0.075
    #     delta_v = np.random.normal(mean_delta, std_dev)
    #     new_speed = agent.current_speed + delta_v
    #     new_speed = max(0.05, min(new_speed, agent.max_speed))
    #     agent.current_speed = new_speed
    #     return new_speed
        

    def compute_velocity(self, agent):
        # Use pyorca's orca function to compute the new velocity
        # Pass the agent and all other agents as colliding agents
        colliding_agents = [a for a in self.agents if a is not agent]
        new_velocity, lines = orca(agent, colliding_agents, self.tau, self.dt)
        
        # Extract speed and heading from the new velocity
        new_speed = np.sqrt(np.sum(new_velocity**2))
        new_heading = np.arctan2(new_velocity[1], new_velocity[0]) if new_speed > 0.01 else getattr(agent, "desired_heading", 0.0)
        
        # Store the heading for use in updates
        agent.desired_heading = new_heading
        
        return new_speed, new_heading

    def modify_velocity(self, agent):
        # pyorca doesn't have a modify_velocity function
        # For now, return the current speed (we'll integrate this later if needed)
        # We can add speed adjustments similar to the original logs
        if not hasattr(agent, "current_speed"):
            agent.current_speed = agent.max_speed  # Initialize if not present
        
        if not hasattr(agent, "collision_detected"):
            agent.collision_detected = False
        
        if not agent.collision_detected:
            agent.current_speed = agent.max_speed
            return agent.current_speed

        # Simple speed reduction during collision (mimicking original behavior)
        mean_delta = -0.1
        std_dev = 0.075
        delta_v = np.random.normal(mean_delta, std_dev)
        new_speed = agent.current_speed + delta_v
        new_speed = max(0.05, min(new_speed, agent.max_speed))
        agent.current_speed = new_speed
        return new_speed

    def update_all_velocities(self):
        results = []
        for agent in self.get_agents():
            new_speed, new_heading = self.compute_velocity(agent)
            adjusted_speed = self.modify_velocity(agent)
            agent.velocity = adjusted_speed * np.array([np.cos(new_heading), np.sin(new_heading)])
            results.append((adjusted_speed, new_heading))
        return results

    def step(self):
        self.update_goals()
        self.update_all_velocities()
        for agent in self.get_agents():
            agent.position += agent.velocity * self.dt