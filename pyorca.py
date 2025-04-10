"""Implementation of the 2D ORCA algorithm as described by J. van der Berg,
S. J. Guy, M. Lin and D. Manocha in 'Reciprocal n-body Collision Avoidance'."""

from __future__ import division

import numpy
from numpy import array, sqrt, copysign, dot
from numpy.linalg import det

from halfplaneintersect import halfplane_optimize, Line, perp

# Method:
# For each robot A and potentially colliding robot B, compute smallest change
# in relative velocity 'u' that avoids collision. Find normal 'n' to VO at that
# point.
# For each such velocity 'u' and normal 'n', find half-plane as defined in (6).
# Intersect half-planes and pick velocity closest to A's preferred velocity.

class Agent(object):
    """A disk-shaped agent."""
    def __init__(self, position, velocity, radius, max_speed, goal=None):
        super(Agent, self).__init__()
        self.position = array(position)
        self.velocity = array(velocity)
        self.radius = radius
        self.max_speed = max_speed
        self.goal = array(goal) if goal is not None else array(position)  # Goal as numpy array
        self.pref_velocity = array([0., 0.])  # Preferred velocity, updated by orca()

# def orca(agent, colliding_agents, t, dt):
#     """Compute ORCA solution for agent. NOTE: velocity must be _instantly_
#     changed on tick *edge*, like first-order integration, otherwise the method
#     undercompensates and you will still risk colliding."""
#     # Compute preferred velocity toward goal
#     dx = agent.goal - agent.position
#     dist = sqrt(dot(dx, dx))
#     if dist > 0.1:  # Move if more than 0.1m from goal
#         speed = min(agent.max_speed, dist / dt)  # Cap speed to max_speed or dist/dt
#         agent.pref_velocity = (speed * dx / dist)  # Normalize and scale direction
#     else:
#         agent.pref_velocity = array([0., 0.])  # Stop near goal

#     # Compute ORCA lines for collision avoidance
#     lines = []
#     for collider in colliding_agents:
#         dv, n = get_avoidance_velocity(agent, collider, t, dt)
#         line = Line(agent.velocity + dv / 2, n)
#         lines.append(line)
    
#     # Optimize to find new velocity closest to preferred velocity
#     new_velocity = halfplane_optimize(lines, agent.pref_velocity)
#     return new_velocity, lines

def orca(agent, colliding_agents, t, dt):
    """Compute ORCA solution for agent."""
    # Compute preferred velocity toward goal
    dx = agent.goal - agent.position
    dist = sqrt(dot(dx, dx))
    if dist > 0.1:  # Move if more than 0.1m from goal
        speed = min(agent.max_speed, dist / dt)
        agent.pref_velocity = (speed * dx / dist)
        print(f"Agent at {agent.position} heading to {agent.goal}, pref_velocity: {agent.pref_velocity}")
    else:
        agent.pref_velocity = array([0., 0.])
        print(f"Agent at {agent.position} reached goal {agent.goal}")

    # Compute ORCA lines for collision avoidance
    lines = []
    for collider in colliding_agents:
        dv, n = get_avoidance_velocity(agent, collider, t, dt)
        line = Line(agent.velocity + dv / 2, n)
        lines.append(line)
    
    new_velocity = halfplane_optimize(lines, agent.pref_velocity)
    return new_velocity, lines

def get_avoidance_velocity(agent, collider, t, dt):
    """Get the smallest relative change in velocity between agent and collider
    that will get them onto the boundary of each other's velocity obstacle
    (VO), and thus avert collision."""

    x = -(agent.position - collider.position)
    v = agent.velocity - collider.velocity
    r = agent.radius + collider.radius

    x_len_sq = norm_sq(x)

    if x_len_sq >= r * r:
        adjusted_center = x/t * (1 - (r*r)/x_len_sq)
        if dot(v - adjusted_center, adjusted_center) < 0:
            w = v - x/t
            u = normalized(w) * r/t - w
            n = normalized(w)
        else:
            leg_len = sqrt(x_len_sq - r*r)
            sine = copysign(r, det((v, x)))
            rot = array(
                ((leg_len, sine),
                (-sine, leg_len)))
            rotated_x = rot.dot(x) / x_len_sq
            n = perp(rotated_x)
            if sine < 0:
                n = -n
            u = rotated_x * dot(v, rotated_x) - v
    else:
        w = v - x/dt
        u = normalized(w) * r/dt - w
        n = normalized(w)
    return u, n

def norm_sq(x):
    return dot(x, x)

def normalized(x):
    l = norm_sq(x)
    assert l > 0, (x, l)
    return x / sqrt(l)

def dist_sq(a, b):
    return norm_sq(b - a)