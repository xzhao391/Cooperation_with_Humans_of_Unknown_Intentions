import numpy as np
from multiagent.core import World, Agent, Landmark, Arm
from multiagent.scenario import BaseScenario
from numpy import linalg as LA

class Scenario(BaseScenario):

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        world.num_agents = num_agents
        num_adversaries = 1
        num_landmarks = 2
        # add agents
        world.agents = [Arm(), Agent(), Agent()]
        world.episode = 0
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            if i > 0:
                agent.size = .8 if i < num_adversaries else .1

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        size_land = [.8, .6]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = size_land[i]
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        world.agents[0].color = np.array([1, 0, 0])
        world.agents[1].color = np.array([0.3, 0.9, 0.3])
        world.agents[2].color = np.array([0.6, 0.6, 0.9])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.15, 0.15, 0.15])
        # set goal landmark
        goal = [np.array([-.7, 0]), np.array([.6, -.2])]
        state = [np.array([.7, 0]), np.array([-.7, 0])]
        # set initial states
        for i, agent in enumerate(world.agents):
            if i == 0:
                agent.l1 = .3
                agent.l2 = .2
                agent.initial_mass = 2
                agent.state.center = np.array([0, .6])
                agent.state.p_pos = np.array([-np.pi/2, 0])
                agent.state.p_vel = np.array([0, 0])
                J1_pos, J2_pos = self.forward_kinematics(agent)
                agent.J1_pos = J1_pos
                agent.J2_pos = J2_pos
            else:
                agent.goal = goal[i- 1]
                agent.state.p_pos = state[i- 1]
                agent.state.p_vel = np.zeros(world.dim_p)
                if i > 0:
                    agent.state.past_traj = [agent.state.p_pos] * 4

        p_pos = np.array([[0, -1], [0, 1.2]])
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = p_pos[i]
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            return np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        else:
            dists = []
            for l in world.landmarks:
                dists.append(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
            dists.append(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
            return tuple(dists)

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        adversary = world.agents[0]
        adv_rew = 0
        Q = np.array([[1.5, 0], [0, .3]])

        if agent.name == 'agent 1':
            friend = world.agents[2]
            agent_diff = agent.goal - agent.state.p_pos
            friend_rew = -agent_diff@Q@agent_diff
            if LA.norm(agent_diff) < .1:
                friend_rew += 2
            if self.is_collision(agent, friend):
                adv_rew -= 15
            if self.adv_collision(adversary, agent):
                adv_rew -= 25
            if world.episode %2 != 0:   #human not yield
                friend_diff = friend.goal - friend.state.p_pos
                friend_rew -= 2*friend_diff@Q@friend_diff
                if LA.norm(friend_diff) < .1:
                    friend_rew += 2
                if self.is_collision(agent, friend):
                    adv_rew -= 20
        else:
            agent_diff = agent.goal - agent.state.p_pos
            friend_rew = -agent_diff@Q@agent_diff
            if LA.norm(agent_diff) < .2:
                friend_rew += 1
            if world.episode %2 == 0:   #human yield
                friend = world.agents[1]
                friend_diff = friend.goal - friend.state.p_pos
                friend_rew -= 2*friend_diff@Q@friend_diff
                if LA.norm(friend_diff) < .1:
                    friend_rew += 2
                if self.is_collision(agent, friend):
                    adv_rew -= 25
                if self.fake_obstacle(agent, np.array([0, .6])):
                    adv_rew -= 25

        for i in world.landmarks:
            if self.is_collision(agent, i):
                friend_rew -= 15
        return friend_rew + adv_rew

    def adversary_reward(self, agent, world):
        adv_rew = 0
        if self.adv_collision(agent, world.agents[1]):
            adv_rew += 10
        return adv_rew


    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            if agent.name == 'agent 0':
                entity_pos.append(entity.state.p_pos - agent.state.center)
            else:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            if agent.name == 'agent 0':
                other_pos.append(other.state.p_pos - agent.state.center)
            elif other.name == 'agent 0':
                other_pos.append(other.state.p_pos)
            else:
                if agent.name == 'agent 1' or agent.name == 'agent 2':
                    for i, p in enumerate(other.state.past_traj):
                        other_pos.append(p - agent.state.p_pos)
                else:
                    other_pos.append(other.state.p_pos - agent.state.p_pos)
        if agent.name == 'agent 0':
            return np.concatenate([agent.state.p_pos] + [agent.state.p_vel] + entity_pos + other_pos)
        elif agent.name == 'agent 2':
            return np.concatenate([agent.goal - agent.state.p_pos] + [agent.state.p_pos] + [agent.state.p_vel] + entity_pos + other_pos + [np.array([world.episode % 2])])
        else:
            return np.concatenate([agent.goal - agent.state.p_pos] + [agent.state.p_pos] + [agent.state.p_vel] + entity_pos + other_pos)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def fake_obstacle(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + .5
        return True if dist < dist_min else False

    def adv_collision(self, adv, circle):
        collision1 = self.lineCircle(np.array([0,1]), adv.J1_pos, circle.state.p_pos)
        collision2 = self.lineCircle(adv.J1_pos, adv.J2_pos, circle.state.p_pos)
        if collision1 or collision2:
            return True
        return False

    def lineCircle(self, p1, p2, circle):
        inside1 = self.pointCircle(p1, circle)
        inside2 = self.pointCircle(p2, circle)
        if (inside1 or inside2):
            return True
        distX = p1[0] - p2[0]
        distY = p1[1] - p2[1]
        len = np.sqrt((distX * distX) + (distY * distY))
        dot = (((circle[0] - p1[0]) * (p2[0] - p1[0])) + (circle[1] - p1[1]) * (p2[1] - p1[1])) / pow(len, 2)
        closestX = p1[0] + (dot * (p2[0] - p1[0]))
        closestY = p1[1] + (dot * (p2[1] - p1[1]))
        onSegment = self.linePoint(p1, p2, [closestX, closestY])
        if not onSegment:
            return False
        distX = closestX - circle[0]
        distY = closestY - circle[1]
        distance = np.sqrt((distX * distX) + (distY * distY))
        if distance <= .1:
            return True
        return False

    def pointCircle(self, p, circle):
        distX = p[0] - circle[0]
        distY = p[1] - circle[1]
        distance = np.sqrt((distX * distX) + (distY * distY))
        if (distance <=.11):
            return True
        return False

    def linePoint(self, p1, p2, pxy):
        d1 = self.dist(pxy, p1)
        d2 = self.dist(pxy, p2)
        lineLen = self.dist(p1, p2)
        buffer = 0.05
        if (d1 + d2 >= lineLen - buffer) and (d1 + d2 <= lineLen + buffer):
            return True

    def dist(self, p1, p2):
        distX = p1[0] - p2[0]
        distY = p1[1] - p2[1]
        distance = np.sqrt((distX * distX) + (distY * distY))
        return distance

    def forward_kinematics(self, agent):
        angle = agent.state.p_pos
        J1_pos = np.array([np.cos(angle[0]), np.sin(angle[0])]) * agent.l1 + agent.state.center
        J2_pos = np.array([np.cos(angle[0]+angle[1]), np.sin(angle[0]+angle[1])]) * agent.l2 + J1_pos
        return J1_pos, J2_pos

    def done(self, agent, world):
        if agent.name == 'agent 0':
            return True
        elif LA.norm(agent.state.p_pos-agent.goal, 2) <= .1:
            return True
        return False
