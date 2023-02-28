import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class PoacherWorld(World):
    def __init__(self):
        super(PoacherWorld, self).__init__()
        # poacher world always has 3 agents
        self.poacher = Agent()
        self.ranger = Agent()
        self.uav = Agent()
        self.agents = [self.poacher, self.ranger, self.uav]
        for i, agent in enumerate(self.agents):
            agent.id = i
            agent.name = ['poacher', 'ranger', 'uav'][i]
            agent.size = [0.075, 0.075, 0.05][i]
            agent.accel = [2, 2.15, 3][i]
            agent.max_speed = [1, 1, 1.3][i]
            agent.sight = [0.3, 0.3, 0.5][i]
            agent.color = [[0.85, 0.35, 0.35],
                    [0.35, 0.35, 0.85], 
                    [0.35, 0.85, 0.35]][i]
            agent.poacher_caught_rew = [-1, 5.5, 5][i]
            agent.animal_caught_rew = [7, -1.5, -1][i]
            agent.caught = False

            # Implemented
            agent.both_observe_poacher_rew = [-0.1, 2, 2][i]
            agent.both_observe_nothing_rew = [0, -0.1, -0.1][i]
            agent.uav_observe_poacher_rew = [-0.1, 0, 2][i]
            agent.ranger_observe_poacher_rew = [-0.1, 2, 0][i]

        # variables that track what happened in this step
        self.was_poacher_caught = False
        self.was_animal_caught = False

        # Implemented
        self.was_poacher_observed_by_uav = False
        self.was_poacher_observed_by_ranger = False


    def step(self):
        super().step()

        # check if poacher caught
        self.was_poacher_caught = False
        if self.is_collision(self.ranger, self.poacher) and not self.poacher.caught:
            self.poacher.caught = True
            self.was_poacher_caught = True

        # check if any animal caught
        self.was_animal_caught = False
        for i, landmark in enumerate(self.landmarks):
            if self.is_collision(self.poacher, landmark):
                landmark.poacher_collision_time += 1
            else:
                landmark.poacher_collision_time = 0

            if landmark.poacher_collision_time >= 20 and not landmark.caught:
                landmark.caught = True
                self.was_animal_caught = True

        # Implemented
        self.was_poacher_observed_by_uav = False
        if self.is_visible(self.uav, self.poacher):
          self.was_poacher_observed_by_uav = True

        self.was_poacher_observed_by_ranger = False
        if self.is_visible(self.ranger, self.poacher):
          self.was_poacher_observed_by_ragner = True

    def is_collision(self, agent1, agent2):
        '''
        Return True if agent1 and agent2 are currently colliding.
        '''
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def is_visible(self, agent, target):
        '''
        Return True if target is within the agent's sight.
        '''
        delta_pos = agent.state.p_pos - target.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        return True if dist < agent.sight else False

class Scenario(BaseScenario):
    def make_world(self):
        world = PoacherWorld()
        world.dim_p = 2 # world dimension

        # create animals
        num_landmarks = 8
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'animal %d' % i
            landmark.size = 0.06
            landmark.poacher_collision_time = 0
            landmark.caught = False
            landmark.color = [0.5, 0.5, 0.5]

        self.reset_world(world)
        return world

    def reset_world(self, world):
        # set initial agent position and velocity
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p) # position
            agent.state.p_vel = np.zeros(world.dim_p) # velocity
            agent.caught = False
        # set animal positions
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.array([[0.647,0.268], # position
                    [0.268,0.647], 
                    [-0.268,0.647], 
                    [-0.647,0.268], 
                    [-0.647,-0.268], 
                    [-0.268,-0.647], 
                    [0.268,-0.647], 
                    [0.647,-0.268]][i])
            landmark.state.p_vel = np.zeros(world.dim_p) # velocity
            landmark.poacher_collision_time = 0
            landmark.caught = False
        world.was_poacher_caught = False # has the poacher been caught
        world.was_animal_caught = False # has any animal been caught

        # Implemented
        world.was_poacher_observed_by_uav = False
        world.was_poacher_observed_by_ranger = False

    def benchmark_data(self, agent, world):
        return self.reward(agent, world)

    def done(self, agent, world):
        '''
        Return 1 if the episode is over.
        '''
        if world.was_poacher_caught or world.was_animal_caught:
            return 1
        else:
            return 0

    def reward(self, agent, world):
        '''
        Return this agent's reward.
        '''
        if world.was_poacher_caught:
          return agent.poacher_caught_rew
        elif world.was_animal_caught:
          return agent.animal_caught_rew
          
        # Implemented
        elif world.was_poacher_observed_by_uav and world.was_poacher_observed_by_ranger:
          return agent.both_observe_poacher_rew
        elif world.was_poacher_observed_by_uav == False and world.was_poacher_observed_by_ranger:
          return agent.ranger_observe_poacher_rew
        elif world.was_poacher_observed_by_ranger and world.was_poacher_observed_by_ranger == False:
          return agent.uav_observe_poacher_rew
        elif world.was_poacher_observed_by_ranger == False and world.was_poacher_observed_by_uav == False:
          return agent.both_observe_nothing_rew
        else:
          return 0

    def is_visible(self, agent, target):
        '''
        Return True if target is within the agent's sight.
        '''
        delta_pos = agent.state.p_pos - target.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        return True if dist < agent.sight else False

    def is_observed(self, agent, target, world):
        '''
        Returns True if the target should be included in the agent observation.
        '''
        if agent.id == target.id:
            return True

        if agent.name == 'poacher': # poacher -> ranger/uav
            return self.is_visible(agent, target)
        elif target.name == 'poacher': # ranger/uav -> poacher 
            return self.is_visible(world.ranger, target) or self.is_visible(world.uav, target)
        else: # ranger/uav -> ranger/uav
            assert {agent.name, target.name} == {'ranger', 'uav'}
            return True

    def observation(self, agent, world):
        '''
        Returns the agent observation, including this agent's position and velocity, the relative
        position and velocity of every observable agent, and the position of every landmark. 
        '''
        agent_obs = [agent.state.p_pos, agent.state.p_vel]
        for target in world.agents:
            if agent.name == target.name:
                continue
            one_hot_id = np.zeros(len(world.agents))
            one_hot_id[target.id] = 1
            obs = [target.state.p_pos - agent.state.p_pos, target.state.p_vel, one_hot_id]
            if self.is_observed(agent, target, world):
                agent_obs += obs
            else:
                agent_obs += [np.zeros_like(elt) for elt in obs]
        for landmark in world.landmarks:
            agent_obs.append(landmark.state.p_pos - agent.state.p_pos)
        full_obs = np.concatenate(agent_obs)
        return full_obs

