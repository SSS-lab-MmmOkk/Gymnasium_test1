import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
from mesa.time import RandomActivation
from mesa import Model
from agents import PedestrianAgent, VehicleAgent

class CrosswalkModel(Model):
    def __init__(self, sumo_cmd):
        self.schedule = RandomActivation(self)
        traci.start(sumo_cmd)

        self.pedestrian = PedestrianAgent("pedestrian", self)
        self.schedule.add(self.pedestrian)

        self.vehicle = VehicleAgent("vehicle", self)
        self.schedule.add(self.vehicle)

    def step(self):
        traci.simulationStep()
        self.schedule.step()

class CrosswalkEnv(gym.Env):
    def __init__(self):
        super(CrosswalkEnv, self).__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.sumo_cmd = ["sumo", "-c", "cross.sumocfg"]
        self.model = None

    def reset(self, seed=None, options=None):
        if self.model:
            traci.close()
        self.model = CrosswalkModel(self.sumo_cmd)
        traci.simulationStep()
        traci.simulationStep()
        traci.simulationStep()
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        self.model.step()
        traci.simulationStep()
        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._is_terminated()
        truncated = False
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        ped_pos = (0, 0)
        if "pedestrian" in traci.person.getIDList():
            ped_pos = traci.person.getPosition("pedestrian")
        veh_pos = (0, 0)
        if "vehicle" in traci.vehicle.getIDList():
            veh_pos = traci.vehicle.getPosition("vehicle")
        return np.array([ped_pos[0], veh_pos[0]], dtype=np.float32)

    def _get_reward(self):
        return 0

    def _is_terminated(self):
        return traci.simulation.getMinExpectedNumber() == 0

    def _get_info(self):
        return {}

    def close(self):
        traci.close()
