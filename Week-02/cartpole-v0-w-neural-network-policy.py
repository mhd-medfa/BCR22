import gym
import numpy as np
from time import sleep

class Network:
    def __init__(self, env) -> None:
        self.cumultative_reward = 0
        self.pvariance = 0.1 # variance of initial parameters
        self.nhiddens = 5 # number of internal neurons
        self.ninputs = env.observation_space.shape[0]
        if (isinstance(env.action_space, gym.spaces.box.Box)):
            self.noutputs = env.action_space.shape[0] #an integer
        else:
            self.noutputs = env.action_space.n #vector of floating point values
    
        W1 = np.random.randn(self.nhiddens,self.ninputs) * self.pvariance      # first connection layer
        W2 = np.random.randn(self.noutputs, self.nhiddens) * self.pvariance    # second connection layer
        b1 = np.zeros(shape=(self.nhiddens, 1))                         # bias internal neurons
        b2 = np.zeros(shape=(self.noutputs, 1))                         # bias motor neurons
        self.env_w = [W1 , W2 ,b1 ,b2 ,self.ninputs]

    def update(self,observation):
        W1 , W2 ,b1 ,b2 ,ninputs = self.env_w
        # convert the observation array into a matrix with 1 column and ninputs rows
        observation.resize(ninputs,1)
        # compute the netinput of the first layer of neurons
        Z1 = np.dot(W1, observation) + b1
        # compute the activation of the first layer of neurons with the tanh function
        A1 = np.tanh(Z1)
        # compute the netinput of the second layer of neurons
        Z2 = np.dot(W2, A1) + b2
        # compute the activation of the second layer of neurons with the tanh function
        A2 = np.tanh(Z2)
        # if the action is discrete
        #  select the action that corresponds to the most activated unit
        if (isinstance(env.action_space, gym.spaces.box.Box)):
            action = A2
        else:
            action = np.argmax(A2)
        return action
    
    def evaluate(self, nepisodes):
        for i_episode in range(nepisodes):
            self.cumultative_reward = 0
            observation = env.reset()
            print("Episide {}".format(i_episode))
            print("Initial observation: {}".format(observation))
            done = False
            t = 0
            while not done:
                env.render()
                action = network.update(observation)
                observation, reward, done, info = env.step(action)
                self.cumultative_reward += reward
                print("t = {}, observation: {}".format(t, observation))
                print("t = {}, reward = {}, cumultative_reward = {}".format(t, reward, self.cumultative_reward))
                t+=1
                sleep(0.01)
        print("Episode {} finished after {} timesteps".format(i_episode, t))
        print("Cumulatitive reward: ", self.cumultative_reward)
        env.close()
        return self.cumultative_reward/nepisodes
    

env = gym.make("CartPole-v0")
# env = gym.make("CartPole-v1")
# env = gym.make("MountainCarContinuous-v0")
# env = gym.make("MountainCar-v0")
network = Network(env)
fitness = network.evaluate(100)

