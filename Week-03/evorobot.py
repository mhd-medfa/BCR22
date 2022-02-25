from audioop import avg
import gym
import numpy as np
from time import sleep

class Network:
    def __init__(self, env, hidden=5):
        self.cumultative_reward = 0
        self.pvariance = 0.1 # variance of initial parameters
        self.nhiddens = hidden # number of internal neurons
        self.ninputs = env.observation_space.shape[0]
        if (isinstance(env.action_space, gym.spaces.box.Box)):
            self.noutputs = env.action_space.shape[0] #an integer
        else:
            self.noutputs = env.action_space.n #vector of floating point values
    
        self.W1 = np.random.randn(self.nhiddens,self.ninputs) * self.pvariance      # first connection layer
        self.W2 = np.random.randn(self.noutputs, self.nhiddens) * self.pvariance    # second connection layer
        self.b1 = np.zeros(shape=(self.nhiddens, 1))                         # bias internal neurons
        self.b2 = np.zeros(shape=(self.noutputs, 1))                         # bias motor neurons
        self.nparameteres = self.nhiddens*self.ninputs + self.noutputs*self.nhiddens + self.nhiddens + self.noutputs
    
    def set_genotype(self, genotype):
        idx1 = self.ninputs * self.nhiddens
        idx2 = idx1 + self.noutputs * self.nhiddens
        idx3 = idx2 + self.nhiddens
        idx4 = idx3 + self.noutputs
        # print(idx4) # 37 params in case of CartPole-v0

        splitted_params = np.split(genotype, [idx1, idx2, idx3, idx4])
        self.W1 = splitted_params[0].reshape(self.nhiddens, self.ninputs)
        self.W2 = splitted_params[1].reshape(self.noutputs, self.nhiddens)
        self.b1 = splitted_params[2].reshape(self.nhiddens, 1)
        self.b2 = splitted_params[3].reshape(self.noutputs, 1)
    
    def compute_nparams(self):
        return self.nparameteres
    
    def update(self,observation):
        # W1 , W2 ,b1 ,b2 ,ninputs = self.env_w
        # convert the observation array into a matrix with 1 column and ninputs rows
        observation.resize(self.ninputs,1)
        # compute the netinput of the first layer of neurons
        Z1 = np.dot(self.W1, observation) + self.b1
        # compute the activation of the first layer of neurons with the tanh function
        A1 = np.tanh(Z1)
        # compute the netinput of the second layer of neurons
        Z2 = np.dot(self.W2, A1) + self.b2
        # compute the activation of the second layer of neurons with the tanh function
        A2 = np.tanh(Z2)
        # if the action is discrete
        #  select the action that corresponds to the most activated unit
        if (isinstance(env.action_space, gym.spaces.box.Box)):
            action = A2
        else:
            action = np.argmax(A2)
        return action
    
    def evaluate(self, nepisodes, show=False):
        
        for i_episode in range(nepisodes):
            self.cumultative_reward = 0
            observation = env.reset()
            # print("Episide {}".format(i_episode))
            # print("Initial observation: {}".format(observation))
            done = False
            t = 0
            while not done:
                if show:
                    env.render()
                action = network.update(observation)
                observation, reward, done, info = env.step(action)
                self.cumultative_reward += reward
                # print("t = {}, observation: {}".format(t, observation))
                # print("t = {}, reward = {}, cumultative_reward = {}".format(t, reward, self.cumultative_reward))
                t+=1
                sleep(0.005)
        # print("Episode {} finished after {} timesteps".format(i_episode, t))
        # print("Cumulatitive reward: ", self.cumultative_reward)
        env.close()
        return self.cumultative_reward/nepisodes
    

# env = gym.make("CartPole-v0")
env = gym.make("Acrobot-v1")
# env = gym.make("CartPole-v1")
# env = gym.make("MountainCarContinuous-v0")
# env = gym.make("MountainCar-v0")
network = Network(env, hidden=5)
popsize = 10
half_popsize = int(popsize/2)
variance = 0.1
pertub_variance = 0.2
ngeneration = 100
episodes = 3
threshold = 10
nparameteres = network.compute_nparams()
population = np.random.randn(popsize, nparameteres) * variance
# population = np.zeros((popsize, nparameteres)) * variance

show = False
for g in range(ngeneration):
    #evaluation individuals
    fitness = []
    if g >= ngeneration*0.:
        show=True

    for i in range(popsize):
        network.set_genotype(population[i])
        fit = network.evaluate(episodes, show=show)
        fitness.append(fit)
    #replacing the worst genotype with pertubed versions of the bests genotypes
    indexbest = np.argsort(fitness)
    for i in range(half_popsize):
        population[indexbest[i+half_popsize]] = population[indexbest[i]] + np.random.rand(1, nparameteres) * pertub_variance
    
    max_fit = fitness[indexbest[-1]]
    avg_fit = np.mean(fitness)
    print("********************************")
    print("Generation: {}".format(g))
    print("********************************")
    print("Max Fitness:")
    print(max_fit)
    print("Average Fitness:")
    print(avg_fit)
    print("--------------------------------")
    if max_fit >= threshold:
        print("Done")
        fit = network.evaluate(episodes, show=True)
        input()
        break

