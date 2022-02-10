import gym

def cartpole_v0():
    """ Description:
        The agent (a car) is started at the bottom of a valley. For any given state
    the agent may choose to accelerate to the left, right or cease any
    acceleration.

    Observation space is a 2-dim vector, where the 1st element represents the "car position" and the 2nd element represents the "car velocity".
    There are 3 discrete deterministic actions:
    - 0: Accelerate to the Left
    - 1: Don't accelerate
    - 2: Accelerate to the Right
    
    Reward: Reward of 0 is awarded if the agent reached the flag
    (position = 0.5) on top of the mountain. Reward of -1 is awarded if the position of the agent is less than 0.5.
    
    Starting State: The position of the car is assigned a uniform random value in [-0.6 , -0.4]. The starting velocity of the car is always assigned to 0.
    
    Episode Termination: The car position is more than 0.5. Episode length is greater than 200

    Returns:
        [void]: [void]
    """
    env = gym.make('MountainCar-v0')
    total_reward = 0

    for i_episode in range(100):
        # Starting State:
        #     All observations are assigned a uniform random value in [-0.05..0.05]
        print("Episide {}".format(i_episode))
        observation = env.reset()
        print("Initial observation: {}".format(observation))

        for t in range(1000):
            env.render()
            print("t = {}, observation: {}".format(t, observation))
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            print("t = {}, observation: {}".format(t, observation))
            print("t = {}, reward = {}".format(t, reward))
            total_reward += reward #Reward is 1 for every step taken, including the termination step
            if done:
                print("Episode {} finished after {} timesteps".format(i_episode, t+1))
                print("Cumulatitive reward: ", total_reward)
                break
        
        

    env.close()

if __name__ == "__main__":
    cartpole_v0()