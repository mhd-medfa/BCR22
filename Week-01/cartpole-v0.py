import gym

def cartpole_v0():
    """ Description:
         A pole is attached by an un-actuated joint to a cart, which moves along
         a frictionless track. The pendulum starts upright, and the goal is to
         prevent it from falling over by increasing and reducing the cart's
         velocity.
        
        Starting State:
            All observations are assigned a uniform random value in [-0.05..0.05]
        
        Actions:
            Type: Discrete(2)
            Num   Action
            0     Push cart to the left
            1     Push cart to the right
            Note: The amount the velocity that is reduced or increased is not
            fixed; it depends on the angle the pole is pointing. This is because
            the center of gravity of the pole increases the amount of energy needed
            to move the cart underneath it
        
        Episode Termination:
            Pole Angle is more than 12 degrees.
            Cart Position is more than 2.4 (center of the cart reaches the edge of
            the display).
            Episode length is greater than 200.
            Solved Requirements:
            Considered solved when the average return is greater than or equal to
            195.0 over 100 consecutive trials.

    Returns:
        [void]: [void]
    """
    env = gym.make('CartPole-v0')
    total_reward = 0

    for i_episode in range(10):
        # Starting State:
        #     All observations are assigned a uniform random value in [-0.05..0.05]
        print("Episide {}".format(i_episode))
        observation = env.reset()
        print("Initial observation: {}".format(observation))

        for t in range(100):
            env.render()
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