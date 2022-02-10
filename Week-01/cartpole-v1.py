import gym

def cartpole_v0():
    """ Description:
         The inverted pendulum swingup problem is a classic problem in the control literature. In this
        version of the problem, the pendulum starts in a random position, and the goal is to swing it up so
        it stays upright.
        The diagram below specifies the coordinate system used for the implementation of the pendulum's
        dynamic equations.
        ![Pendulum Coordinate System](./diagrams/pendulum.png)
        - `x-y`: cartesian coordinates of the pendulum's end in meters.
        - `theta`: angle in radians.
        - `tau`: torque in `N * m`. Defined as positive _counter-clockwise_.
        
        Action Space:
        The action is the torque applied to the pendulum.
        | Num | Action | Min  | Max |
        |-----|--------|------|-----|
        | 0   | Torque | -2.0 | 2.0 |
        
        Observation Space:
        The observations correspond to the x-y coordinate of the pendulum's end, and its angular velocity.
        | Num | Observation      | Min  | Max |
        |-----|------------------|------|-----|
        | 0   | x = cos(theta)   | -1.0 | 1.0 |
        | 1   | y = sin(angle)   | -1.0 | 1.0 |
        | 2   | Angular Velocity | -8.0 | 8.0 |
        
        Rewards:
        The reward is defined as:
        ```
        r = -(theta^2 + 0.1*theta_dt^2 + 0.001*torque^2)
        ```
        where `theta` is the pendulum's angle normalized between `[-pi, pi]`.
        Based on the above equation, the minimum reward that can be obtained is `-(pi^2 + 0.1*8^2 +
        0.001*2^2) = -16.2736044`, while the maximum reward is zero (pendulum is
        upright with zero velocity and no torque being applied).
        
        Starting State:
        The starting state is a random angle in `[-pi, pi]` and a random angular velocity in `[-1,1]`.
        
        Episode Termination:
        An episode terminates after 200 steps. There's no other criteria for termination.
        
        Arguments:
        - `g`: acceleration of gravity measured in `(m/s^2)` used to calculate the pendulum dynamics. The default is
        `g=10.0`.

    Returns:
        [void]: [void]
    """
    env = gym.make('CartPole-v1', g=9.81)
    total_reward = 0

    for i_episode in range(10):
        # Starting State:
        #     All observations are assigned a uniform random value in [-0.05..0.05]
        print("Episide {}".format(i_episode))
        observation = env.reset()
        print("Initial observation: {}".format(observation))

        for t in range(100):
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