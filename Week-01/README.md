
# Week-01

In order to familiarize myself with AI-Gym commands a python script by using the “Cartpole-v0” environment has been written and attached to this repo **Week-01/cartpole-v0.py**, where the agent performs 10 evaluation episodes (random actions, for 10 episodes 100 timesteps) by showing the behavior and by printing the step, the observation vector, the action vector, the reward, and the fitness (i.e. the cumulative reward)

- Three other classic control tasks (CartPole-v1, MountainCar-v0, MountainCarContinuous-v0) have been studied with code review for each of them.

## 1- Cartpole-v1

### Observation Space
The observations correspond to the x-y coordinate of the pendulum's end, and its angular velocity.

Num | Observation      | Min  | Max 
-----|------------------|------|-----
0   | x = cos(theta)   | -1.0 | 1.0 
1   | y = sin(angle)   | -1.0 | 1.0 
2   | Angular Velocity | -8.0 | 8.0 

### Action Space
The action is the torque applied to the pendulum.

Box(1)

Num | Action | Min  | Max 
-----|--------|------|-----
0   | Torque | -2.0 | 2.0 

### Starting State
The starting state is a random angle in [-pi, pi] and a random angular velocity in [-1,1].

### Episode Termination
An episode terminates after 200 steps. There's no other criteria for termination.

## 2- MountainCar-v0  

Description: Get an under powered car to the top of a hill (top = 0.5 position)

### Observation

The observation space is a 2-dim vector, where the 1st element represents the "car position" and the 2nd element represents the "car velocity".

Box(2)

Num | Observation  | Min  | Max  
----|--------------|------|----   
0   | position     | -1.2 | 0.6
1   | velocity     | -0.07| 0.07


### Actions

Discrete(3)

Num | Action
----|-------------
0   | Accelerate to the Left   
1   | Don't accelerate     
2   | Accelerate to the Right  

### Reward

Reward of 0 is awarded if the agent reached the flag
    (position = 0.5) on top of the mountain. Reward of -1 is awarded if the position of the agent is less than 0.5.
    
### Starting State

The position of the car is assigned a uniform random value in [-0.6 , -0.4]. The starting velocity of the car is always assigned to 0.
    
### Episode Termination

The car position is more than 0.5. Episode length is greater than 200

## 3- MountainCarContinuous-v0 

Description: Get an under powered car to the top of a hill (top = 0.5 position)

### Observation
The observation space is a 2-dim vector, where the 1st element represents the "car position" and the 2nd element represents the "car velocity".

Box(2)

Num | Observation  | Min  | Max  
----|--------------|------|----   
0   | position     | -1.2 | 0.6
1   | velocity     | -0.07| 0.07


### Actions

Type: Discrete(3)

Num | Action
----|-------------
0   | push left   
1   | no push     
2   | push right  

### Reward

Reward of 100 is awarded if the agent reached the flag (position = 0.45) on top of the mountain. Reward is decrease based on amount of energy consumed each step.

### Starting State

The position of the car is assigned a uniform random value in [-0.6 , -0.4]. The starting velocity of the car is always assigned to 0.

### Episode Termination

The car position is more than 0.45. Episode length is greater than 200
