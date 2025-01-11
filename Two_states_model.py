import numpy as np
import random
import matplotlib.pyplot as plt 
import time 

class Agent:
        
    def __init__(self, ):
        self.states = []
        self.action_space = np.array([0, 1])
        self.action = None
        self.action_history = []

    def gen_states(self, nb_actions = 50):
        """
        Generate active and idle states based on normal distribution

        Returns:
            ndarray: states
        """

        mean = 10
        std = 2
        state_transitions = np.random.normal(mean, std, size=nb_actions)
        idle_state = 0

        states = []

        for transition in state_transitions:
            states += [0]*int(transition) + [1] # Fill array with 1 every transition time

        self.states = np.array(states)
        
    def random_policy(self):
        
        for state in self.states:
            if state:
                self.action = np.random.choice(self.action_space)
                self.action_history.append(self.action)
            else:
                self.action_history.append(np.nan)

        self.action_history = np.array(self.action_history)

np.random.seed(seed = 42)

mouse0 = Agent()
mouse0.gen_states()
mouse0.random_policy()

mouse1 = Agent()
mouse1.gen_states()
mouse1.random_policy()

agents = mouse0, mouse1

# Adjust, for all agents, the state and action sizes to max size
target_size = max(mouse0.action_history.size, mouse1.action_history.size)
for agent in agents:
    agent.states = np.pad(agent.states, (0, target_size - agent.states.size), constant_values=0)
    agent.action_history = np.pad(agent.action_history, (0, target_size - agent.action_history.size), constant_values=np.nan)

# Declare time window and compute joint rewards
tw_duration = 4
joint_rewards = 0 
t = 0

while t < target_size: 
    
    if mouse0.states[t]: # If agent0 init, check for action on the opp side 
        init_action = mouse0.action_history[t]
        if (t + tw_duration) >= target_size: # Check for out of bounds index
            opp_array = mouse1.action_history[t: target_size]
        else:
            opp_array = mouse1.action_history[t: (t+tw_duration)] 

        idx_opp_action = np.where(opp_array == init_action)[0]
        if idx_opp_action.size == 0: # If there is joint action, jump to the ja position
            t += tw_duration 
            continue
        # If there is a joint action, move to where t of joint action
        joint_rewards += 1
        t += idx_opp_action.item()
            
    
    if mouse1.states[t]:
        init_action = mouse1.action_history[t]
        if (t + tw_duration) >= target_size: # Check for out of bounds index
            opp_array = mouse0.action_history[t: target_size]
        else:
            opp_array = mouse0.action_history[t: (t+tw_duration)]

        idx_opp_action = np.where(opp_array == init_action)[0]
        if idx_opp_action.size == 0:
            t += tw_duration
            continue
        # If there is a joint action, move to where t of joint action
        joint_rewards += 1
        t += idx_opp_action.item()

    t += 1 # If no active states or joint action, move to next t

print("REWARDS:", joint_rewards)

"""
# Plot results
#time = np.arange(states.size)

plt.figure(figsize=(10, 4))
#plt.plot(time, states, label="transitions")
#plt.hist(state_transitions, bins=np.arange(20), label='State (0: Idle, 1: Active)')
plt.xlabel('Time Step')
plt.ylabel('State')
plt.title('Two-State Model Simulation')
plt.legend()
plt.show()
"""