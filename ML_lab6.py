#!/usr/bin/env python
# coding: utf-8

# # Name - Aniket Kharat
# # Roll no - 2447008 , Batch - A

# In[18]:


import numpy as np

maze = np.array([[0, 0, 0, 1],
                 [0, 1, 0, 1],
             
                 [0, 0, 0, 0],
                 [1, 0, 1, 0]])


# In[19]:


# Parameters
actions = ['up', 'down', 'left', 'right']
q_table = np.zeros((maze.shape[0], maze.shape[1], len(actions)))
gamma = 0.9 
alpha = 0.8  
epsilon = 0.1  


# In[20]:


# Rewards and goal
goal = (3, 3)
rewards = np.zeros_like(maze)
rewards[goal] = 1


# In[21]:


# Movement definitions
def move(state, action):
    if action == 'up' and state[0] > 0:
        return (state[0] - 1, state[1])
    elif action == 'down' and state[0] < maze.shape[0] - 1:
        return (state[0] + 1, state[1])
    elif action == 'left' and state[1] > 0:
        return (state[0], state[1] - 1)
    elif action == 'right' and state[1] < maze.shape[1] - 1:
        return (state[0], state[1] + 1)
    return state


# In[22]:


# Q-learning algorithm
def q_learning(episodes=1000):
    for _ in range(episodes):
        state = (0, 0)
        while state != goal:
            if np.random.uniform(0, 1) < epsilon: 
                action_idx = np.random.choice(len(actions))
            else:  
                action_idx = np.argmax(q_table[state[0], state[1]])

            next_state = move(state, actions[action_idx])
            reward = rewards[next_state]
            q_value = q_table[state[0], state[1], action_idx]
            
            
            q_table[state[0], state[1], action_idx] = q_value + alpha * (reward + gamma * np.max(q_table[next_state[0], next_state[1]]) - q_value)
            state = next_state



# In[23]:


q_learning()
print(q_table)


# In[ ]:




