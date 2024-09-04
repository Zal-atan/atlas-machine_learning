# This is a README for the  Deep Q Learning repo.

### In this repo we will practicing basic uses of Deep Q Learning in Machine Learning to make an Agent which can play the clasic Atari Game 'Breakout'
<br>

### Author - Ethan Zalta
<br>


# Tasks
### There is 1 task in this project

## Task 0
* Write a python script train.py that utilizes keras, keras-rl, and gym to train an agent that can play Atari’s Breakout:

    * Your script should utilize keras-rl‘s DQNAgent, SequentialMemory, and EpsGreedyQPolicy
    * Your script should save the final policy network as policy.h5

* Write a python script play.py that can display a game played by the agent trained by train.py:

    * Your script should load the policy network saved in policy.h5
    * Your agent should use the GreedyQPolicy
