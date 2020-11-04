# Framework for Deep Reinforcement Learning on Stock Market Environment
Using Deep reinforcement learning (DRL) agent to learn from historical data.

This framework can be used for other real life problem with environment setup to replace on StockMarketEnv() function. 
Please feel free to use it for your own use cases.

### Basics to consider during stock purchase decision in terms of DRL.

Training an reinforcement learning framework require understanding of both technical details of finance and method can be used in RL to churn out decent RL.

<br>
<figure>
  <img src = "./assets/RL.png" width = 80% style = "border: thin silver solid; padding: 10px">
      <figcaption style = "text-align: center; font-style: italic">Fig 1. - Reinforcement Learning.</figcaption>
</figure> 
<br>

Basic such as reinforcement learning require setup from agent (just like traders or mastermind) , observations (state for example like trading volume,news sentiment and other important factors) and lastly the action (buy and sell as discrete action - __easiest__ to number of share to buy and sell as continuous space - __hardest__)


### Outline for stock DRL 
The approach we are taking to continuous improve this repo by introducing stages by stages improvement as author getting more work done:
1. Using DQN (Deep Q-Network) to start off with continuous state space, keeping discrete action space (buy and sell call only , consider like binary state)
2. Using DDPG (Deep Deterministic Policy Gradient) to deal with continuous state and action space (buy and sell call with volume)

```
WHY NOT Q-Learning?????

Q-learning tabular method will not be cover in this repository as author think it's not feasible for stock trading activites due to:
1. Q-learning alone is unable to cope the continuous state and action space due to the nature of finite MDP.
```

### Deep Q-Network for continuous state space and discrete action space
To learn more about the differences between DQN and DDQN Minh et al. 2015 : https://arxiv.org/pdf/1509.06461.pdf

Credit to developer at Kaggle for the initial written logic: https://www.kaggle.com/itoeiji/deep-reinforcement-learning-on-stock-data


Library use :
Pytorch and Python3

go to the official website: http://pytorch.org/

1. Select Windows as your operating system
2. Select your Package Manager such as pip or conda
3. Select you python version
4. Select CUDA or choose none You will get the command that will install pytorch on your system based on your selection.
5. For example, if you choose Windows, pip, python 3.6 and none in the listed steps, you will get the following commands:

```
pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.0-cp36-cp36m-win_amd64.whl 
pip3 install torchvision
```

### Results

Deep Q-Network runs on BAUTO stock in KLSE (Malaysia Stock Exchange) over 1500 episodes.
![DQN Image](assets/DQN_Reward.png)

DoubleDeep Q-Network runs on BAUTO stock in KLSE (Malaysia Stock Exchange) over 1500 episodes.
![DDQN Image](assets/DDQN_Reward_graph.png)


### Next Step
```
1. Using saved model from DQN or DDQN to load multiple different stock with train and test set in Malaysia KLSE stock market as reference - estimated completion by November15 2020
2. Implement prioritized replay and dueling DQN as benchmark - estimated completion by December 1st 2020
3. Current action is on discrete state 0 for hold , 1 for buy and 2 for sell call - Using DQN . Transition to Policy Gradient for continuous action space -  estimated completion by January 1st 2021.
  b. Will upload Policy Gradient code in Policy Gradient in Policy_Gradient folder for future development.
```


## Deterministic Deep Policy Gradient (Actor-Critic Method) for both state and action to be continuous space (buy and sell with volume)

__Author still working on this part. Please stay tune as author weekly update as he learnt through the method__

In actor-critic method , policy gradient method has been deployed to use for both actor and critic agent. Understanding policy gradient and information behind this technique is crucial for us to use in our project.
Below are some of the resources that could help in policy gradient understanding before going into actor critic:
* math  behind policy gradient - http://machinelearningmechanic.com/deep_learning/reinforcement_learning/2019/12/06/a_mathematical_introduction_to_policy_gradient.html
* Detail explanation on policy gradient and how it links to actor and critic - https://deepmind.com/learning-resources/-introduction-reinforcement-learning-david-silver


In reinforcement learning, an agent makes observations and takes actions within an environment, and in return it receives rewards. Its objective is to learn to act in a way that will maximize its expected long-term rewards. 



There are several types of RL algorithms, and they can be divided into three groups:

- **Critic-Only**: Critic-Only methods, also known as Value-Based methods, first find the optimal value function and then derive an optimal policy from it. 


- **Actor-Only**: Actor-Only methods, also known as Policy-Based methods, search directly for the optimal policy in policy space. This is typically done by using a parameterized family of policies over which optimization procedures can be used directly. 


- **Actor-Critic**: Actor-Critic methods combine the advantages of actor-only and critic-only methods. In this method, the critic learns the value function and uses it to determine how the actor's policy parramerters should be changed. In this case, the actor brings the advantage of computing continuous actions without the need for optimization procedures on a value function, while the critic supplies the actor with knowledge of the performance. Actor-critic methods usually have good convergence properties, in contrast to critic-only methods.  The **Deep Deterministic Policy Gradients (DDPG)** algorithm is one example of an actor-critic method.

<br>
<figure>
  <img src = "./assets/Actor-Critic.png" width = 80% style = "border: thin silver solid; padding: 10px">
      <figcaption style = "text-align: center; font-style: italic">Fig 2. - Actor-Critic Reinforcement Learning.</figcaption>
</figure> 
<br>

In this notebook, we will use DDPG to determine the optimal execution of portfolio transactions. In other words, we will use the DDPG algorithm to solve the optimal liquidation problem. But before we can apply the DDPG algorithm we first need to formulate the optimal liquidation problem so that in can be solved using reinforcement learning. In the next section we will see how to do this. 
