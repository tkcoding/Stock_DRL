# Stock_DRL
Using Deep reinforcement learning (DRL) agent to learn from historical data.

This framework can be used for other real life problem with environment setup to replace on StockMarketEnv() function. 
Please feel free to use it for your own use cases.

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

# Results

Deep Q-Network runs on BAUTO stock in KLSE (Malaysia Stock Exchange) over 1500 episodes.
![DQN Image](assets/DQN_Reward.png)

DoubleDeep Q-Network runs on BAUTO stock in KLSE (Malaysia Stock Exchange) over 1500 episodes.
![DDQN Image](assets/DDQN_Reward_graph.png)

# Next Step
```
1. Using saved model from DQN or DDQN to load multiple different stock with train and test set in Malaysia KLSE stock market as reference - estimated completion by November15 2020
2. Implement prioritized replay and dueling DQN as benchmark - estimated completion by December 1st 2020
```
