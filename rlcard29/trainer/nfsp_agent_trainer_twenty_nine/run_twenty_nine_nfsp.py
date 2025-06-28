"""
File: rlcard29/trainer/nfsp_agent_trainer_twenty_nine/run_twenty_nine_nfsp.py
Author: Arnob Das
Date: 2025-06-28
"""
    
import rlcard29
from rlcard29.agents.random_agent import RandomAgent
from rlcard29.envs.twenty_nine import TwentyNineEnv
from rlcard29.envs.registration import register
register('twenty_nine', TwentyNineEnv)

import os
import torch
from rlcard.agents.nfsp_agent import NFSPAgent

if __name__ == '__main__':
    env = rlcard29.make('twenty_nine')
    nfsp_agent = NFSPAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        hidden_layers_sizes=[128,128],
        q_mlp_layers=[128,128],
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    agents = [nfsp_agent] + [RandomAgent(num_actions=env.num_actions) for _ in range(3)]
    env.set_agents(agents)
    for episode in range(1, 10001):
        trajectories, payoffs = env.run(is_training=True)
        nfsp_agent.feed(trajectories[0])
        if episode % 100 == 0:
            print(f"Episode {episode}: Payoff={payoffs[0]}")
        if episode % 1000 == 0:
            torch.save(nfsp_agent.q_estimator.q_network.state_dict(), f'nfsp_model_ep{episode}.pth')
            print(f"Model saved to nfsp_model_ep{episode}.pth")
    print('Training finished.') 