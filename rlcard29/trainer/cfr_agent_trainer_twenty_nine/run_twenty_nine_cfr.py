"""
File: rlcard29/trainer/cfr_agent_trainer_twenty_nine/run_twenty_nine_cfr.py
Author: Arnob Das
Date: 2025-06-28
"""
    
import rlcard29
from rlcard.agents.random_agent import RandomAgent
from rlcard29.envs.twenty_nine import TwentyNineEnv
from rlcard.envs.registration import register
register('twenty_nine', TwentyNineEnv)

from rlcard.agents.cfr_agent import CFRAgent

if __name__ == '__main__':
    env = rlcard29.make('twenty_nine')
    agents = [CFRAgent(env)] + [RandomAgent(num_actions=env.num_actions) for _ in range(3)]
    env.set_agents(agents)
    for episode in range(1, 10001):
        trajectories, payoffs = env.run(is_training=True)
        if episode % 100 == 0:
            print(f"Episode {episode}: Payoff={payoffs[0]}")
    print('Training finished.') 