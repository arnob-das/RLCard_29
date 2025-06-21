import rlcard
import rlcard29
from rlcard.agents import RandomAgent
from rlcard29.agents.human_agent_twenty_nine.human_agent import HumanAgent

def print_header(message):
    print("\n" + "="*20)
    print(f"  {message}")
    print("="*20)

if __name__ == '__main__':
    env = rlcard.make('twenty_nine')
    human_agent = HumanAgent(env.action_num)
    random_agent = RandomAgent(env.action_num)
    # Set player 0 and 2 to be human, 1 and 3 to be random
    env.set_agents([human_agent, random_agent, human_agent, random_agent])

    print_header("29 Card Game")
    
    while env.game.get_match_winner() is None:
        print_header(f"Starting New Round")
        
        state, player_id = env.reset()
        
        # Agent will print its own state
        if env.agents[player_id].use_raw:
            env.agents[player_id].step(state)

        last_logs = []
        round_over = False
        while not round_over:
            agent = env.agents[player_id]
            action, _ = agent.eval_step(state)

            state, player_id = env.step(action)
            
            # Print the logs for the move
            current_logs = env.game.get_game_log()
            new_logs = current_logs[len(last_logs):]
            if new_logs:
                print("\n--- Game Update ---")
                for log_item in new_logs:
                    print(f"  -> {log_item}")
            last_logs = current_logs
            
            # Human agent will print its own state when its turn comes
            if env.agents[player_id].use_raw:
                # This call is only for printing, the real action is chosen above
                pass

            if state['raw_obs']['phase'] == 'end':
                round_over = True
            
        # Round is over, print summary
        print_header("Round Over")
        summary = env.game.get_round_summary()
        print("\n--- Round Summary ---")
        if summary['bid_winner'] is not None:
            print(f"  Bid: {summary['bid_value']} by Player {summary['bid_winner']} (Team {summary['bidding_team_id']})")
            print(f"  Trump: {summary['trump_suit']}")
            print(f"  Team {summary['bidding_team_id']} {'WON' if summary['bid_successful'] else 'LOST'} the bid.")
            print(f"  Final Points: Team 0: {summary['team_points'][0]} | Team 1: {summary['team_points'][1]}")
        else:
            print("  Round ended with no successful bid.")
        
        print(f"\n  Match Score: Team 0: {env.game.match_scores[0]} | Team 1: {env.game.match_scores[1]}")

    print_header("Match Over!")
    winner = env.game.get_match_winner()
    print(f"Team {winner} wins the match!")
    print(f"Final Score: Team 0: {env.game.match_scores[0]} | Team 1: {env.game.match_scores[1]}") 