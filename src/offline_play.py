"""
This script allows you to play against a fixed model.
You (human player) will be player 0, and the AI model will be player 1.
"""

import textarena as ta 
from agent import LLMAgent

# initialize the agents
agents = {
    0: LLMAgent(model_name="HuggingFaceTB/SmolLM-1.7B" ,quantize=True, device="auto"),
    1: LLMAgent(model_name="HuggingFaceTB/SmolLM-1.7B" ,quantize=True, device="auto"),
    2: LLMAgent(model_name="HuggingFaceTB/SmolLM-1.7B" ,quantize=True, device="auto"),
    3: LLMAgent(model_name="HuggingFaceTB/SmolLM-1.7B" ,quantize=True, device="auto"),
    4: LLMAgent(model_name="HuggingFaceTB/SmolLM-1.7B" ,quantize=True, device="auto"),
    5: LLMAgent(model_name="HuggingFaceTB/SmolLM-1.7B" ,quantize=True, device="auto"),
    6: LLMAgent(model_name="HuggingFaceTB/SmolLM-1.7B" ,quantize=True, device="auto"),
    7: LLMAgent(model_name="HuggingFaceTB/SmolLM-1.7B" ,quantize=True, device="auto"),
    8: LLMAgent(model_name="HuggingFaceTB/SmolLM-1.7B" ,quantize=True, device="auto"),
}

env = ta.make(env_id="SecretMafia-v0")
env.reset(num_players=len(agents))

done = False 
while not done:
  player_id, observation = env.get_observation()
  action = agents[player_id](observation)
  done, step_info = env.step(action=action)
rewards, game_info = env.close()

print(f"Rewards: {rewards}")
print(f"Game Info: {game_info}")