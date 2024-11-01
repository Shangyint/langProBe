from typing import List, Tuple

from langProBe.AlfWorld.AlfWorld_utils.alfworld_server_manager import alfworld_manager

from .predict import AlfWorldSignature

import dspy

class AlfWorldCoT(dspy.Module):
    def __init__(self, max_steps=40):
        self.module = dspy.ChainOfThought(signature=AlfWorldSignature)
        self.max_steps = max_steps
    
    def format_trace(self, trace: List[Tuple[str, str]]) -> str:
        # return "\n\n".join([f"Code:\n{code}\nOutput:\n{output}" for code, output in trace])
        assert len(trace) > 0
        return "\n\n".join([f"Step {idx+1}:\nThought: {thought}\nAction: {action}\nObservation: {obs}" for idx, (thought, action, obs) in enumerate(trace)])

    def parse_initial_obs(self, obs_text):
        lines = [line.strip() for line in obs_text.split("\n") if len(line.strip()) > 0]
        assert len(lines) == 3
        assert lines[0] == "-= Welcome to TextWorld, ALFRED! =-"
        assert lines[1].startswith('You are in the middle of a room. Looking quickly around you, you see a')
        assert lines[2].startswith('Your task is to: ')

        task_instruction = lines[2][len('Your task is to: '):]
        room_description = lines[1][len('You are in the middle of a room. Looking quickly around you, you see '):]
        return task_instruction, room_description

    def forward(self, game_file):
        with alfworld_manager.acquire_server(game_filepath=game_file) as (server, init_obs, init_info):
            env = server.request
            task_instruction, room_description = self.parse_initial_obs(init_obs)

            won = init_info['won'][0]
            assert won == False

            trace = [('Let me start by looking around the room. After that, I will plan my actions.', 'look', 'You are in the middle of a room. Looking quickly around you, you see ' + room_description)]

            obs  = init_obs
            info = init_info

            for i in range(self.max_steps):
                while True:
                    try:
                        module_output = self.module(
                            objective=task_instruction,
                            past_steps=self.format_trace(trace)
                        )
                        thought = module_output.rationale.strip()
                        selected_action = module_output.action.strip()
                        print("thought", thought)
                        print("selected_action", selected_action)
                        break
                    except ValueError as ve:
                        # TODO: This is a hack to deal with the fact that the model runs out of context window. Need better ways to resolve this
                        if ve.args[0] == "Expected dict_keys(['code']) but got dict_keys([])":
                            if len(trace) >= 2:
                                trace = [trace[0]] + trace[2:]                          
                            else:
                                raise ve
                        else:
                            raise ve

                if selected_action == "admissible actions":
                    obs = repr(info['admissible_commands'][0])
                else:
                    obs, score, done, info = env.step(selected_action)

                print("obs", obs)
                trace.append((thought, selected_action, obs))

                if "Nothing happens." in obs:
                    trace.append(('Let me identify what are the admissible actions.', 'admissible actions', repr(info['admissible_commands'][0])))

                won = info['won'][0]
                print("won", won)
                print("")
                assert won in [True, False]
                if won:
                    break
            
        return dspy.Prediction(success=won)
