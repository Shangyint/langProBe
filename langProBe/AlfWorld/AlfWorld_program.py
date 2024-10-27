from typing import List, Tuple

from langProBe.AlfWorld.AlfWorld_utils.alfworld_server_manager import alfworld_manager
from .AlfWorld_utils.alfredtwenv_wrapper import AlfredTWEnvOneGame

import dspy

class AlfWorldReactSignature(dspy.Signature):
    """You are a super intelligent AI Assistant whose job is to complete my household tasks.

To do this, you will need to take embodied actions in my house. For this you will undertake a *multi-step conversation*. That is, you will determine the next action to be taken, and the environment will provide you with the observation after the action is taken, based on which, you will determine the next action to be taken. The environment will let you take actions on my behalf.

You can take any of the following actions:
* go to {r}
* take {o} from {r}
* put {o} in/on {r}
* put {o} into {outero}
* put {outero} in {r}
* open {r}
* close {r}
* toggle {r}
* heat {o} with {r}
* clean {o} with {r}
* cool {o} with {r}
* slice {co} with {ko}
* inventory
* examine {r}
* use {o}
* look
* admissible actions

After each action, an observation will be provided from the environment. If the observation is 'Nothing happens.', it indicates that no action was executed, which could be due to an invalid action, or wrong formatting of the action. Use "admissible actions" to check if the action is valid.

**Key instructions and disclaimers**:

1. Only generate valid actions, i.e., do not put them in ```...``` or add any extra formatting.
2. Take only one action at a time.
3. You must make all decisions completely autonomously and not ask for any clarifications or confirmations from me or anyone else.
4. To "put" an object, you must have it in your inventory.
5. To get a list of admissible actions, you can use the "admissible actions" action.
6. You can only carry one object in your inventory at a time.
7. Think carefully about how you can use the appliances/resources in the room to complete the task."""
    objective = dspy.InputField(description="Task to be completed", format=str)
    past_steps = dspy.InputField(description="Past actions and observations", format=str)
    action = dspy.OutputField(description="Next action to be taken", format=str)

class AlfWorldReact(dspy.Module):
    def __init__(self, max_steps=40):
        self.module = dspy.Predict(signature=AlfWorldReactSignature)
        self.max_steps = max_steps
    
    def format_trace(self, trace: List[Tuple[str, str]]) -> str:
        # return "\n\n".join([f"Code:\n{code}\nOutput:\n{output}" for code, output in trace])
        assert len(trace) > 0
        return "\n\n".join([f"Action: {action}\nObservation: {obs}" for action, obs in trace])

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
            print("task_instruction", task_instruction)
            print("room_description", room_description)

            won = init_info['won'][0]
            assert won == False

            trace = [('look', 'You are in the middle of a room. Looking quickly around you, you see ' + room_description)]

            for i in range(self.max_steps):
                while True:
                    try:
                        module_output = self.module(
                            objective=task_instruction,
                            past_steps=self.format_trace(trace)
                        )
                        selected_action = module_output.action.strip()
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
                trace.append((selected_action, obs))
                won = info['won'][0]
                print("won", won)
                print("")
                assert won in [True, False]
                if won:
                    break
            
        return dspy.Prediction(trace=trace, success=won)

class AlfWorldReactWithThought(dspy.Module):
    def __init__(self, max_steps=40):
        signature = AlfWorldReactSignature.prepend(name="thought", field=dspy.OutputField(description="Thought process for the next action", format=str))
        self.module = dspy.Predict(signature=signature)
        self.max_steps = max_steps
    
    def format_trace(self, trace: List[Tuple[str, str]]) -> str:
        # return "\n\n".join([f"Code:\n{code}\nOutput:\n{output}" for code, output in trace])
        assert len(trace) > 0
        return "\n\n".join([f"Thought: {thought}\nAction: {action}\nObservation: {obs}" for thought, action, obs in trace])

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

            for i in range(self.max_steps):
                while True:
                    try:
                        module_output = self.module(
                            objective=task_instruction,
                            past_steps=self.format_trace(trace)
                        )
                        thought = module_output.thought.strip()
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
                won = info['won'][0]
                print("won", won)
                print("")
                assert won in [True, False]
                if won:
                    break
            
        return dspy.Prediction(trace=trace, success=won)