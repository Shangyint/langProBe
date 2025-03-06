from typing import List, Tuple

from langProBe.AlfWorld.AlfWorld_utils.alfworld_server_manager import alfworld_manager

import dspy

from langProBe.dspy_program import LangProBeDSPyMetaProgram

class AlfWorldReactSignature(dspy.Signature):
    """You are a super intelligent AI Assistant whose job is to complete my household tasks by taking actions in my house. To complete the task, you will determine the next action to be taken, and the environment will provide you with the observation after the action is taken, based on which, you will determine the next action to be taken.

You can take any of the following actions:
* go to {r}
* take {o} from {r}
* put {o} in/on {r}
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

**Key instructions**:

* Take only one action at a time.
* To get a list of admissible actions, you can use "admissible actions".
* You can only carry one object in your inventory at a time.
* To "put" an object on a target, you must have it in your inventory, and be near the target.
* Each action should be wrapped like environment[action]."""
    objective = dspy.InputField(description="Task to be completed", format=str)
    success = dspy.OutputField(description="SUCCESS after task is marked complete")

class DSPyTool:
    name: str
    desc: str
    input_variable: str

class AlfWorldEnvTool(DSPyTool):
    name = "environment"
    desc = "Performs the given action in the environment and provides the observation after taking the action."
    input_variable = "action"

    def set_state(self, env, won, info):
        self.env = env
        self.won = won
        self.info = info

    def __call__(self, action):
        if self.won:
            return "Task completed successfully. You can now call Finish[]."
        selected_action = action.strip()
        if selected_action == "admissible actions":
            obs = repr(info['admissible_commands'][0])
        else:
            obs, score, done, info = self.env.step(selected_action)
            self.won = info['won'][0]
            self.info = info
        
        if self.won:
            return "Task completed successfully. You can now call Finish[]."
        return obs

class AlfWorldReAct(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self, max_steps=40):
        self.tool = AlfWorldEnvTool()
        self.module = dspy.ReAct(signature=AlfWorldReactSignature, tools=[self.tool], max_iters=max_steps)

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

            info = init_info

            self.tool.set_state(env, won, info)

            objective = "You are in the middle of a room. Looking quickly around you, you see " + room_description + "\n\n" + "Your task is to: " + task_instruction

            output = self.module(objective=objective)
            
        return dspy.Prediction(success=self.tool.won)
