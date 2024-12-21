import json
import os

from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
import yaml


class AlfredTWEnvOneGame(AlfredTWEnv):
    def __init__(self, gamefile_path=None):
        with open('langProBe/AlfWorld/configs/base_config.yaml', 'r') as reader:
            self.config = yaml.safe_load(reader)
        self.train_eval = 'eval'
        self.gamefile_path = gamefile_path
        self.collect_game_files()

    def collect_game_files(self, verbose=False):
        self.game_files = []

        assert self.gamefile_path.endswith('game.tw-pddl')

        root = os.path.dirname(self.gamefile_path)
        assert os.path.exists(os.path.join(root, 'traj_data.json'))
        assert os.path.exists(os.path.join(root, 'game.tw-pddl'))

        # Filenames
        # json_path = os.path.join(root, 'traj_data.json')
        game_file_path = os.path.join(root, 'game.tw-pddl')

        assert 'movable' not in root and 'Sliced' not in root, f'Movable & slice trajs not supported {root}'

        with open(game_file_path, 'r') as f:
            gamedata = json.load(f)

        assert 'solvable' in gamedata, f'Missing solvable key! {game_file_path}'
        assert gamedata['solvable'], f'Unsolvable game! {game_file_path}'

        # Add to game file list
        self.game_files.append(game_file_path)

        self.num_games = len(self.game_files)

        assert self.train_eval == 'eval', 'Only eval mode supported'
