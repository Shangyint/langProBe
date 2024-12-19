import os
import sys
import traceback

from flask import Flask, request, jsonify

from .alfredtwenv_wrapper import AlfredTWEnvOneGame

app = Flask(__name__)


@app.route("/reset", methods=["POST"])
def reset():
    try:
        obs, info = game_env.reset()
        assert len(obs) == 1
        return jsonify({"status": "success", "result": {"obs": obs[0], "info": info}})
    except Exception:
        return jsonify({"status": "error", "message": traceback.format_exc()})


@app.route("/step", methods=["POST"])
def step():
    try:
        print("Called")
        action = request.json.get("action")
        print("Action", action)
        # if not action:
        #     return jsonify({"status": "error", "message": "Action is required"}), 400

        result = game_env.step([action])
        print("Result", result)
        assert all(len(x) == 1 for x in result[:-1])
        obs, scores, dones, infos = result
        return jsonify(
            {
                "status": "success",
                "result": {
                    "obs": obs[0],
                    "scores": scores[0],
                    "dones": dones[0],
                    "infos": infos,
                },
            }
        )
    except Exception:
        return jsonify({"status": "error", "message": traceback.format_exc()})


if __name__ == "__main__":
    game_filepath = sys.argv[1]
    portnum = int(sys.argv[2])

    if not os.path.exists(game_filepath):
        raise FileNotFoundError(f"Game file not found: {game_filepath}")

    game_env = AlfredTWEnvOneGame(gamefile_path=game_filepath).init_env(batch_size=1)

    app.run(host="0.0.0.0", port=portnum, debug=True)
