import gymnasium as gym
import browsergym.webarena  # register webarena tasks as gym environments

BASE_URL = "http://3.13.36.195"

SLEEP = 1.5
# set the URLs of each website, we use the demo sites as an example
import os

os.environ["SHOPPING"] = f"{BASE_URL}:7770"
os.environ["SHOPPING_ADMIN"] = f"{BASE_URL}:7780/admin"
os.environ["REDDIT"] = f"{BASE_URL}:9999"
os.environ["GITLAB"] = f"{BASE_URL}:8023"
os.environ["MAP"] = f"{BASE_URL}:3000"
os.environ[
    "WIKIPEDIA"
] = f"{BASE_URL}:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
os.environ[
    "HOMEPAGE"
] = "PASS"  # The home page is not currently hosted in the demo site

# set up WA_SHOPPING environment variables too
os.environ["WA_SHOPPING"] = f"{BASE_URL}:7770"
os.environ["WA_SHOPPING_ADMIN"] = f"{BASE_URL}:7780/admin"
os.environ["WA_REDDIT"] = f"{BASE_URL}:9999"
os.environ["WA_GITLAB"] = f"{BASE_URL}:8023"
os.environ["WA_MAP"] = f"{BASE_URL}:3000"
os.environ[
    "WA_WIKIPEDIA"
] = f"{BASE_URL}:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
os.environ[
    "WA_HOMEPAGE"
] = "PASS"  # The home page is not currently hosted in the demo site


env = gym.make("browsergym/webarena.310")
obs, info = env.reset()
done = False
env_ids = [id for id in gym.envs.registry.keys() if id.startswith("browsergym/webarena")]
print("\n".join(env_ids))

while not done:
    action = None # DSPy agent here!
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
