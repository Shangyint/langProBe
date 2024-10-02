import gymnasium as gym
import browsergym.webarena  # register webarena tasks as gym environments
import dspy

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


env = gym.make("browsergym/webarena.67")
obs, info = env.reset()
done = False
# dict_keys(['chat_messages', 'goal', 'goal_image_urls', 'open_pages_urls', 'active_page_index', 'url', 'screenshot', 'dom_object', 'axtree_object', 'extra_element_properties', 'focused_element_bid', 'last_action', 'last_action_error', 'elapsed_time'])
print(obs["chat_messages"])
print(obs["goal"])
print(obs["open_pages_urls"])
print(obs["url"])
print(obs["axtree_object"])

# Just a scratch! NOT WORKING
class SimpleAgent(dspy.Module):
    def __init__(self):
        super().__init__()

        self.agent = dspy.ChainOfThought("observation -> answer")

    def forward(self, task_id):
        env = gym.make(f"browsergym/webarena.{task_id}")
        obs, info = env.reset()
        done = False

        while not done:
            action = self.agent(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated