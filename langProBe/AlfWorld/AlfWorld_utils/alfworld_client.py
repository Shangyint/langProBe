import requests


class AlfWorldClient:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")

    def _make_request(self, method, endpoint, data=None):
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}

        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method in ["POST", "PUT", "DELETE"]:
            response = requests.request(method, url, headers=headers, json=data)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        response.raise_for_status()
        return response.json()

    def reset(self):
        response = self._make_request("POST", "/reset")
        if not response["status"] == "success":
            raise Exception(response["message"])
        return response["result"]["obs"], response["result"]["info"]

    def step(self, action):
        response = self._make_request("POST", "/step", data={"action": action})
        if not response["status"] == "success":
            raise Exception(response["message"])
        return (
            response["result"]["obs"],
            response["result"]["scores"],
            response["result"]["dones"],
            response["result"]["infos"],
        )
