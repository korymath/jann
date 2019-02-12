import json
from locust import HttpLocust, TaskSet, task


headers = {
  "Accept": "application/json",
  "Content-Type": "application/json"
}

# Setting up a json data package for post request.
data = {
    "queryResult": {
      "queryText": "locust"
    }
  }


class UserBehavior(TaskSet):
    @task(1)
    def post_to_reply(self):
        self.client.post("/model_inference", json.dumps(data), headers=headers)


class WebsiteUser(HttpLocust):
    task_set = UserBehavior
    min_wait = 5000
    max_wait = 9000
