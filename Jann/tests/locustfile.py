import json
from locust import task
from locust import between
from locust import HttpUser

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


class WebsiteUser(HttpUser):
  wait_time = between(5, 9)
  @task(1)
  def post_to_reply(self):
    self.client.post("/model_inference", json.dumps(data), headers=headers)
