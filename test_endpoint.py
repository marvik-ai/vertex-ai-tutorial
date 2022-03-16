from google.cloud import aiplatform

def endpoint_predict_sample(
    project: str, location: str, instances: list, endpoint: str
):
    aiplatform.init(project=project, location=location)

    endpoint = aiplatform.Endpoint(endpoint)

    prediction = endpoint.predict(instances=instances)
    print(prediction)
    return prediction


if __name__ == "__main__":
    instances_to_test = [
        ["OBSCENITY", "RESIDENCE", "false"]
    ]
    endpoint_predict_sample(
        project="346590416306",
        endpoint="1937295507676200960",
        location="us-central1",
        instances=instances_to_test
    )
