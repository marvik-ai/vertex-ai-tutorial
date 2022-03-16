if __name__ == "__main__":
    from datetime import datetime
    from google.cloud import aiplatform

    aiplatform.init(project="YOUR_PROJECT", staging_bucket="YOUR_BUCKET")
    TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

    with open("./crime_query.sql", "r") as query_file:
        query_string = query_file.read()

    print(TIMESTAMP)
    run = aiplatform.PipelineJob(
        display_name="chicago-crime-model-pipeline",
        template_path="chicago_crime_model_pipeline.json",
        job_id="chicago-crime-model-pipeline-{0}".format(TIMESTAMP),
        parameter_values={"query_string": query_string, "year_query_param": 2001},
        enable_caching=False
    )
    run.submit()
