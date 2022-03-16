from kfp.v2 import compiler
from kfp.v2.dsl import pipeline, component, Artifact, Dataset, Input, Metrics, Model, Output, OutputPath


PROJECT_ID = "YOUR_PROJECT_ID"
BUCKET_NAME = "gs://YOUR_BUCKET_NAME/"
PIPELINE_ROOT = f"{BUCKET_NAME}pipeline_root/"


@component(
    packages_to_install=["google-cloud-bigquery", "pandas", "pyarrow"],
    base_image="python:3.9",
    output_component_file="get_crime_dataset.yaml"
)
def get_dataframe(
        query_string: str,
        year_query_param: int,
        output_data_path: OutputPath("Dataset")
):
    from google.cloud import bigquery

    bqclient = bigquery.Client(project="YOUR_PROJECT")

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("year", "INT64", year_query_param)
        ]
    )

    df = (
        bqclient.query(query_string, job_config=job_config).result().to_dataframe(create_bqstorage_client=True)
    )

    print(f"Dataset shape: {df.shape}")

    df.to_csv(output_data_path)


@component(
    packages_to_install=["sklearn", "pandas", "joblib"],
    base_image="python:3.9",
    output_component_file="model_component.yaml",
)
def train_model(
        dataset: Input[Dataset],
        metrics: Output[Metrics],
        model: Output[Model]
):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from joblib import dump
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    df = pd.read_csv(dataset.path)

    X = df[["primary_type", "location_description", "domestic"]].copy()
    y = df["arrest"].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    random_forest_model = Pipeline([('vec', OneHotEncoder(sparse=False, handle_unknown="ignore")),
                                    ('clf', RandomForestClassifier())])

    random_forest_model.fit(X_train, y_train)

    acc = accuracy_score(y_test, random_forest_model.predict(X_test))
    f1 = f1_score(y_test, random_forest_model.predict(X_test))
    metrics.log_metric("dataset_shape", df.shape)
    metrics.log_metric("accuracy", acc)
    metrics.log_metric("f1", f1)
    dump(random_forest_model, model.path + ".joblib")


@component(
    packages_to_install=["google-cloud-aiplatform"],
    base_image="python:3.9",
    output_component_file="deploy_component.yaml",
)
def deploy_model(
        model: Input[Model],
        project: str,
        vertex_endpoint: Output[Artifact],
        vertex_model: Output[Model]
):
    from google.cloud import aiplatform

    aiplatform.init(project=project)

    deployed_model = aiplatform.Model.upload(
        display_name="chicago-crime-pipeline",
        artifact_uri=model.uri.replace("/model", "/"),
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
    )
    endpoint = deployed_model.deploy(machine_type="n1-standard-8")

    vertex_endpoint.uri = endpoint.resource_name
    vertex_model.uri = deployed_model.resource_name


@pipeline(
    pipeline_root=PIPELINE_ROOT,
    name="chicago-crime-pipeline",
)
def pipeline(
        query_string: str,
        year_query_param: int,
        output_data_path: str = "crime_data.csv",
        project: str = PROJECT_ID
):
    dataset_task = get_dataframe(query_string, year_query_param)

    model_task = train_model(
        dataset=dataset_task.output
    )

    deploy_task = deploy_model(
        model=model_task.outputs["model"],
        project=project
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path="chicago_crime_model_pipeline.json"
    )