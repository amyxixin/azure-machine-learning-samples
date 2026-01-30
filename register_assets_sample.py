
from azure.ai.ml import MLClient, Input, load_component
from azure.ai.ml.entities import Environment, Model, BatchEndpoint, PipelineComponentBatchDeployment, BuildContext
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
import json


def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def get_ml_client(config):
    """Create and return Azure ML client."""
    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=config["subscription_id"],
        resource_group_name=config["resource_group"],
        workspace_name=config["workspace_name"]
    )


def register_environment(ml_client, name, dockerfile_path):
    """Register or retrieve an environment."""
    try:
        env = ml_client.environments.get(name=name, label="latest")
    except:
        env = Environment(
            name=name,
            description=f"Environment for {name}",
            build=BuildContext(path=".", dockerfile_path=dockerfile_path)
        )
        env = ml_client.environments.create_or_update(env)
    print(f"✅ Environment: {env.name}:{env.version}")
    return env


def register_component(ml_client, yaml_path):
    """Register or retrieve a component from YAML."""
    component = load_component(source=yaml_path)
    try:
        component = ml_client.components.get(name=component.name, version=component.version)
    except:
        component = ml_client.components.create_or_update(component)
    print(f"✅ Component: {component.name}:{component.version}")
    return component


def main():
    # Load config
    config = load_config("config.json")
    ml_client = get_ml_client(config)

    # Configuration
    cluster_name = config["cluster"]
    experiment = config["experiment"]

    # Register environment
    env = register_environment(ml_client, "iGuard-env", "Dockerfile")

    # Register components
    train_component = register_component(ml_client, "train.yaml")
    score_component = register_component(ml_client, "score.yaml")

    # Get training data
    data = ml_client.data.get(name=config["datasource_name"], label="latest")
    print(f"✅ Data: {data.name}")

    # Create and run training pipeline
    @pipeline(name=f"{experiment}_training", default_compute=cluster_name)
    def training_pipeline(training_data, columns):
        train_job = train_component(
            input_data=training_data,
        )
        return {"output_model": train_job.outputs.output_model}

    # Submit training job
    job = ml_client.jobs.create_or_update(
        training_pipeline(
            training_data=Input(type=AssetTypes.URI_FILE, path=data.id),
        )
    )
    print(f"Training job submitted: {job.studio_url}")

    # Wait for completion
    print(ml_client.jobs.stream(job.name))
    if job.status != "Completed":
        print(f"Training failed: {job.status}")
        return

    # Get train job output and register model
    child_jobs = list(ml_client.jobs.list(parent_job_name=job.name))
    train_job = next((j for j in child_jobs if "train_job" in j.display_name.lower()), child_jobs[0])
    
    model = ml_client.models.create_or_update(Model(
        path=f"azureml://jobs/{train_job.name}/outputs/artifacts/paths/model/",
        name=f"{experiment}-model",
        type=AssetTypes.MLFLOW_MODEL
    ))
    print(f"✅ Model: {model.name}:{model.version}")

    # Create scoring pipeline
    @pipeline(name=f"{experiment}_scoring", default_compute=cluster_name)
    def scoring_pipeline(input_data: Input(type=AssetTypes.URI_FILE)):
        score_job = score_component(
            input_data=input_data,
            model_path=model.id,
        )
        return {"output_data": score_job.outputs.output_data}

    # Create batch endpoint and deployment
    endpoint_name = f"{experiment}-endpoint"
    deployment_name = f"{experiment}-deployment"

    try:
        endpoint = ml_client.batch_endpoints.get(endpoint_name)
    except:
        endpoint = BatchEndpoint(name=endpoint_name)
        ml_client.batch_endpoints.begin_create_or_update(endpoint).result()
    print(f"Endpoint: {endpoint_name}")

    # Create deployment
    pipeline_component = ml_client.components.create_or_update(scoring_pipeline().component)
    deployment = PipelineComponentBatchDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        component=pipeline_component,
        settings={"continue_on_step_failure": False, "default_compute": cluster_name}
    )
    ml_client.batch_deployments.begin_create_or_update(deployment).result()

    # Set as default deployment
    endpoint = ml_client.batch_endpoints.get(endpoint_name)
    endpoint.defaults.deployment_name = deployment_name
    ml_client.batch_endpoints.begin_create_or_update(endpoint).result()
    print(f"Deployment: {deployment_name}")



if __name__ == "__main__":
    main()
