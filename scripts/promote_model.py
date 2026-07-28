import os
import mlflow

def get_model_accuracy(client, model_name, stage):
    """
    Get the accuracy of the latest model in a stage.
    """

    versions = client.get_latest_versions(
        model_name,
        stages=[stage]
    )

    if len(versions) == 0:
        return None, None

    version = versions[0]

    run = client.get_run(version.run_id)

    accuracy = run.data.metrics.get("accuracy", None)

    return version, accuracy


def promote_model():

    dagshub_token = os.getenv("CAPSTONE_TEST")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    mlflow.set_tracking_uri(
        "https://dagshub.com/krunalahir/MLOPS--Capstone-project.mlflow"
    )

    client = mlflow.MlflowClient()

    model_name = "my_model"

    staging_version, staging_acc = get_model_accuracy(
        client,
        model_name,
        "Staging"
    )

    production_version, production_acc = get_model_accuracy(
        client,
        model_name,
        "Production"
    )

    # First deployment
    if production_version is None:

        client.transition_model_version_stage(
            name=model_name,
            version=staging_version.version,
            stage="Production"
        )

        print("First production model deployed")

        return

    print(f"Production Accuracy : {production_acc}")
    print(f"Staging Accuracy    : {staging_acc}")

    if staging_acc > production_acc:

        client.transition_model_version_stage(
            name=model_name,
            version=production_version.version,
            stage="Archived"
        )

        client.transition_model_version_stage(
            name=model_name,
            version=staging_version.version,
            stage="Production"
        )

        print("New model promoted to Production")

    else:

        print("Current Production model is better.")

        client.transition_model_version_stage(
            name=model_name,
            version=staging_version.version,
            stage="Archived"
        )

if __name__ == "__main__":
    promote_model()