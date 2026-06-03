import argparse
import os
import wandb
from pathlib import Path
import yaml
from data_loader import DataPrep
from models import RetinaNet
from train import ModelTrainer


def get_data_dir(path: Path = Path("../data")) -> Path:
    """
    Auto-detects if running on Compute Canada via SLURM environment variables.
    Falls back to local path if running on a local machine.
    """

    slurm_tmp = os.environ.get("SLURM_TMPDIR")
    if slurm_tmp:
        print(f"Using Compute Canada. Reading data from: {slurm_tmp}")
        return Path(slurm_tmp) / "data"
    else:
        print(f"Running on local machine detected. Reading data from: {path}")
        return Path(path)


def main():
    parser = argparse.ArgumentParser(description="Malaria Detection Training Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the model yaml config file (e.g., config/model_resnet50.yaml)",
    )
    args = parser.parse_args()

    # Load model configurations
    with open(args.config, "r") as f:
        job_config = yaml.safe_load(f)

    print(f"Loaded configuration: {job_config['experiment_name']}")

    # initializing wandb
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="infernowolf99-university-for-development-studies",
        # Set the wandb project where this run will be logged.
        project="malaria-detection-testing",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": job_config["hyperparameters"]["lr"],
            "architecture": "RetinaNet",
            "dataset": "Malaria-test",
            "epochs": job_config["hyperparameters"]["num_epochs"],
        },
    )

    #  data paths
    data_dir = get_data_dir(path=job_config["data"]["path"])

    t_wbc_data = DataPrep(
        name="Trophozoite plus WBC dataset",
        root_path=data_dir,
        batch_size=job_config["hyperparameters"]["batch_size"],
        pin_memory=job_config["hyperparameters"]["pin_memory"],
        num_workers=job_config["hyperparameters"]["num_workers"],
    )

    train, val, test = t_wbc_data.build_loaders()

    print("Beginning execution pipeline setup...")

    retina_wrapper = RetinaNet(
        num_classes=job_config["model"]["num_classes"], pre_trained=False
    )

    retina_model = retina_wrapper.get_model()

    checkpoint_path = job_config["checkpoints"]["path"]
    checkpoint_name = job_config["checkpoints"]["name"]
    checkpoint_file = os.path.join(checkpoint_path, checkpoint_name)

    t_wbc_trainer = ModelTrainer(
        model=retina_model,
        train_loader=train,
        val_loader=val,
        lr=job_config["hyperparameters"]["lr"],
        checkpoint_name=checkpoint_name,
        checkpoint_path=checkpoint_path,
    )

    start_epoch = t_wbc_trainer.load_checkpoint(checkpoint_file=checkpoint_file)

    num_epochs = job_config["hyperparameters"]["num_epochs"]

    for epoch in range(start_epoch, num_epochs):
        results = t_wbc_trainer.run_epoch(epoch_idx=epoch)

        run.log(
            {
                "epoch": results["epoch"],
                "train_loss": results["train_loss"],
                "val_loss": results["val_loss"],
                "mAP_50": results["mAP_50"],
            }
        )

        print("EPOCH RESULTS", results)

    run.finish()


if __name__ == "__main__":
    main()
