import argparse
import os
from pathlib import Path
import torch
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
        help="Path to the model yaml config file (e.g., config/model_resnet50.yaml)"
    )
    args = parser.parse_args()

    # Load model configurations
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print(f"Loaded configuration: {config['experiment_name']}")

    #  data paths
    data_dir = get_data_dir(path=config["data"]["path"])

    t_wbc_data = DataPrep(
        name="Trophozoite plus WBC dataset",
        root_path=data_dir,
        batch_size=config["hyperparameters"]["batch_size"],
        pin_memory=config["hyperparameters"]["pin_memory"],
        num_workers=config["hyperparameters"]["num_workers"],
    )

    train, val, test = t_wbc_data.build_loaders();

    print("Beginning execution pipeline setup...")
    retina_wrapper = RetinaNet(
        num_classes=config["model"]["num_classes"], pre_trained=False
    )

    retina_model = retina_wrapper.get_model()
        
    t_wbc_trainer = ModelTrainer(model = retina_model, train_loader=train, val_loader=val, lr=config['hyperparameters']['lr'])
    
    num_epochs = config['hyperparameters']['num_epochs']
    
    
    for epoch in range(num_epochs):
        results = t_wbc_trainer.run_epoch(epoch_idx=epoch)
        
        print("EPOCH RESULTS", results)
        
if __name__ == "__main__":
    main()
