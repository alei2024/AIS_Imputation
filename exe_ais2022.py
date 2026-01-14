import argparse
import datetime
import json
import os

import torch
import yaml

from dataset_ais import get_dataloader
from main_model import CSDI_AIS
from utils import evaluate, train


parser = argparse.ArgumentParser(description="CSDI - AIS2022-12")
parser.add_argument("--config", type=str, default="base_ais.yaml")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.1)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/ais2022_12_" + current_time + "/"
print("model folder:", foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

data_cfg = config.get("data", {})
train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
    seed=args.seed,
    batch_size=config["train"]["batch_size"],
    missing_ratio=config["model"]["test_missing_ratio"],
    eval_length=int(data_cfg.get("eval_length", 144)),
    freq=str(data_cfg.get("freq", "10min")),
    stride=int(data_cfg.get("stride", data_cfg.get("eval_length", 144))),
    min_observed_ratio=float(data_cfg.get("min_observed_ratio", 0.2)),
    mmsi_keep_ratio=float(data_cfg.get("mmsi_keep_ratio", 1.0)),
    max_ships=int(data_cfg.get("max_ships", 0)),
    max_windows_per_ship=int(data_cfg.get("max_windows_per_ship", 0)),
    valid_ratio=float(data_cfg.get("valid_ratio", 0.1)),
    test_ratio=float(data_cfg.get("test_ratio", 0.1)),
)

model = CSDI_AIS(config, args.device).to(args.device)

if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

scaler = scaler.to(args.device)
mean_scaler = mean_scaler.to(args.device)
evaluate(
    model,
    test_loader,
    nsample=args.nsample,
    scaler=scaler,
    mean_scaler=mean_scaler,
    foldername=foldername,
)

