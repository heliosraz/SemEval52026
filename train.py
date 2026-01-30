from data_processing import load_data, read_yaml_file
import torch
from tqdm import tqdm
from sys import argv
from torch.utils.data import DataLoader, Dataset, Subset
from data_structs import WordSenseData, AugWordSenseData
import pandas as pd
from safetensors.torch import save_file, load_file
import os
import models
import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import List, Dict, Tuple, Any
import wandb
import metrics
from transformers import (
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
import atexit
from peft import LoraModel, LoraConfig

"""
To run script:
python main.py "sentence-transformers/all-roberta-large-v1" data/train.json data/dev.json
"""

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("DEVICE: {}".format(device))

if device == torch.device("cuda"):
    tensor_parallel = True
else:
    tensor_parallel = False


os.makedirs("checkpoint", exist_ok=True)
os.environ["TOKENIZERS_PARALLELISM"] = "False"

task_dataset = {
    "eval": WordSenseData,
    "finetuning": AugWordSenseData,
    "pretrain": AugWordSenseData,
}
model_key = {
    "GeneralistModel_nosep": models.GeneralistModel_nosep,
    "GeneralistModel": models.GeneralistModel,
    "PretrainedGeneralistModel": models.PretrainedGeneralistModel,
    "BaselineModule": models.BaselineModule,
    "CrossContextSimilarityModule": models.CrossContextSimilarityModule,
    "SynonymModel": models.SynonymModel,
    "PretrainedSynonymModel": models.PretrainedSynonymModel,
    "GeneralistModelScored": models.GeneralistModelScored,
}
metric_key = {
    "mask": metrics.accuracy,
    "average": metrics.range,
    "interval": metrics.range,
}
loss_key = {
    "CrossEntropyLoss": torch.nn.CrossEntropyLoss,
    "KLDivLoss": torch.nn.KLDivLoss,
}


def load_model(model, path):
    state_dict = load_file(path)

    model.load_state_dict(state_dict, strict=False)
    print(f"Model loaded from {path} (strict=False)")


class Trainer:
    def __init__(
        self,
        name,
        model,
        train_set: Dataset,
        dev_set: Dataset,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
        input_tags: str | List[str] = ["full_context", "judged_meaning"],
        label_tag: str = "label",
        metric_label: str = "label",
        metric=None,
        freeze_schedule: Dict[int, (Tuple[List[torch.Tensor]] | Any)] = {},
        mask: bool = False,
        k: int = 2,
    ):
        self.model_name = name
        self.model = model
        self.train_set = train_set
        self.dev_set = dev_set
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.input_tags = input_tags
        self.label_tag = label_tag
        self.metric_label = metric_label
        self.freeze_schedule = freeze_schedule
        self.compute_metric = metric
        self.mask = mask
        self.k = k
        self.top_k = []
        atexit.register(self.termination_save)

    def one_step(self, batch):
        X_batch = batch
        if self.mask:
            y_pred, mask_keys = self.model(
                X_batch, select=self.input_tags, mask=self.mask
            )
            batch["mask"] = torch.Tensor(mask_keys["mask_ids"])
        else:
            y_pred = self.model(X_batch, select=self.input_tags, mask=self.mask)

        y_labels = (
            torch.stack(batch[self.label_tag], dim=1).float()
            if isinstance(batch[self.label_tag], list)
            else batch[self.label_tag]
        )
        if isinstance(self.loss_fn, torch.nn.CrossEntropyLoss):
            y_labels = y_labels.long()
        elif isinstance(self.loss_fn, torch.nn.KLDivLoss):
            y_pred = torch.log_softmax(y_pred, dim=1)
        loss = self.loss_fn(y_pred, y_labels.to(device))
        y_metric = batch[self.metric_label]
        y_preds = torch.softmax(y_pred, dim=1)
        y_preds = [
            sum([(i + 1) * prob for i, prob in enumerate(pred.tolist())])
            for pred in y_preds
        ]
        metric = self.compute_metric(y_preds, y_metric)
        return loss, metric

    def run(
        self,
        wandb_run,
        n_epochs: int = 100,
        batch_size=64,
        save_weights_plots: bool = True,
        delta=1e-5,
        patience=10,
    ):
        # self.train_set = Subset(self.train_set, range(10))
        train_loader = DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=True,
        )
        # self.dev_set = Subset(self.dev_set, range(10))
        dev_loader = DataLoader(self.dev_set, batch_size=batch_size, shuffle=False)

        delta_hits = 0
        prev_vloss = 1_000_000.0

        best_vloss = 1_000_000.0
        for epoch in tqdm(range(n_epochs), desc="Epochs:", position=0):
            if str(epoch) in self.freeze_schedule:
                # Freeze
                for layer in self.freeze_schedule[str(epoch)]["freeze"]:
                    for name, param in self.model.named_parameters():
                        if layer in name:
                            print(f"Freezing {name}")
                            param.requires_grad = False
                # Unfreeze
                for layer in self.freeze_schedule[str(epoch)]["unfreeze"]:
                    for name, param in self.model.named_parameters():
                        if layer in name:
                            print(f"Unfreezing {name}")
                            param.requires_grad = True
            running_tacc = 0
            running_loss = 0.0
            self.model.train()
            for batch in tqdm(train_loader, desc="Training Batch:", leave=False):
                self.optimizer.zero_grad()
                loss, metric = self.one_step(batch)
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                running_loss += loss.item()
                running_tacc += metric  # accuracy

            avg_loss = running_loss / len(train_loader)
            running_vloss = 0.0

            self.model.eval()
            # Disable gradient computation and reduce memory consumption.
            running_vacc = 0
            with torch.no_grad():
                for v_batch in tqdm(dev_loader, desc="Dev Batch:", leave=False):
                    v_loss, v_metric = self.one_step(v_batch)
                    running_vloss += v_loss.item()
                    running_vacc += v_metric  # acc

            avg_vloss = running_vloss / len(dev_loader)
            print("LOSS train {} dev {}".format(avg_loss, avg_vloss))
            print(
                "ACCURACY train {} dev {}".format(
                    running_tacc / len(self.train_set), running_vacc / len(self.dev_set)
                )
            )

            wandb_run.log(
                {
                    "train_loss": avg_loss,
                    "train_acc": running_tacc / len(self.train_set),
                    "valid_loss": avg_vloss,
                    "valid_acc": running_vacc / len(self.dev_set),
                }
            )

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                print("Logging model...")
                best_vloss = avg_vloss
                state_dict = self.get_state_dict()
                self.top_k.append(state_dict)
                if len(self.top_k) > self.k:
                    self.top_k.pop(0)
                if save_weights_plots:
                    self.plot_linear_weights(
                        [
                            self.model.base_model.K.weight,
                            self.model.base_model.Q.weight,
                            self.model.base_model.V.weight,
                        ],
                        [
                            "model.base_model.K",
                            "model.base_model.Q",
                            "model.base_model.V",
                        ],
                        running_tacc / len(self.train_set),
                        running_vacc / len(self.dev_set),
                        avg_loss,
                        avg_vloss,
                    )
            if np.abs(prev_vloss - avg_vloss) <= delta:
                delta_hits += 1
                if delta_hits == patience:
                    break
            if avg_vloss > 20:
                break
        model_dir = self.save_model(self.top_k)
        return model_dir

    def plot_linear_weights(
        self,
        weights_list,
        layer_names,
        train_acc,
        dev_acc,
        train_loss,
        dev_loss,
        figsize=(18, 5),
    ):

        fig, axes = plt.subplots(1, len(weights_list), figsize=figsize)

        # Plot each weight matrix
        for weights, name, ax in zip(weights_list, layer_names, axes):
            weights = weights.detach().cpu().numpy()

            im = ax.matshow(
                weights,
                cmap="coolwarm",
                aspect="auto",
                vmin=-np.abs(weights).max(),
                vmax=np.abs(weights).max(),
            )
            ax.set_title(
                f"{name}\nShape: {weights.shape}", fontsize=12, fontweight="bold"
            )
            ax.set_xlabel("Input Features")
            ax.set_ylabel("Output Features")

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Create metrics caption
        caption = (
            f"Training: Acc = {train_acc:.4f}, Loss = {train_loss:.4f} | "
            f"Dev: Acc = {dev_acc:.4f}, Loss = {dev_loss:.4f}"
        )

        # Add caption below the plots
        fig.text(
            0.5,
            0.02,
            caption,
            ha="center",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

        plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space for caption
        wandb_run.log({"weights_visualization": wandb.Image(fig)})
        print(f"Figure saved to wandb")
        plt.close("all")

    def get_state_dict(self):
        state_dict = {
            name: param
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        return state_dict

    def save_model(self, stat_dicts):
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = "checkpoint/{}/{}".format(self.model_name, timestamp)
        os.makedirs(model_dir, exist_ok=True)
        for i, state_dict in enumerate(stat_dicts):
            model_fp = os.path.join(
                model_dir, "{}_{}.safetensors".format(self.model_name, i)
            )
            save_file(state_dict, model_fp)
            print("Model saved at {}".format(model_fp))
        return model_dir

    def termination_save(self):
        print("Training Terminated: saving models...")
        self.save_model(self.top_k)


def get_lr_scheduler(optim, total_steps, **kwargs):
    scheduler_type = kwargs["scheduler_type"]
    warmup_ratio = kwargs["warmup_ratio"]
    if scheduler_type == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optim,
            num_warmup_steps=total_steps * warmup_ratio,
            num_training_steps=total_steps,
        )
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optim,
            num_warmup_steps=total_steps * warmup_ratio,
            num_training_steps=total_steps,
        )
    return lr_scheduler


def get_loss_fn(**kwargs):
    loss_type = kwargs["loss_type"]
    loss_reduction = kwargs["loss_reduction"]
    return loss_key[loss_type](reduction=loss_reduction)


def print_parameters(model: torch.nn.Module):
    print("#" * 5 + "NAMED PARAMETERS" + "#" * 5)
    for name, _ in model.named_parameters():
        print(name)


def main(config):
    ## Data Processing
    task = config.training["task"]
    train_df = load_data(config.data["train_data"])
    dev_df = load_data(config.data["val_data"])
    train_set = task_dataset[task](train_df)
    dev_set = task_dataset[task](dev_df)

    ## Model Initialization
    base_model = model_key[config.model["architecture"]]
    encoder = config.model["encoder"]
    if config["model"]["wrapper"]:
        wrapper = model_key[config["model"]["wrapper"]]
        model = wrapper(
            base_type=base_model,
            base_name=encoder,
            max_length=config["model"]["max_len"],
            hidden_sizes=config["model"]["hidden_sizes"],
            d_attn=config["model"]["d_attn"],
            drop_attn=config["model"]["drop_attn"],
            drop_cls=config["model"]["drop_cls"],
            device=device,
        ).to(device)
    else:
        model = base_model(
            model_name=encoder,
            max_length=config["model"]["max_len"],
            drop_cls=config["model"]["drop_cls"],
            device=device,
        ).to(device)
    if config.training["prev_path"]:
        load_model(model, config.training["prev_path"])
        # lora_config = LoraConfig(
        #     r=8,
        #     lora_alpha=32,
        #     target_modules=["word_embeddings", "position_embeddings", "q", "v"],
        # )
        # print(model.named_modules)
        # model = LoraModel(model, lora_config, "default")

    print_parameters(model)

    ## Training Parameters
    loss_fn = get_loss_fn(**config.training)
    optim_params = {
        "betas": tuple(config.training["loss_betas"]),
        "lr": config.training["lr_others"],
        "weight_decay": config.training["weight_decay_other"],
    }

    param_groups = {}
    for group in config.training["param_groups"]:
        param_groups[group] = []
        for name, param in model.named_parameters():
            for layer in config.training["param_groups"][group]["layers"]:
                if layer in name:
                    param_groups[group].append(param)
    optim = torch.optim.AdamW(
        [
            {
                "params": param_groups[group],
                "lr": config.training["param_groups"][group]["lr"],
                "weight_decay": config.training["param_groups"][group]["weight_decay"],
            }
            for group in param_groups
        ],
        **optim_params,
    )
    total_steps = (
        len(train_set)
        // config.training["train_batch_size"]
        * config.training["epochs"]
    )
    lr_scheduler = get_lr_scheduler(optim, total_steps=total_steps, **config.training)

    # Double checking config feasibility
    if (
        (type(loss_fn), config.data["label_tag"])
        == (torch.nn.CrossEntropyLoss, "probs")
        or (type(loss_fn), config.data["label_tag"]) == (torch.nn.KLDivLoss, "mask")
        or (type(loss_fn), config.data["label_tag"]) == (torch.nn.KLDivLoss, "average")
    ):
        raise TypeError(
            "Loss function and label tag mismatch. \
                        Please check the training parameters:\n\
                        type(loss_fun), label_tag = {},{}".format(
                type(loss_fn), config.data["label_tag"]
            )
        )

    # Model Training and Eval
    base_name = encoder.split("/")[-1]
    trainer = Trainer(
        "_".join([config.model["name"], base_name]),
        model,
        train_set,
        dev_set,
        input_tags=config.data["input_tags"],
        label_tag=config.data["label_tag"],
        metric_label=config.data["metric_tag"],
        loss_fn=loss_fn,
        optimizer=optim,
        lr_scheduler=lr_scheduler,
        metric=metric_key[config.data["metric_tag"]],
        freeze_schedule=(
            config.training["freeze_components"]
            if config.training["freeze_components"]
            else {}
        ),
        mask=config.training["masking"],
        k=config.training["save_total_limit"],
    )
    model_path = trainer.run(
        wandb_run,
        config.training["epochs"],
        config.training["train_batch_size"],
        save_weights_plots=False,
    )
    return model_path


if __name__ == "__main__":
    train_config = read_yaml_file(argv[1])
    wandb_run = wandb.init(
        entity="heliosra-n-a",
        project="2026set5",
        settings=wandb.Settings(
            x_disable_stats=True,
            console="off",
            save_code=False,
            quiet=True,
        ),
        config=train_config,
    )

    main(wandb.config)

    wandb.finish()
