import pickle
import os
import mlflow.pytorch
import torch
import torch.nn as nn
from mlflow import log_artifact, log_metrics, log_param, log_params
from torch.backends import cudnn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import sys

sys.path.append("..")
from models.CNN_Transformer_Mixtureoutput import (
    CNN,
    MDN,
    MLP,
    PositionEmbedding_Vs30,
    TransformerEncoder,
    full_model,
)
from utils.multiple_sta_dataset import multiple_station_dataset

"""
set up mlflow experiment:
In Terminal, you need to type
"mlflow ui"

enter to UI at local host
create an experiment, its name: "bias to close station"

connect mlflow ui :
mlflow server --host localhost --port 5000 

"""

mlflow.set_tracking_uri("http://localhost:5000")


def train_process(
    full_model_instance,
    full_data,
    optimizer,
    hyper_param,
    num_of_gaussian=5,
    train_data_size=0.8,
    experiment_name=None,
    run_name=None,
):
    """Train and validate the multi-station model with MLflow logging and early stopping."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    with mlflow.start_run(
        run_name=run_name,
        experiment_id=experiment.experiment_id,
    ) as run:
        log_params(
            {
                "epochs": hyper_param["num_epochs"],
                "batch size": hyper_param["batch_size"],
                "learning rate": hyper_param["learning_rate"],
            }
        )
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        cudnn.benchmark = True

        train_set_size = int(len(full_data) * train_data_size)
        valid_set_size = len(full_data) - train_set_size
        torch.manual_seed(0)
        train_dataset, val_dataset = random_split(
            full_data, [train_set_size, valid_set_size]
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=hyper_param["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=5,
            drop_last=True,
        )
        valid_loader = DataLoader(
            dataset=val_dataset,
            batch_size=hyper_param["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=5,
            drop_last=True,
        )

        gaussian_loss = nn.GaussianNLLLoss(reduction="none")
        training_loss = []
        validation_loss = []
        print(f'train {hyper_param["num_epochs"]} times')
        the_last_loss = 100
        if hyper_param["learning_rate"] >= 5e-05:
            patience = 10
        elif hyper_param["learning_rate"] >= 0:
            patience = 15
        print("patience", patience)
        trigger_times = 0
        for epoch in range(hyper_param["num_epochs"]):
            print(f"Epoch:{epoch+1}")
            print("--------------------train_start--------------------")
            for sample in tqdm(train_loader):
                optimizer.zero_grad()
                weight, sigma, mu = full_model_instance(sample)
                pga_label = (
                    sample["label"]
                    .reshape(hyper_param["batch_size"], full_data.label_target, 1)
                    .cuda()
                )
                mask = ~pga_label.eq(0)

                pga_label_masked = torch.masked_select(pga_label, mask).reshape(-1, 1)
                weight_masked = torch.masked_select(weight, mask).reshape(
                    -1, num_of_gaussian
                )
                sigma_masked = torch.masked_select(sigma, mask).reshape(
                    -1, num_of_gaussian
                )
                mu_masked = torch.masked_select(mu, mask).reshape(-1, num_of_gaussian)
                train_loss = torch.mean(
                    torch.sum(
                        weight_masked
                        * gaussian_loss(mu_masked, pga_label_masked, sigma_masked),
                        axis=1,
                    )
                ).cuda()
                train_loss.backward()
                optimizer.step()
            print("train_loss", train_loss)
            training_loss.append(train_loss.data)

            for sample in tqdm(valid_loader):
                weight, sigma, mu = full_model_instance(sample)

                pga_label = (
                    sample["label"]
                    .reshape(hyper_param["batch_size"], full_data.label_target, 1)
                    .cuda()
                )
                mask = ~pga_label.eq(0)

                pga_label_masked = torch.masked_select(pga_label, mask).reshape(-1, 1)
                weight_masked = torch.masked_select(weight, mask).reshape(
                    -1, num_of_gaussian
                )
                sigma_masked = torch.masked_select(sigma, mask).reshape(
                    -1, num_of_gaussian
                )
                mu_masked = torch.masked_select(mu, mask).reshape(-1, num_of_gaussian)
                val_loss = torch.mean(
                    torch.sum(
                        weight_masked
                        * gaussian_loss(mu_masked, pga_label_masked, sigma_masked),
                        axis=1,
                    )
                ).cuda()
            print("val_loss", val_loss)
            validation_loss.append(val_loss.data)
            log_metrics(
                {"train_loss": train_loss.item(), "val_loss": val_loss.item()},
                step=epoch + 1,
            )
            if train_loss.data < -1 and (epoch + 1) % 5 == 0:
                checkpoint_path = (
                    f"../../saved_models/model{hyper_param['model_index']}_checkpoints"
                )
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)
                torch.save(
                    full_model_instance.state_dict(),
                    f"{checkpoint_path}/epoch{epoch+1}_model.pt",
                )
            current_loss = val_loss.data
            if the_last_loss < -1:
                patience = 15
            if current_loss > the_last_loss:
                trigger_times += 1
                print("early stop trigger times:", trigger_times)

                if trigger_times >= patience:
                    path = "../../saved_models"
                    print(f"Early stopping! stop at epoch: {epoch+1}")
                    with open(
                        f"{path}/train loss{hyper_param['model_index']}", "wb"
                    ) as fp:
                        pickle.dump(training_loss, fp)
                        log_artifact(f"{path}/train loss{hyper_param['model_index']}")
                    with open(
                        f"{path}/validation loss{hyper_param['model_index']}", "wb"
                    ) as fp:
                        pickle.dump(validation_loss, fp)
                        log_artifact(
                            f"{path}/validation loss{hyper_param['model_index']}"
                        )
                    log_param("epoch early stop", epoch + 1)
                    return training_loss, validation_loss

                continue

            else:
                print("trigger 0 time")
                trigger_times = 0
                path = "../../saved_models"
                model_file = f"{path}/model{hyper_param['model_index']}_vel.pt"
                torch.save(full_model_instance.state_dict(), model_file)
                log_artifact(model_file)

            the_last_loss = current_loss
        print(
            "Train Epoch: {}/{} Traing_Loss: {} Val_Loss: {}".format(
                epoch + 1, hyper_param["num_epochs"], train_loss.data, val_loss.data
            )
        )


if __name__ == "__main__":
    train_data_size = 0.8
    model_index = 1
    num_epochs = 300
    for batch_size in [16]:
        for learning_rate in [5e-05, 2.5e-05]:
            for run_index in range(3):
                model_index += 1
                hyper_param = {
                    "model_index": model_index,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                }
                print(f"learning rate: {learning_rate}")
                print(f"batch size: {batch_size}")
                num_of_gaussian = 5
                emb_dim = 150
                mlp_dims = (150, 100, 50, 30, 10)

                cnn_model = CNN(downsample=3, mlp_input=7665).cuda()
                pos_emb_model = PositionEmbedding_Vs30(emb_dim=emb_dim).cuda()
                transformer_model = TransformerEncoder()
                mlp_model = MLP(input_shape=(emb_dim,), dims=mlp_dims).cuda()
                mdn_model = MDN(input_shape=(mlp_dims[-1],)).cuda()

                full_model_instance = full_model(
                    cnn_model,
                    pos_emb_model,
                    transformer_model,
                    mlp_model,
                    mdn_model,
                    pga_targets=25,
                    data_length=4000,
                )
                optimizer = torch.optim.Adam(
                    [
                        {"params": cnn_model.parameters()},
                        {"params": transformer_model.parameters()},
                        {"params": mlp_model.parameters()},
                        {"params": mdn_model.parameters()},
                    ],
                    lr=learning_rate,
                )
                full_data = multiple_station_dataset(
                    "../../data/processed/TSMIP_2016_demo.hdf5",
                    mode="train",
                    mask_waveform_sec=3,
                    weight_label=False,
                    oversample=1.25,
                    oversample_mag=4,
                    test_year=2017,
                    mask_waveform_random=True,
                    mag_threshold=0,
                    label_key="pgv",
                    input_type="vel",
                    data_length_sec=20,
                    station_blind=True,
                    bias_to_closer_station=True,
                )
                training_loss, validation_loss = train_process(
                    full_model_instance,
                    full_data,
                    optimizer,
                    hyper_param,
                    experiment_name="TT-SAM Training",
                    run_name=f"{model_index} model train",
                )