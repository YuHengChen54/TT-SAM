import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

sys.path.append("../..")
from src.models.CNN_Transformer_Mixtureoutput import (
    CNN,
    MDN,
    MLP,
    PositionEmbedding_Vs30,
    TransformerEncoder,
    full_model,
)
from src.utils.multiple_sta_dataset import multiple_station_dataset
from analysis.model_analysis import Intensity_Plotter


def run_prediction_pipeline():
    """Run ensemble prediction and save result tables and figures for each mask window."""
    for mask_sec in [3, 5, 7, 10, 13, 15]:
        mask_after_sec = mask_sec
        label_key = "pgv"
        dataset = multiple_station_dataset(
            "../../data/processed/TSMIP_2016_demo.hdf5",
            mode="test",
            mask_waveform_sec=mask_after_sec,
            test_year=2016,
            label_key=label_key,
            mag_threshold=0,
            input_type="vel",
            data_length_sec=20,
        )

        device = torch.device("cuda")
        for model_name in ["TTSAM_Official"]:
            model_path = f"../../saved_models/{model_name}.pt"
            emb_dim = 150
            mlp_dims = (150, 100, 50, 30, 10)
            cnn_model = CNN(downsample=3, mlp_input=7665).cuda()
            pos_emb_model = PositionEmbedding_Vs30(emb_dim=emb_dim).cuda()
            transformer_model = TransformerEncoder().cuda()
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
            ).to(device)
            full_model_instance.load_state_dict(torch.load(model_path))
            loader = DataLoader(dataset=dataset, batch_size=1)

            mixture_mu = []
            labels = []
            p_picks = []
            eq_ids = []
            label_times = []
            latitudes = []
            longitudes = []
            elevations = []
            for batch_index, sample in tqdm(enumerate(loader)):
                picks = sample["p_picks"].flatten().numpy().tolist()
                label_time = sample[f"{label_key}_time"].flatten().numpy().tolist()
                lat = sample["target"][:, :, 0].flatten().tolist()
                lon = sample["target"][:, :, 1].flatten().tolist()
                elev = sample["target"][:, :, 2].flatten().tolist()
                p_picks.extend(picks)
                p_picks.extend([np.nan] * (25 - len(picks)))
                label_times.extend(label_time)
                label_times.extend([np.nan] * (25 - len(label_time)))
                latitudes.extend(lat)
                longitudes.extend(lon)
                elevations.extend(elev)

                eq_id = sample["EQ_ID"][:, :, 0].flatten().numpy().tolist()
                eq_ids.extend(eq_id)
                eq_ids.extend([np.nan] * (25 - len(eq_id)))
                weight, sigma, mu = full_model_instance(sample)

                weight = weight.cpu()
                sigma = sigma.cpu()
                mu = mu.cpu()
                if batch_index == 0:
                    mixture_mu = torch.sum(weight * mu, dim=2).cpu().detach().numpy()
                    labels = sample["label"].cpu().detach().numpy()
                else:
                    mixture_mu = np.concatenate(
                        [mixture_mu, torch.sum(weight * mu, dim=2).cpu().detach().numpy()],
                        axis=1,
                    )
                    labels = np.concatenate(
                        [labels, sample["label"].cpu().detach().numpy()], axis=1
                    )
            labels = labels.flatten()
            mixture_mu = mixture_mu.flatten()

            output = {
                "EQ_ID": eq_ids,
                "p_picks": p_picks,
                f"{label_key}_time": label_times,
                "predict": mixture_mu,
                "answer": labels,
                "latitude": latitudes,
                "longitude": longitudes,
                "elevation": elevations,
            }
            output_df = pd.DataFrame(output)
            output_df = output_df[output_df["answer"] != 0]
            os.makedirs(f"../../result/tables/model_{model_name}_analysis", exist_ok=True)
            output_df.to_csv(
                f"../../result/tables/model_{model_name}_analysis/model {model_name} {mask_after_sec} sec prediction_vel.csv",
                index=False,
            )

            fig, ax = Intensity_Plotter.plot_true_predicted(
                y_true=output_df["answer"],
                y_pred=output_df["predict"],
                aggregation_method="point",
                point_size=12,
                target_metric=label_key,
            )

            ax.set_title(
                f"{mask_after_sec}s True Predict Plot, 2016 data model {model_name}",
                fontsize=20,
                pad=25,
            )

            os.makedirs(f"../../result/figures/model_{model_name}_analysis", exist_ok=True)
            fig.savefig(
                f"../../result/figures/model_{model_name}_analysis/model {model_name} {mask_after_sec} sec_vel.png"
            )


if __name__ == "__main__":
    run_prediction_pipeline()
