import numpy as np
import torch
import torch.nn as nn


class LambdaLayer(nn.Module):
    """Apply a custom tensor transform with a small epsilon offset."""

    def __init__(self, lambd, eps=1e-4):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
        self.eps = eps

    def forward(self, x):
        return self.lambd(x) + self.eps


class MLP(nn.Module):
    """Build a configurable multi-layer perceptron."""

    def __init__(
        self,
        input_shape,
        dims=(500, 300, 200, 150),
        activation=nn.ReLU(),
        last_activation=None,
    ):
        super(MLP, self).__init__()
        if last_activation is None:
            last_activation = activation
        self.dims = dims
        self.first_fc = nn.Linear(input_shape[0], dims[0])
        self.first_activation = activation

        more_hidden = []
        if len(self.dims) > 2:
            for index in range(1, len(self.dims) - 1):
                more_hidden.append(nn.Linear(self.dims[index - 1], self.dims[index]))
                more_hidden.append(nn.ReLU())

        self.more_hidden = nn.ModuleList(more_hidden)

        self.last_fc = nn.Linear(dims[-2], dims[-1])
        self.last_activation = last_activation

    def forward(self, x):
        output = self.first_fc(x)
        output = self.first_activation(output)
        if self.more_hidden:
            for layer in self.more_hidden:
                output = layer(output)
        output = self.last_fc(output)
        output = self.last_activation(output)
        return output


class CNN(nn.Module):
    """Extract waveform features using CNN blocks and an MLP head."""

    def __init__(
        self,
        input_shape=(-1, 6000, 3),
        activation=nn.ReLU(),
        downsample=1,
        mlp_input=11665,
        mlp_dims=(500, 300, 200, 150),
        eps=1e-8,
    ):
        super(CNN, self).__init__()
        self.input_shape = input_shape
        self.activation = activation
        self.downsample = downsample
        self.mlp_input = mlp_input
        self.mlp_dims = mlp_dims
        self.eps = eps

        self.lambda_layer_1 = LambdaLayer(
            lambda t: t
            / (
                torch.max(
                    torch.max(torch.abs(t), dim=1, keepdim=True).values,
                    dim=2,
                    keepdim=True,
                ).values
                + self.eps
            )
        )
        self.unsqueeze_layer1 = LambdaLayer(lambda t: torch.unsqueeze(t, dim=1))
        self.lambda_layer_2 = LambdaLayer(
            lambda t: torch.log(
                torch.max(torch.max(torch.abs(t), dim=1).values, dim=1).values
                + self.eps
            )
            / 100
        )
        self.unsqueeze_layer2 = LambdaLayer(lambda t: torch.unsqueeze(t, dim=1))

        self.conv2d1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, downsample), stride=(1, downsample)),
            nn.ReLU(),
        )

        self.conv2d2 = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=(16, 3), stride=(1, 3)), nn.ReLU()
        )

        self.conv1d1 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=16), nn.ReLU())
        self.maxpooling = nn.MaxPool1d(2)

        self.conv1d2 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=16), nn.ReLU())
        self.conv1d3 = nn.Sequential(nn.Conv1d(128, 32, kernel_size=8), nn.ReLU())
        self.conv1d4 = nn.Sequential(nn.Conv1d(32, 32, kernel_size=8), nn.ReLU())
        self.conv1d5 = nn.Sequential(nn.Conv1d(32, 16, kernel_size=4), nn.ReLU())
        self.mlp = MLP((self.mlp_input,), dims=self.mlp_dims)

    def forward(self, x):
        # print("intitial shape", x.size())
        output = self.lambda_layer_1(x)
        output = self.unsqueeze_layer1(output)

        scale = self.lambda_layer_2(x)
        scale = self.unsqueeze_layer2(scale)

        output = self.conv2d1(output)
        output = self.conv2d2(output)
        output = torch.squeeze(output, dim=-1)
        output = self.conv1d1(output)
        output = self.maxpooling(output)
        output = self.conv1d2(output)
        output = self.maxpooling(output)
        output = self.conv1d3(output)
        output = self.maxpooling(output)
        output = self.conv1d4(output)
        output = self.conv1d5(output)
        output = torch.flatten(output, start_dim=1)
        output = torch.cat((output, scale), dim=1)
        output = self.mlp(output)

        return output


class PositionEmbedding_Vs30(
    nn.Module
):
    """Create sinusoidal embeddings from station metadata including Vs30."""

    def __init__(
        self,
        wavelengths=((5, 30), (110, 123), (0.01, 5000), (100, 1600)),
        emb_dim=500,
        **kwargs
    ):
        super(PositionEmbedding_Vs30, self).__init__(**kwargs)
        self.wavelengths = wavelengths
        self.emb_dim = emb_dim

        min_lat, max_lat = wavelengths[0]
        min_lon, max_lon = wavelengths[1]
        min_depth, max_depth = wavelengths[2]
        min_vs30, max_vs30 = wavelengths[3]
        assert emb_dim % 10 == 0
        lat_dim = emb_dim // 5
        lon_dim = emb_dim // 5
        depth_dim = emb_dim // 10
        vs30_dim = emb_dim // 10

        self.lat_coeff = (
            2
            * np.pi
            * 1.0
            / min_lat
            * ((min_lat / max_lat) ** (np.arange(lat_dim) / lat_dim))
        )
        self.lon_coeff = (
            2
            * np.pi
            * 1.0
            / min_lon
            * ((min_lon / max_lon) ** (np.arange(lon_dim) / lon_dim))
        )
        self.depth_coeff = (
            2
            * np.pi
            * 1.0
            / min_depth
            * ((min_depth / max_depth) ** (np.arange(depth_dim) / depth_dim))
        )
        self.vs30_coeff = (
            2
            * np.pi
            * 1.0
            / min_vs30
            * ((min_vs30 / max_vs30) ** (np.arange(vs30_dim) / vs30_dim))
        )

        lat_sin_mask = np.arange(emb_dim) % 5 == 0
        lat_cos_mask = np.arange(emb_dim) % 5 == 1
        lon_sin_mask = np.arange(emb_dim) % 5 == 2
        lon_cos_mask = np.arange(emb_dim) % 5 == 3
        depth_sin_mask = np.arange(emb_dim) % 10 == 4
        depth_cos_mask = np.arange(emb_dim) % 10 == 9
        vs30_sin_mask = np.arange(emb_dim) % 10 == 5
        vs30_cos_mask = np.arange(emb_dim) % 10 == 8

        self.mask = np.zeros(emb_dim)
        self.mask[lat_sin_mask] = np.arange(lat_dim)
        self.mask[lat_cos_mask] = lat_dim + np.arange(lat_dim)
        self.mask[lon_sin_mask] = 2 * lat_dim + np.arange(lon_dim)
        self.mask[lon_cos_mask] = 2 * lat_dim + lon_dim + np.arange(lon_dim)
        self.mask[depth_sin_mask] = 2 * lat_dim + 2 * lon_dim + np.arange(depth_dim)
        self.mask[depth_cos_mask] = (
            2 * lat_dim + 2 * lon_dim + depth_dim + np.arange(depth_dim)
        )
        self.mask[vs30_sin_mask] = (
            2 * lat_dim + 2 * lon_dim + 2 * depth_dim + np.arange(vs30_dim)
        )
        self.mask[vs30_cos_mask] = (
            2 * lat_dim + 2 * lon_dim + 2 * depth_dim + vs30_dim + np.arange(vs30_dim)
        )
        self.mask = self.mask.astype("int32")

    def forward(self, x):
        lat_base = (
            x[:, :, 0:1].cuda() * torch.Tensor(self.lat_coeff).cuda()
        )
        lon_base = x[:, :, 1:2].cuda() * torch.Tensor(self.lon_coeff).cuda()
        depth_base = x[:, :, 2:3].cuda() * torch.Tensor(self.depth_coeff).cuda()
        vs30_base = x[:, :, 3:4] * torch.Tensor(self.vs30_coeff).cuda()
        output = torch.cat(
            [
                torch.sin(lat_base),
                torch.cos(lat_base),
                torch.sin(lon_base),
                torch.cos(lon_base),
                torch.sin(depth_base),
                torch.cos(depth_base),
                torch.sin(vs30_base),
                torch.cos(vs30_base),
            ],
            dim=-1,
        )
        mask_tensor = torch.from_numpy(np.array(self.mask)).long()
        gather_index = (
            (mask_tensor.unsqueeze(0).unsqueeze(0))
            .expand(x.shape[0], 1, self.emb_dim)
            .cuda()
        )
        output = torch.gather(output, -1, gather_index).cuda()
        return output


class TransformerEncoder(nn.Module):
    """Encode token sequences with stacked transformer encoder layers."""

    def __init__(
        self,
        d_model=150,
        nhead=10,
        batch_first=True,
        activation="gelu",
        dropout=0.0,
        dim_feedforward=1000,
    ):
        super(TransformerEncoder, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=batch_first,
            activation=activation,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
        ).cuda()
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, 6).cuda()

    def forward(self, x, src_key_padding_mask=None):
        out = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return out


class MDN(nn.Module):
    """Predict Gaussian mixture parameters from input features."""

    def __init__(self, input_shape=(150,), n_hidden=20, n_gaussians=5):
        super(MDN, self).__init__()
        self.z_h = nn.Sequential(nn.Linear(input_shape[0], n_hidden), nn.Tanh())
        self.z_weight = nn.Linear(n_hidden, n_gaussians)
        self.z_sigma = nn.Linear(n_hidden, n_gaussians)
        self.z_mu = nn.Linear(n_hidden, n_gaussians)

    def forward(self, x):
        z_h = self.z_h(x)
        weight = nn.functional.softmax(self.z_weight(z_h), -1)
        sigma = torch.exp(self.z_sigma(z_h))
        mu = self.z_mu(z_h)
        return weight, sigma, mu


class full_model(nn.Module):
    """Compose all submodules for end-to-end MDN prediction."""

    def __init__(
        self,
        model_CNN,
        model_Position,
        model_Transformer,
        model_mlp,
        model_MDN,
        max_station=25,
        pga_targets=15,
        emb_dim=150,
        data_length=6000,
    ):
        super(full_model, self).__init__()
        self.data_length = data_length
        self.model_CNN = model_CNN
        self.model_Position = model_Position
        self.model_Transformer = model_Transformer
        self.model_mlp = model_mlp
        self.model_MDN = model_MDN
        self.max_station = max_station
        self.pga_targets = pga_targets
        self.emb_dim = emb_dim

    def forward(self, data):
        cnn_output = self.model_CNN(
            torch.DoubleTensor(data["waveform"].reshape(-1, self.data_length, 9))
            .float()
            .cuda()
        )
        cnn_output_reshape = torch.reshape(
            cnn_output, (-1, self.max_station, self.emb_dim)
        )
        emb_output = self.model_Position(
            torch.DoubleTensor(data["sta"].reshape(-1, 1, data["sta"].shape[2]))
            .float()
            .cuda()
        )
        emb_output = emb_output.reshape(-1, self.max_station, self.emb_dim)
        station_pad_mask = data["sta"] == 0
        station_pad_mask = torch.all(station_pad_mask, 2)

        pga_pos_emb_output = self.model_Position(
            torch.DoubleTensor(data["target"].reshape(-1, 1, data["target"].shape[2]))
            .float()
            .cuda()
        )
        pga_pos_emb_output = pga_pos_emb_output.reshape(
            -1, self.pga_targets, self.emb_dim
        )
        target_pad_mask = torch.ones_like(data["target"], dtype=torch.bool)
        target_pad_mask = torch.all(target_pad_mask, 2)

        pad_mask = torch.cat((station_pad_mask, target_pad_mask), dim=1).cuda()

        add_pe_cnn_output = torch.add(cnn_output_reshape, emb_output)
        transformer_input = torch.cat((add_pe_cnn_output, pga_pos_emb_output), dim=1)
        transformer_output = self.model_Transformer(transformer_input, pad_mask)

        mlp_input = transformer_output[:, -self.pga_targets :, :].cuda()

        mlp_output = self.model_mlp(mlp_input)

        weight, sigma, mu = self.model_MDN(mlp_output)

        return weight, sigma, mu


def gaussian_distribution(y, mu, sigma):
    """Compute Gaussian densities for each MDN component."""
    one_div_sqrt_two_pi = 1.0 / np.sqrt(2.0 * np.pi)
    result = (y.expand_as(mu) - mu) * torch.reciprocal(sigma)
    result = -0.5 * (result * result)
    return (torch.exp(result) * torch.reciprocal(sigma)) * one_div_sqrt_two_pi


def mdn_loss_fn(pi, sigma, mu, y):
    """Compute negative log-likelihood loss for MDN outputs."""
    result = gaussian_distribution(y, mu, sigma) * pi
    result = torch.sum(result, dim=1)
    result = -torch.log(result)
    return torch.mean(result)
