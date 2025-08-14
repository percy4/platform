import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange

class MultiHeadAttention(nn.Module):
    def __init__(self, hid_channels: int, heads: int, dropout: float):
        super().__init__()
        self.hid_channels = hid_channels
        self.heads = heads
        self.keys = nn.Linear(hid_channels, hid_channels)
        self.queries = nn.Linear(hid_channels, hid_channels)
        self.values = nn.Linear(hid_channels, hid_channels)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(hid_channels, hid_channels)

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        queries = rearrange(self.queries(x),
                            "b n (h d) -> b h n d",
                            h=self.heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.hid_channels ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self,
                 hid_channels: int,
                 expansion: int = 4,
                 dropout: float = 0.):
        super().__init__(
            nn.Linear(hid_channels, expansion * hid_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * hid_channels, hid_channels),
        )
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, hid_channels: int, heads: int, dropout: float,
                 forward_expansion: int, forward_dropout: float):
        super().__init__(
            ResidualAdd(
                nn.Sequential(nn.LayerNorm(hid_channels),
                              MultiHeadAttention(hid_channels, heads, dropout),
                              nn.Dropout(dropout))),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(hid_channels),
                    FeedForwardBlock(hid_channels,
                                     expansion=forward_expansion,
                                     dropout=forward_dropout),
                    nn.Dropout(dropout))))


class TransformerEncoder(nn.Sequential):
    def __init__(self,
                 depth: int,
                 hid_channels: int,
                 heads: int = 10,
                 dropout: float = 0.5,
                 forward_expansion: int = 4,
                 forward_dropout: float = 0.5):
        super().__init__(*[
            TransformerEncoderBlock(hid_channels=hid_channels,
                                    heads=heads,
                                    dropout=dropout,
                                    forward_expansion=forward_expansion,
                                    forward_dropout=forward_dropout)
            for _ in range(depth)
        ])

class ClassificationHead(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 hid_channels: int = 32,
                 dropout: float = 0.5):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_channels, hid_channels * 8),
                                nn.ELU(), nn.Dropout(dropout),
                                nn.Linear(hid_channels * 8, hid_channels),
                                nn.ELU(), nn.Dropout(dropout),
                                nn.Linear(hid_channels, num_classes))

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc(x)
        return x

class multi_time_conv_base_encoder(nn.Module):
    def __init__(self, n_filters, n_chs, timeFilterLens, dp_rate=0.5, avgpool_kernel=50, d_expansion=4):
        super().__init__()
        self.n_chs = n_chs
        self.spatialConv = nn.Conv2d(1, n_filters, (n_chs, 1))
        self.timeConvs = nn.ModuleList([
            nn.Conv2d(1, n_filters, (1, t_len), padding='same') for t_len in timeFilterLens[:-1]
        ])
        self.timeconv1 = nn.Conv2d(1, n_filters, (1, timeFilterLens[-1]), padding='same')
        self.bn1 = nn.GroupNorm(4, n_filters)
        self.bn2 = nn.GroupNorm(4, n_filters)
        self.dropout = nn.Dropout(dp_rate)
        self.avgpool = nn.AvgPool2d((1, avgpool_kernel), padding=(0, avgpool_kernel // 2))
        self.projection = nn.Sequential(
            Rearrange('b h1 h2 e -> b 1 h1 h2 e'),
            nn.Conv3d(1, d_expansion, 1, 1),
            # transpose, conv could enhance fiting ability slightly
            Rearrange('b h0 h1 h2 e -> b e (h0 h1 h2)')
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, self.n_chs, x.shape[-1])
        x = self.spatialConv(x)
        x = self.bn1(x)
        x = x.permute(0, 2, 1, 3)
        out = self.timeconv1(x)
        for timeConv in self.timeConvs:
            out_time = timeConv(x)
            out_time = self.bn2(out_time)
            out = torch.cat([out, out_time], dim=-1)

        out = self.bn2(out)
        out = F.elu(out)
        out = self.dropout(out)

        out = self.avgpool(out)
        out = self.projection(out)
        out = F.elu(out)
        return out


class MSCT(nn.Module):
    def __init__(self,
                 encoder_layer,
                 hid_channels: int = 40,
                 depth: int = 6,
                 heads: int = 10,
                 dropout: float = 0.5,
                 forward_expansion: int = 4,
                 forward_dropout: float = 0.5,
                 freeze_encoder: bool = False):
        super().__init__()

        self.embd = encoder_layer
        self.transformer_encoder = TransformerEncoder(depth,
                                                      hid_channels,
                                                      heads=heads,
                                                      dropout=dropout,
                                                      forward_expansion=forward_expansion,
                                                      forward_dropout=forward_dropout)
        if freeze_encoder:
            for param in self.embd.parameters():
                param.requires_grad = False
            for param in self.transformer_encoder.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embd(x)
        x = self.transformer_encoder(x)
        return x

class MSCT_for_classify(nn.Module):
    def __init__(self, pretrain_model, num_classes=9, num_electrodes=32, frequency=250, eeg_duration=5,cls_dim=32,):
        super().__init__()
        self.pretrain_model = pretrain_model
        self.hidden_dim = num_electrodes*8*8*4
        self.cls = ClassificationHead(self.hidden_dim, num_classes,hid_channels=cls_dim)
    def forward(self, x):
        x = self.pretrain_model(x)
        x = x.flatten(start_dim=1)
        x = self.cls(x)
        return x


def get_data_loader(X,y, batch_size=32, shuffle=True, num_workers=0):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return data_loader



# model: MSCT_fit.entrypoint.MSCT
def main(model, data: np.ndarray):
    print(f"model type: {type(model)}")
    print(f"data type: {type(data)}")

    y = np.zeros((data.shape[0], 1))
    device = next(model.parameters()).device
    data = torch.tensor(data).to(device)
    test_loader = get_data_loader(data, y, batch_size=1, shuffle=False)
    # model.to(device)
    model.eval()
    with torch.no_grad():  # 在验证集上评估时，我们不需要计算梯度
        total_val_loss = 0
        prediction = []
        for x, y in test_loader:
            y_pred = model(x)
            # print(y_pred)
            prediction.append(y_pred)

        prediction = torch.cat(prediction, dim=0)
        avg_val_loss = total_val_loss / len(test_loader)
    result = prediction.cpu().numpy()
    result = np.argmax(result, axis=1)


    print(f"result type: {type(result)}")
    return [result]


if __name__ == "__main__":
    import pickle

    model = pickle.load(open("../MSCT_fit_model_0.pkl", "rb"))
    # data = pickle.load(open("../load_sample_data_0.pkl", "rb"))
    data = np.random.rand(32 * 5, 32, 1250)
    result = main(model, data)
    # print(result[0].shape)
