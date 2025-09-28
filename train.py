# > A slow and inefficient implementation of a slightly modified Trompt model
# > From the ICLM 2023 paper https://arxiv.org/abs/2305.18446

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.amp import autocast, GradScaler

import os
import urllib.request
from tqdm import tqdm


class TromptCell(nn.Module):
    def __init__(self, n_columns, n_prompts, d_model):
        super().__init__()
        # Embeddings (Figure 3.2)
        self.feature_emb_weight = nn.Parameter(torch.empty(n_columns, d_model))
        self.feature_emb_bias = nn.Parameter(torch.empty(1, n_columns, d_model))
        self.ln_emb = nn.LayerNorm(d_model)

        # Importance Getter (Figure 3.1)
        self.ln_col = nn.LayerNorm(d_model)
        self.ln_prompt = nn.LayerNorm(d_model)
        self.dense_imp = nn.Linear(2 * d_model, d_model)

        self.emb_column = nn.Parameter(torch.empty(n_columns, d_model))
        self.emb_prompt = nn.Parameter(torch.empty(n_prompts, d_model))

        # Modified expansion block (Figure 3.3)
        # Without non-linearities! This is important to make significant speed-ups possible.
        self.dense_expand = nn.Linear(1, n_prompts)

        self.reset_parameters()

    def reset_parameters(self):
        d_rsqrt = self.feature_emb_weight.shape[1] ** -0.5
        nn.init.uniform_(self.feature_emb_weight, -d_rsqrt, d_rsqrt)
        nn.init.uniform_(self.feature_emb_bias, -d_rsqrt, d_rsqrt)
        nn.init.normal_(self.emb_column, std=0.01)
        nn.init.normal_(self.emb_prompt, std=0.01)

    def forward(self, x: torch.Tensor, prev_cell_out: torch.Tensor) -> torch.Tensor:
        x_emb = torch.addcmul(
            self.feature_emb_bias, x.unsqueeze(-1), self.feature_emb_weight
        )
        x_emb = self.ln_emb(F.relu(x_emb))
        x_prompt = self.emb_prompt
        x_prompt = (
            self.dense_imp(torch.cat([self.ln_prompt(x_prompt), prev_cell_out], dim=-1))
            + x_prompt
        )
        x_column = self.ln_col(self.emb_column)
        mask = torch.softmax(torch.matmul(x_prompt, x_column.transpose(0, 1)), dim=-1)
        x_out = torch.einsum("pc,bcd->bpd", mask, x_emb)
        x_out = x_out + torch.einsum(
            "pc,bcd,pa->bpd", mask, x_emb, self.dense_expand.weight
        )
        x_out = x_out + torch.einsum("pc,p->p", mask, self.dense_expand.bias).unsqueeze(
            -1
        ).unsqueeze(0)
        return x_out


class TromptDownstream(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.dense0 = nn.Linear(d_model, 1)
        self.dense1 = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.dense_out = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pw = torch.softmax(self.dense0(x).squeeze(-1), dim=-1)
        xnew = torch.einsum("bcp,bcpd->bcd", pw, x)
        return self.dense_out(self.ln(F.relu(self.dense1(xnew)))).squeeze(-1)


class Trompt(nn.Module):
    def __init__(self, n_columns, n_prompts, d_model, n_cycles):
        super().__init__()
        self.tcells = nn.ModuleList(
            [TromptCell(n_columns, n_prompts, d_model) for _ in range(n_cycles)]
        )
        self.tdown = TromptDownstream(d_model)
        self.prompt = nn.Parameter(torch.empty(n_prompts, d_model))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.prompt, std=0.01)

    def forward(self, x):
        x_prompt = self.prompt
        outputs = []
        for cell in self.tcells:
            outputs.append(cell(x, x_prompt))
        outputs = self.tdown(torch.stack(outputs, dim=1))
        return outputs


def load_from_url(url, cache_dir="."):
    filename = os.path.join(cache_dir, url.split("/")[-1])
    if not os.path.exists(filename):
        with tqdm(unit="B", unit_scale=True, desc=filename) as pbar:
            urllib.request.urlretrieve(
                url, filename, reporthook=lambda _, b, t: pbar.update(b)
            )
    return torch.load(filename, map_location=torch.device("cpu"), weights_only=True)


TRAIN_DATA = (
    "https://huggingface.co/datasets/puhsu/hw01-data/resolve/main/train_dataset.pt"
)
VAL_DATA = "https://huggingface.co/datasets/puhsu/hw01-data/resolve/main/val_dataset.pt"

if __name__ == "__main__":
    torch.manual_seed(0)
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True
    # torch.backends.cudnn.benchmark = True

    train_dataset = torch.utils.data.TensorDataset(
        *map(torch.nan_to_num, load_from_url(TRAIN_DATA))
    )
    val_dataset = torch.utils.data.TensorDataset(
        *map(torch.nan_to_num, load_from_url(VAL_DATA))
    )

    Y_mean = train_dataset.tensors[1].mean()
    Y_std = train_dataset.tensors[1].std()
    train_dataset.tensors = (
        train_dataset.tensors[0],
        (train_dataset.tensors[1] - Y_mean) / Y_std,
    )

    model = Trompt(
        n_columns=train_dataset.tensors[0].shape[1],
        n_prompts=128,
        d_model=128,
        n_cycles=6,
    )
    model = nn.DataParallel(model, device_ids=[0, 1])
    device = torch.device("cuda")
    model.to(device)

    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=2,
        batch_size=512,  # Reduced for stability
        shuffle=True,
        pin_memory=True,
        persistent_workers=False,  # Simplified
    )
    val_dl = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=2,
        batch_size=1024,
        pin_memory=True,
        persistent_workers=False,  # Simplified
    )

    # Fixed: use standard AdamW without fused (more compatible)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scaler = GradScaler()

    EPOCHS = 5

    for e in range(1, EPOCHS + 1):
        model.train()
        for batch in tqdm(train_dl):
            x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(device_type="cuda", dtype=torch.float16):
                pred = model(x)
                # Fixed: ensure correct dimensions for loss computation
                loss = F.mse_loss(pred, y.unsqueeze(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        mae = 0
        with torch.inference_mode():
            for batch in val_dl:
                x, y = batch
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with autocast(device_type="cuda", dtype=torch.float16):
                    pred = model(x)

                mae += (pred.mean(dim=-1) * Y_std + Y_mean - y).abs().sum().item()

            mae = mae / len(val_dataset)

            print(f">>> Epoch {e:>02}")
            print(f"Validation MAE = {mae:.5f}")
            print(">>>\n")
