
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from data_loader_scaled import load_data
from deepgravity_epidemic import DeepGravityEpidemic, build_feature_vector
import joblib

oa2features, oa2centroid, epidemic_by_week, o2d2flow = load_data()

X, Y = [], []
week = 1
for origin in o2d2flow[week]:
    for dest in o2d2flow[week][origin]:
        x_vec = build_feature_vector(origin, dest, oa2features, oa2centroid, epidemic_by_week, week)
        flujo = o2d2flow[week][origin][dest]
        epi = epidemic_by_week.loc[(dest, week)] if (dest, week) in epidemic_by_week.index else None
        contagios = epi["scaled_incidence"] if epi is not None else 0
        X.append(x_vec)
        Y.append([flujo, contagios])

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)

input_dim = X.shape[1]
model = DeepGravityEpidemic(input_dim)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

dataloader = DataLoader(TensorDataset(X, Y), batch_size=64, shuffle=True)

print("\n🚀 Entrenando modelo escalado...")
for epoch in range(100):
    total_loss = 0
    for xb, yb in dataloader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

torch.save(model.state_dict(), "modelo_epidemia_scaled.pt")
print("\n✅ Modelo guardado como modelo_epidemia_scaled.pt")
