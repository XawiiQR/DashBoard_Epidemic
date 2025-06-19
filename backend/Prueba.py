from data_loader_scaled import load_data
from deepgravity_epidemic import DeepGravityEpidemic, build_feature_vector
import torch

# Cargar datos
oa2features, oa2centroid, epidemic_by_week, o2d2flow = load_data()

# Cargar el modelo entrenado
input_dim = len(list(oa2features.values())[0]) + 4
model = DeepGravityEpidemic(input_dim)
model.load_state_dict(torch.load("modelo_epidemia_scaled.pt"))
model.eval()

# Elegimos un par de zonas: origen y destino (usa los FIPS de tus datos)
origin = 1
destination = 6
week = 1

# Crear vector de entrada
x = build_feature_vector(origin, destination, oa2features, oa2centroid, epidemic_by_week, week)
x_tensor = torch.tensor(x).unsqueeze(0)

# Hacer predicción
with torch.no_grad():
    prediction = model(x_tensor).squeeze().numpy()

print("📍 PREDICCIÓN NORMAL:")
print(f"Flujo predicho (escalado): {prediction[0]:.2f}")
print(f"Contagios estimados (escalado): {prediction[1]:.2f}")
