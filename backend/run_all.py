import torch
from data_loader_scaled import load_data
from deepgravity_epidemic import DeepGravityEpidemic, build_feature_vector
from what_if_epidemic_scaled import simulate_what_if
from visualize_results import plot_comparison

# -----------------------
# Cargar datos y modelo
# -----------------------
oa2features, oa2centroid, epidemic_by_week, o2d2flow = load_data()

# ⚠️ Fijar input_dim manualmente para que coincida con el modelo entrenado
input_dim = 16  # <-- usa el mismo valor que se usó en main_scaled.py al entrenar

model = DeepGravityEpidemic(input_dim)
model.load_state_dict(torch.load("modelo_epidemia_scaled.pt"))
model.eval()

# -----------------------
# Predicción normal
# -----------------------
origin = 1
destination = 6
week = 1

x = build_feature_vector(origin, destination, oa2features, oa2centroid, epidemic_by_week, week)
x_tensor = torch.tensor(x).unsqueeze(0)

with torch.no_grad():
    prediction = model(x_tensor).squeeze().numpy()

print("📍 PREDICCIÓN NORMAL")
print(f"Flujo predicho (escalado): {prediction[0]:.2f}")
print(f"Contagios estimados (escalado): {prediction[1]:.2f}")

# -----------------------
# Simulación WHAT-IF
# -----------------------
print("\n🔧 SIMULANDO CAMBIO EN POIs (duplicar Salud en destino)...")
result = simulate_what_if(
    origin=origin,
    destination=destination,
    week=week,
    oa2features=oa2features,
    oa2centroid=oa2centroid,
    epidemic_by_week=epidemic_by_week,
    model=model,
    poi_changes={"scaled_Salud_densidad": 0.5}
)

print("\n📊 RESULTADO WHAT-IF:")
for k, v in result.items():
    print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")

# -----------------------
# Visualización
# -----------------------
plot_comparison(result, label="Mitad Salud")
