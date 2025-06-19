from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from deepgravity_epidemic import DeepGravityEpidemic, build_feature_vector
from data_loader_scaled import load_data
import os

app = Flask(__name__)
CORS(app)  # 🔓 Habilita CORS

oa2features, oa2centroid, epidemic_by_week, o2d2flow = load_data()
input_dim = 16
model = DeepGravityEpidemic(input_dim)
model.load_state_dict(torch.load("modelo_epidemia_scaled.pt"))
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    origin = data["origin"]
    destination = data["destination"]
    week = data["week"]
    x = build_feature_vector(origin, destination, oa2features, oa2centroid, epidemic_by_week, week)
    with torch.no_grad():
        x_tensor = torch.tensor(x).unsqueeze(0)
        pred = model(x_tensor).squeeze().numpy()
    return jsonify({"flujo": float(pred[0]), "contagios": float(pred[1])})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 1000))
    app.run(host="0.0.0.0", port=port)
