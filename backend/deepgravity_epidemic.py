
import torch
import torch.nn as nn

class DeepGravityEpidemic(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # [flujo_predicho, contagios_estimados]
        )

    def forward(self, x):
        return self.net(x)  # Output: [batch, 2] => flujo, nuevos_casos


def build_feature_vector(oa_origin, oa_dest, oa2features, oa2centroid, epidemic_by_week, week):
    # Features del origen y destino
    f_o = list(oa2features.get(oa_origin, {}).values())
    f_d = list(oa2features.get(oa_dest, {}).values())

    # Distancia geodésica aproximada
    lat1, lon1 = oa2centroid.get(oa_origin, (0, 0))
    lat2, lon2 = oa2centroid.get(oa_dest, (0, 0))
    dist = ((lat1 - lat2)**2 + (lon1 - lon2)**2) ** 0.5

    # Datos epidemiológicos destino
    try:
        epi_d = epidemic_by_week.loc[(oa_dest, week)]
        new_cases = epi_d["New_Cases"]
        incident_rate = epi_d["Incident_Rate"]
        risk = 1 if epi_d["Risk_Category"] == "High" else 0
    except:
        new_cases = 0
        incident_rate = 0
        risk = 0

    return f_o + f_d + [dist, new_cases, incident_rate, risk]
