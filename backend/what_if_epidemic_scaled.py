
from deepgravity_epidemic import build_feature_vector
import torch

def simulate_what_if(origin, destination, week, oa2features, oa2centroid, epidemic_by_week, model, poi_changes=None):
    oa2features_mod = {k: v.copy() for k, v in oa2features.items()}
    if poi_changes and destination in oa2features_mod:
        for poi_type, factor in poi_changes.items():
            if poi_type in oa2features_mod[destination]:
                oa2features_mod[destination][poi_type] *= factor

    x_original = build_feature_vector(origin, destination, oa2features, oa2centroid, epidemic_by_week, week)
    x_modified = build_feature_vector(origin, destination, oa2features_mod, oa2centroid, epidemic_by_week, week)

    model.eval()
    with torch.no_grad():
        y_orig = model(torch.tensor(x_original).unsqueeze(0)).squeeze().numpy()
        y_mod = model(torch.tensor(x_modified).unsqueeze(0)).squeeze().numpy()

    return {
        "original_flow_scaled": float(y_orig[0]),
        "original_cases_scaled": float(y_orig[1]),
        "modified_flow_scaled": float(y_mod[0]),
        "modified_cases_scaled": float(y_mod[1]),
        "delta_flow": float(y_mod[0] - y_orig[0]),
        "delta_cases": float(y_mod[1] - y_orig[1])
    }
