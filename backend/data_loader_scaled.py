
import pandas as pd

def load_data():
    mobility = pd.read_csv("mobility_scaled.csv")
    epidemics = pd.read_csv("epidemics_scaled.csv")
    population = pd.read_csv("population_clean.csv")
    pois = pd.read_csv("pois_scaled.csv")

    population = population.rename(columns={"FIPS": "cbg"})
    pois = pois.rename(columns={"FIPS": "cbg"})
    epidemics = epidemics.rename(columns={"FIPS": "cbg"})

    mobility = mobility.rename(columns={"FIPS_O": "origin", "FIPS_D": "destination"})

    oa2centroid = dict(zip(population["cbg"], zip(population["Lat"], population["Long"])))

    features_cols = [col for col in pois.columns if col.startswith("scaled_")]
    oa2features = pois.set_index("cbg")[features_cols].to_dict(orient="index")

    epidemic_by_week = epidemics.set_index(["cbg", "Week"])[["scaled_new_cases", "scaled_incidence", "Risk_Category"]]

    o2d2flow = {}
    for _, row in mobility.iterrows():
        w = int(row["Week"])
        o, d = row["origin"], row["destination"]
        o2d2flow.setdefault(w, {}).setdefault(o, {})[d] = row["scaled_flow"]

    return oa2features, oa2centroid, epidemic_by_week, o2d2flow
