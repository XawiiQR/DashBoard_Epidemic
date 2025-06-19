
import matplotlib.pyplot as plt

def plot_comparison(results, label="What-if"):
    labels = ["Flujo", "Contagios"]
    original = [results["original_flow_scaled"], results["original_cases_scaled"]]
    modified = [results["modified_flow_scaled"], results["modified_cases_scaled"]]

    x = range(len(labels))
    plt.figure(figsize=(8, 5))
    plt.bar(x, original, width=0.4, label="Original", align="center")
    plt.bar([i + 0.4 for i in x], modified, width=0.4, label=label, align="center")
    plt.xticks([i + 0.2 for i in x], labels)
    plt.ylabel("Valor escalado")
    plt.title("Comparación antes y después del cambio urbano")
    plt.legend()
    plt.tight_layout()
    plt.savefig("comparacion_what_if.png")
    plt.show()
