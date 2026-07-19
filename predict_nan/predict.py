from pathlib import Path
from types import SimpleNamespace
import sys
import torch
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data import loadExitAndNanDrugCellData
from model import SADRP

device="cuda:1"
BASE_DIR = Path(__file__).resolve().parent
CONFIG = SimpleNamespace(
    seed=16, train_batch_size=128, test_batch_size=128, num_layers=1, node_dim=128, edge_dim=64, num_heads=4, output_dim=1, max_in_degree=4, max_out_degree=4, max_path_distance=3
)

def save_drug_summary(drug_results):
    summary = [{"drugId": int(d), "avg_ic50": np.mean(v), "unknown_num": len(v)} for d, v in drug_results.items()]
    summary_df = pd.DataFrame(summary).sort_values(by="avg_ic50").reset_index(drop=True)
    mapping_path = BASE_DIR / "mapping.csv"
    if not mapping_path.exists():
        raise FileNotFoundError(f"mapping.csv not found: {mapping_path}")
    mapping_df = pd.read_csv(mapping_path).rename(columns={"DrugId": "drugId"})
    summary_df = summary_df.merge(mapping_df, on="drugId", how="left")
    summary_df = summary_df[["drugId", "Drug Name", "avg_ic50", "Unknown num"]].rename(columns={"drugId": "DrugId"})
    summary_path = BASE_DIR / "drug_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    missing_df = summary_df[summary_df["Unknown num"] > 100]
    if not missing_df.empty:
        print("Drugs with Unknown num > 100:")
        print(missing_df.to_string(index=False))
    return summary_path

def predict_nan_response():
    torch.manual_seed(CONFIG.seed)
    np.random.seed(CONFIG.seed)
    args = CONFIG
    # dataset loading
    _, _, loader, _, num_node_features, num_edge_features, _ = loadExitAndNanDrugCellData(args, "save", "single")

    # model
    model = SADRP(args, num_node_features, num_edge_features).to(device)
    model.load_state_dict(torch.load(BASE_DIR / "model.pth", map_location=device))
    model.eval()
    drug_results = {}

    # prediction
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            preds = model(data).cpu().numpy().flatten()
            for d_id, p in zip(data.drugId, preds):
                drug_results.setdefault(int(d_id), []).append(float(p))
    save_drug_summary(drug_results)


if __name__ == '__main__':
    predict_nan_response()