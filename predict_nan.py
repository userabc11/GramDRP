import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from data import loadExitAndNanDrugCellData
from model import SADRP
from parameter import parse_args

def train_and_predict_nan_response(model_path, output_dir, device='cuda:3', seed=16, epochs=150):
    torch.manual_seed(seed)
    
    args = parse_args()
    args.epochs = epochs
    
    train_loader, val_loader, test_loader, _, num_node_features, num_edge_features, _ = loadExitAndNanDrugCellData(args, "save", "single")
    
    model = SADRP(args, num_node_features, num_edge_features)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # train
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            data = data.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                val_loss += criterion(model(data), data.y).item()
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}: train_loss={total_loss/len(train_loader):.4f}, val_loss={val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_path)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    drug_results = defaultdict(list)
    
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Predicting"):
            data = data.to(device)
            outputs = model(data)
            preds = outputs.cpu().numpy().flatten()
            
            for d_id, c_id, p in zip(data.drugId, data.cellId, preds):
                all_preds.append({"drugId": d_id, "cellId": c_id, "ic50": float(p)})
                drug_results[d_id].append(float(p))
    
    os.makedirs(output_dir, exist_ok=True)
    
    df_all = pd.DataFrame(all_preds)
    df_all.sort_values(by="ic50", ascending=True).to_csv(f"{output_dir}/predictions.csv", index=False)
    
    summary = [{"drugId": d, "avg_ic50": np.mean(v), "num": len(v)} for d, v in drug_results.items()]
    pd.DataFrame(summary).sort_values(by="avg_ic50").to_csv(f"{output_dir}/drug_summary.csv", index=False)


if __name__ == '__main__':
    train_and_predict_nan_response(
        model_path="./outputs/nan/model.pth",
        output_dir="./outputs/nan",
        device='cuda:3',
        seed=42,
        epochs=150
    )