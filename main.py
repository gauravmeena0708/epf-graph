import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv, to_hetero, GATConv # Using SAGEConv for this example
from torch_geometric.utils import negative_sampling, to_undirected
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import random
from collections import defaultdict
import networkx as nx # For initial graph construction if needed, though PyG is primary

# --- 0. Configuration ---
NUM_MONTHS_DATA = 24 # Total months of data to simulate
NUM_ESTABLISHMENTS = 20
PREDICT_WINDOW = 1 # Predict transfers for the next 1 month
TRAIN_HISTORY_WINDOW = 3 # Use last 3 months of data to predict next month
HIDDEN_CHANNELS = 64
NUM_EPOCHS = 50
LEARNING_RATE = 0.01

# --- 1. Enhanced Data Generation with Temporal Aspect ---
def generate_temporal_epfo_data(num_establishments, num_months):
    establishments = [f"EST{str(i).zfill(3)}" for i in range(1, num_establishments + 1)]
    industries = ["IT", "Manufacturing", "Consulting", "Healthcare", "Finance"]
    cities = ["Bangalore", "Pune", "Hyderabad", "Chennai", "Mumbai"]
    size_categories = ["Small", "Medium", "Large"]

    establishment_details = {
        est_id: {
            "name": f"Company {est_id.replace('EST', '')}",
            "industry": random.choice(industries),
            "city": random.choice(cities),
            "size_category": random.choice(size_categories)
        } for est_id in establishments
    }

    # Encode categorical features (once for all establishments)
    all_industries = sorted(list(set(d["industry"] for d in establishment_details.values())))
    all_cities = sorted(list(set(d["city"] for d in establishment_details.values())))
    all_sizes = sorted(list(set(d["size_category"] for d in establishment_details.values())))

    industry_encoder = LabelEncoder().fit(all_industries)
    city_encoder = LabelEncoder().fit(all_cities)
    size_encoder = LabelEncoder().fit(all_sizes)

    # One-hot encode
    # Note: For simplicity, keeping one-hot encoders separate. In a large system, you might combine.
    # We'll create one-hot vectors later when constructing PyG data objects.

    establishment_features = {}
    for est_id, details in establishment_details.items():
        # These will be converted to one-hot later
        feature_vector = [
            details["industry"],
            details["city"],
            details["size_category"]
        ]
        establishment_features[est_id] = {
            "raw_features": feature_vector,
            "encoders": {"industry": industry_encoder, "city": city_encoder, "size": size_encoder},
            "original_details": details
        }


    temporal_transfers = [] # List of (source_id, target_id, members, month_idx)
    # Simulate some base level of transfers
    for month in range(num_months):
        num_transfers_this_month = random.randint(5, num_establishments * 2)
        for _ in range(num_transfers_this_month):
            source_est = random.choice(establishments)
            target_est = random.choice(establishments)
            if source_est == target_est:
                continue
            members = random.randint(1, 10)
            temporal_transfers.append({
                "source": source_est,
                "target": target_est,
                "members": members,
                "month": month
            })
        # Simulate some "hot" companies attracting more people for a few months
        if month % 6 < 2 and month > 3 : # For 2 months every 6 months
            hot_target = random.choice(establishments)
            for _ in range(random.randint(3,7)):
                source_est = random.choice(establishments)
                if source_est == hot_target: continue
                temporal_transfers.append({
                    "source": source_est,
                    "target": hot_target,
                    "members": random.randint(2,8),
                    "month": month
                })


    return establishment_features, temporal_transfers, (industry_encoder, city_encoder, size_encoder)

# --- 2. Prepare PyG Data Snapshots ---
def create_pyg_snapshot(establishment_features, transfers_in_month, est_id_to_idx_map, encoders):
    industry_encoder, city_encoder, size_encoder = encoders

    num_nodes = len(establishment_features)
    node_feats_list = [None] * num_nodes

    # Create feature matrix
    for est_id, idx in est_id_to_idx_map.items():
        raw_feats = establishment_features[est_id]["raw_features"]
        # One-hot encode features
        industry_onehot = np.zeros(len(industry_encoder.classes_))
        industry_onehot[industry_encoder.transform([raw_feats[0]])[0]] = 1

        city_onehot = np.zeros(len(city_encoder.classes_))
        city_onehot[city_encoder.transform([raw_feats[1]])[0]] = 1

        size_onehot = np.zeros(len(size_encoder.classes_))
        size_onehot[size_encoder.transform([raw_feats[2]])[0]] = 1

        node_feats_list[idx] = np.concatenate([industry_onehot, city_onehot, size_onehot])

    x = torch.tensor(np.array(node_feats_list), dtype=torch.float)

    source_nodes = [est_id_to_idx_map[t['source']] for t in transfers_in_month]
    target_nodes = [est_id_to_idx_map[t['target']] for t in transfers_in_month]
    edge_attr_members = [[t['members']] for t in transfers_in_month] # Edge weight

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_members, dtype=torch.float)

    # For link prediction, graphs are typically undirected during embedding learning
    # Or one can use specific directed GNNs. Here we'll make it undirected for SAGE.
    # edge_index_undirected = to_undirected(edge_index)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
    return data

# --- 3. GNN Model for Link Prediction (GraphSAGE based) ---
class GNNLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        # No explicit decoder here; we'll use dot product of embeddings for simplicity

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        # edge_label_index contains pairs of nodes for which we want to predict a link
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1) # Dot product score

    def decode_all(self, z):
        prob_adj = z @ z.t() # Score for all possible pairs
        return prob_adj

# --- 4. Training and Evaluation Logic ---
def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()

    z = model.encode(data.x, data.edge_index)

    # Positive edges: existing edges in the current graph
    pos_edge_label_index = data.edge_index
    pos_pred = model.decode(z, pos_edge_label_index)
    pos_edge_label = torch.ones(pos_edge_label_index.size(1), device=z.device)


    # Negative edges: sample non-existing edges
    # Ensure num_nodes is correctly passed or inferred for negative_sampling
    neg_edge_label_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=pos_edge_label_index.size(1) # Match number of positive samples
    )
    neg_pred = model.decode(z, neg_edge_label_index)
    neg_edge_label = torch.zeros(neg_edge_label_index.size(1), device=z.device)

    # Concatenate positive and negative predictions and labels
    edge_label_index = torch.cat([pos_edge_label_index, neg_edge_label_index], dim=-1)
    edge_pred = torch.cat([pos_pred, neg_pred])
    edge_label = torch.cat([pos_edge_label, neg_edge_label])


    loss = criterion(edge_pred, edge_label)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, data_train, data_test): # data_train for embeddings, data_test for links to predict
    model.eval()
    z = model.encode(data_train.x, data_train.edge_index) # Embeddings from training portion

    # Predict links that are in data_test
    pos_edge_label_index = data_test.edge_index
    pos_pred = model.decode(z, pos_edge_label_index)
    pos_edge_label = torch.ones(pos_edge_label_index.size(1))


    # Sample negative edges not in data_train and not in data_test's positive edges
    # This negative sampling for testing should be carefully considered.
    # For simplicity, sample randomly avoiding existing train+test edges.
    # A more robust approach might be to exclude all edges from data_train union data_test.
    neg_edge_label_index = negative_sampling(
        edge_index=torch.cat([data_train.edge_index, data_test.edge_index], dim=-1),
        num_nodes=data_train.num_nodes,
        num_neg_samples=pos_edge_label_index.size(1)
    )
    neg_pred = model.decode(z, neg_edge_label_index)
    neg_edge_label = torch.zeros(neg_edge_label_index.size(1))

    edge_pred = torch.cat([pos_pred, neg_pred])
    edge_label = torch.cat([pos_edge_label, neg_edge_label])

    # Calculate AUC (Area Under ROC Curve)
    from sklearn.metrics import roc_auc_score
    # Apply sigmoid to get probabilities if your criterion doesn't include it
    # (BCEWithLogitsLoss includes sigmoid)
    # For dot product, scores can be used directly or passed through sigmoid
    scores = torch.sigmoid(edge_pred).cpu().numpy()
    labels = edge_label.cpu().numpy()

    if len(np.unique(labels)) < 2: # Handles cases with only one class in a small test batch
        return 0.5 # Or handle as appropriate
    return roc_auc_score(labels, scores)


# --- 5. Main Script Execution ---
if __name__ == '__main__':
    print("Starting EPFO Temporal Graph Analysis with GNNs...")

    # 5.1 Generate Data
    establishment_node_features, all_temporal_transfers, encoders = generate_temporal_epfo_data(NUM_ESTABLISHMENTS, NUM_MONTHS_DATA)
    all_establishment_ids = list(establishment_node_features.keys())
    est_id_to_idx_map = {name: i for i, name in enumerate(all_establishment_ids)}
    idx_to_est_id_map = {i: name for i, name in enumerate(all_establishment_ids)}

    print(f"Generated data for {NUM_ESTABLISHMENTS} establishments over {NUM_MONTHS_DATA} months.")
    print(f"Total transfers recorded: {len(all_temporal_transfers)}")
    # Example: print(establishment_node_features['EST001'])

    # 5.2 Create Monthly Snapshots
    monthly_pyg_graphs = []
    for month_idx in range(NUM_MONTHS_DATA):
        transfers_this_month = [t for t in all_temporal_transfers if t['month'] == month_idx]
        if not transfers_this_month: # Handle months with no transfers
            # Create a graph with nodes but no edges
             # Create feature matrix (same as in create_pyg_snapshot)
            num_nodes = len(establishment_node_features)
            node_feats_list = [None] * num_nodes
            industry_encoder, city_encoder, size_encoder = encoders
            for est_id, idx in est_id_to_idx_map.items():
                raw_feats = establishment_node_features[est_id]["raw_features"]
                industry_onehot = np.zeros(len(industry_encoder.classes_))
                industry_onehot[industry_encoder.transform([raw_feats[0]])[0]] = 1
                city_onehot = np.zeros(len(city_encoder.classes_))
                city_onehot[city_encoder.transform([raw_feats[1]])[0]] = 1
                size_onehot = np.zeros(len(size_encoder.classes_))
                size_onehot[size_encoder.transform([raw_feats[2]])[0]] = 1
                node_feats_list[idx] = np.concatenate([industry_onehot, city_onehot, size_onehot])
            x = torch.tensor(np.array(node_feats_list), dtype=torch.float)
            snapshot = Data(x=x, edge_index=torch.empty((2,0), dtype=torch.long), num_nodes=num_nodes)
        else:
            snapshot = create_pyg_snapshot(establishment_node_features, transfers_this_month, est_id_to_idx_map, encoders)
        monthly_pyg_graphs.append(snapshot)
    print(f"Created {len(monthly_pyg_graphs)} monthly graph snapshots.")

    # 5.3 Initialize Model, Optimizer, Criterion
    # Determine input_channels from the feature vector length
    sample_x_shape = monthly_pyg_graphs[0].x.shape[1]
    model = GNNLinkPredictor(sample_x_shape, HIDDEN_CHANNELS, HIDDEN_CHANNELS // 2) # out_channels for embedding
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss() # Suitable for link prediction scores

    print(f"Model initialized: {model}")

    # 5.4 Temporal Training Loop
    # We train by predicting month M+1 using data up to month M (or a window of months)
    # For simplicity, we'll use a sliding window approach:
    # Train on [M, M+1, ..., M+W-1] to predict links in M+W

    for current_predict_month in range(TRAIN_HISTORY_WINDOW, NUM_MONTHS_DATA - PREDICT_WINDOW +1):
        # Aggregate history graphs for training embeddings
        # Option 1: Use only the most recent graph from history
        # train_graph_data = monthly_pyg_graphs[current_predict_month - 1]

        # Option 2: Aggregate (merge) graphs in the history window
        # This requires careful handling of node features if they change,
        # and merging edge_indices. For static node features, it's simpler.
        history_start_month = current_predict_month - TRAIN_HISTORY_WINDOW
        history_graphs = monthly_pyg_graphs[history_start_month : current_predict_month]

        # Simple aggregation: combine all edges from history, use current node features
        # (Assuming node features are static or represent the latest known state)
        if not history_graphs: continue

        combined_edge_index = torch.cat([g.edge_index for g in history_graphs if g.edge_index.numel() > 0], dim=1)
        # Remove duplicate edges that might result from combining
        combined_edge_index = torch.unique(combined_edge_index, dim=1)


        # Use node features from the latest graph in the history window
        current_node_features = history_graphs[-1].x
        num_nodes_current = history_graphs[-1].num_nodes

        if combined_edge_index.numel() == 0: # If no edges in history
            print(f"Skipping month {current_predict_month}: No edges in training history window.")
            train_graph_data = Data(x=current_node_features, edge_index=torch.empty((2,0), dtype=torch.long), num_nodes=num_nodes_current)
        else:
            train_graph_data = Data(x=current_node_features, edge_index=combined_edge_index, num_nodes=num_nodes_current)


        # Target graph for prediction (ground truth)
        target_graph_data = monthly_pyg_graphs[current_predict_month] # This is the graph for the month we want to predict links for

        if target_graph_data.edge_index.numel() == 0: # No actual transfers in the target month
            print(f"Skipping month {current_predict_month}: No actual transfers in target month to evaluate against.")
            continue

        print(f"\n--- Training to predict Month {current_predict_month} ---")
        print(f"Using history from months {history_start_month} to {current_predict_month-1}")
        print(f"Training graph: {train_graph_data.num_nodes} nodes, {train_graph_data.num_edges} edges")
        print(f"Target (test) graph: {target_graph_data.num_nodes} nodes, {target_graph_data.num_edges} edges")


        for epoch in range(NUM_EPOCHS):
            loss = train(model, train_graph_data, optimizer, criterion)
            if epoch % 10 == 0:
                # The 'test' function here uses target_graph_data's edges as positive examples
                # And data_train for generating embeddings.
                auc = test(model, train_graph_data, target_graph_data)
                print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Test AUC: {auc:.4f} (predicting Month {current_predict_month})")

    print("\n--- Finished Temporal GNN Training and Evaluation ---")

    # Example: How to get predictions for a future, unseen period
    # Let's say we want to predict for NUM_MONTHS_DATA + 1
    # We would use the latest available data to generate embeddings
    if NUM_MONTHS_DATA > 0:
        latest_history_start = max(0, NUM_MONTHS_DATA - TRAIN_HISTORY_WINDOW)
        latest_history_graphs = monthly_pyg_graphs[latest_history_start : NUM_MONTHS_DATA]

        if latest_history_graphs:
            latest_combined_edge_index = torch.cat([g.edge_index for g in latest_history_graphs if g.edge_index.numel() > 0], dim=1)
            latest_combined_edge_index = torch.unique(latest_combined_edge_index, dim=1)
            latest_node_features = latest_history_graphs[-1].x
            num_nodes_latest = latest_history_graphs[-1].num_nodes

            if latest_combined_edge_index.numel() > 0:
                inference_graph_data = Data(x=latest_node_features, edge_index=latest_combined_edge_index, num_nodes=num_nodes_latest)

                model.eval()
                with torch.no_grad():
                    z_final = model.encode(inference_graph_data.x, inference_graph_data.edge_index)
                    # Now z_final contains embeddings for all establishments
                    # To predict links, you can compute scores for all pairs or specific pairs
                    all_pair_scores = model.decode_all(z_final) # This is a (num_nodes x num_nodes) matrix
                    print("\n--- Example: Predicting links for the next period ---")
                    print(f"Embeddings shape: {z_final.shape}")
                    print(f"All-pair scores shape: {all_pair_scores.shape}")

                    # Get top K predictions (excluding self-loops and existing links in inference_graph_data)
                    num_top_k = 10
                    adj = torch.sigmoid(all_pair_scores) # Probabilities
                    adj = adj.triu(diagonal=1) # Consider only upper triangle, no self-loops

                    # Mask out existing edges from the last training period
                    # This is important to predict *new* links
                    existing_mask = torch.zeros_like(adj, dtype=torch.bool)
                    if inference_graph_data.edge_index.numel() > 0:
                       existing_mask[inference_graph_data.edge_index[0], inference_graph_data.edge_index[1]] = True
                       existing_mask[inference_graph_data.edge_index[1], inference_graph_data.edge_index[0]] = True # if undirected in training
                    adj[existing_mask] = -1 # Effectively ignore them

                    flat_scores = adj.flatten()
                    top_k_indices = torch.topk(flat_scores, num_top_k).indices
                    top_k_src = top_k_indices // num_nodes_latest
                    top_k_dst = top_k_indices % num_nodes_latest

                    print(f"\nTop {num_top_k} potential new transfers predicted for the next month:")
                    for i in range(num_top_k):
                        src_node = idx_to_est_id_map[top_k_src[i].item()]
                        dst_node = idx_to_est_id_map[top_k_dst[i].item()]
                        score = adj[top_k_src[i], top_k_dst[i]].item()
                        if score > 0 : # Check if it's not a masked out value
                             print(f"- From {src_node} to {dst_node} (Score: {score:.4f})")

            else:
                print("Not enough data in the latest history to make future predictions.")
        else:
            print("Not enough historical data to make future predictions.")
    else:
        print("No data generated, cannot make future predictions.")