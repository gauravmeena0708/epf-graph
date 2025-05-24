"""
Command-line interface for running advanced GNN applications.

This script allows users to perform various Graph Neural Network (GNN) tasks
on establishment data, including:
1.  Node Classification: Predict node attributes (e.g., industry, city) using GNNs.
2.  Node Clustering: Group similar nodes based on GNN-generated embeddings.
3.  Anomaly Detection: Identify unusual nodes using GNN-based autoencoders.

Users can specify the GNN model (GCN, GraphSAGE, GAT), the application to run,
data file paths, and various hyperparameters for training and model configuration.
"""
import argparse
import torch
import numpy as np
import pandas as pd

# Import GNN application components
from gnn_applications import (
    load_establishment_data,
    preprocess_node_features,
    create_pyg_data_object,
    GCNEncoder,
    SAGEEncoder,
    GATEncoder,
    NodeClassifier,
    train_node_classifier,
    evaluate_node_classifier,
    get_node_embeddings,
    cluster_nodes_kmeans,
    GNNAutoencoder,
    train_gnn_autoencoder,
    get_reconstruction_errors,
    get_anomalies_by_error
)
from feature_engineering import calculate_establishment_flow_features # Import the new function
# from data import save_data_to_csv # For potentially saving results - uncomment if needed

# Default parameters (can be overridden by argparse)
DEFAULT_NODES_FILE = "epfo_establishments_nodes.csv"
DEFAULT_EDGES_FILE = "epfo_transfers_edges.csv"
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_EPOCHS = 100 # Reduced for quick testing; can be increased
DEFAULT_HIDDEN_CHANNELS = 64
DEFAULT_EMBEDDING_SIZE = 32 # Output size of GNN encoders

def main():
    """
    Main function to parse arguments, load data, initialize models,
    and run the selected GNN application.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run GNN Applications on EPFO Data")

    # General arguments for data, model, and training setup
    parser.add_argument('--nodes_file', type=str, default=DEFAULT_NODES_FILE, help="Path to nodes CSV file.")
    parser.add_argument('--edges_file', type=str, default=DEFAULT_EDGES_FILE, help="Path to edges CSV file.")
    parser.add_argument('--gnn_model', type=str, choices=['gcn', 'sage', 'gat'], required=True, help="GNN model architecture to use (GCN, GraphSAGE, GAT).")
    parser.add_argument('--application', type=str, choices=['classification', 'clustering', 'anomaly_detection'], required=True, help="GNN application to run.")
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate for the optimizer.")
    parser.add_argument('--hidden_channels', type=int, default=DEFAULT_HIDDEN_CHANNELS, help="Number of hidden units in GNN layers.")
    parser.add_argument('--embedding_size', type=int, default=DEFAULT_EMBEDDING_SIZE, help="Dimension of node embeddings output by the GNN encoder.")
    parser.add_argument('--num_gnn_layers', type=int, default=2, help="Number of GNN layers in the encoder.")
    parser.add_argument('--dropout_rate', type=float, default=0.5, help="Dropout rate for GNN layers.")
    parser.add_argument('--gat_heads', type=int, default=8, help="Number of attention heads if GAT model is selected.")

    # Arguments for feature engineering (related to 'is_high_future_demand' label)
    parser.add_argument('--num_recent_months', type=int, default=6, help='Number of recent months for flow feature calculation.')
    parser.add_argument('--high_demand_quantile', type=float, default=0.8, help='Quantile for net inflow to define high-demand label.')

    # Arguments specific to Node Classification
    # Added 'is_high_future_demand' to choices
    parser.add_argument('--target_label', type=str, default='industry', 
                        choices=['industry', 'city', 'size_category', 'is_high_future_demand'], 
                        help="Target node attribute for classification.")

    # Arguments specific to Node Clustering
    parser.add_argument('--n_clusters', type=int, default=5, help="Number of clusters for K-Means algorithm.")

    # Arguments specific to Anomaly Detection (using GNN Autoencoder)
    parser.add_argument('--decoder_hidden_dims', nargs='+', type=int, default=[DEFAULT_HIDDEN_CHANNELS], 
                        help=f"List of hidden layer dimensions for the autoencoder's decoder. E.g., --decoder_hidden_dims 64 32. Default: [{DEFAULT_HIDDEN_CHANNELS}]")
    parser.add_argument('--anomaly_top_n', type=int, default=10, help="Number of top anomalous nodes to report.")

    args = parser.parse_args()

    # --- Data Loading and Preprocessing ---
    print("Loading and preprocessing data...")
    # Load graph data from CSVs into NetworkX graph and pandas DataFrames
    nx_graph, nodes_df, edges_df = load_establishment_data(args.nodes_file, args.edges_file) # Capture edges_df

    # --- Data Augmentation (Feature Engineering) ---
    # Store a copy of the original nodes_df before augmentation if needed for other tasks
    # or if calculate_establishment_flow_features modifies it in place (it shouldn't, it returns a new df).
    original_nodes_df_for_pyg = nodes_df.copy() 

    if edges_df is not None and not edges_df.empty:
        print(f"Calculating flow features using last {args.num_recent_months} months and {args.high_demand_quantile} quantile for high demand...")
        try:
            # nodes_df will be augmented with new features and 'is_high_future_demand'
            nodes_df = calculate_establishment_flow_features(
                nodes_df, # Pass the original nodes_df
                edges_df, 
                num_recent_months=args.num_recent_months,
                high_demand_threshold_quantile=args.high_demand_quantile
            )
            print("Flow features calculated and 'is_high_future_demand' label generated.")
            # Ensure 'is_high_future_demand' is boolean for PyG processing
            if 'is_high_future_demand' in nodes_df.columns:
                 nodes_df['is_high_future_demand'] = nodes_df['is_high_future_demand'].astype(bool)
        except ValueError as e:
            print(f"Warning: Could not calculate flow features: {e}. Proceeding without them.")
            # Ensure nodes_df remains the original if calculation fails
            nodes_df = original_nodes_df_for_pyg.copy() 
    else:
        print("Warning: edges_df not available or empty. Skipping flow feature calculation. 'is_high_future_demand' label will not be available.")
        # Ensure nodes_df is the original version if edges_df is missing
        nodes_df = original_nodes_df_for_pyg.copy()


    # --- Preprocess Node Features for GNN ---
    # Define numerical columns that might have been added by feature engineering
    numerical_cols_from_feature_eng = ['total_members_in_recent', 'total_members_out_recent', 'net_inflow_recent', 'avg_monthly_net_inflow_recent']
    # Filter to include only those actually present in nodes_df (handles cases where calculation was skipped or failed)
    actual_numerical_cols_for_gnn = [col for col in numerical_cols_from_feature_eng if col in nodes_df.columns]
    
    if actual_numerical_cols_for_gnn:
        print(f"Using the following numerical columns for GNN features: {actual_numerical_cols_for_gnn}")
    else:
        print("No numerical flow features will be used in the GNN (either not calculated or not found).")

    # A copy of nodes_df (potentially augmented) is passed for preprocessing.
    # The original_nodes_df_for_pyg is passed for label extraction to create_pyg_data_object,
    # ensuring labels are from the state *before* feature engineering might add 'is_high_future_demand' as a feature column.
    # However, if 'is_high_future_demand' is the target, it *must* be in the DataFrame used for label extraction.
    # So, if 'is_high_future_demand' is the target, the augmented nodes_df should be used for label extraction.
    
    df_for_feature_preprocessing = nodes_df.copy()
    df_for_label_extraction = nodes_df.copy() if args.target_label == 'is_high_future_demand' else original_nodes_df_for_pyg.copy()

    processed_node_features_df, feature_names = preprocess_node_features(
        df_for_feature_preprocessing, # This nodes_df contains flow features if calculated
        numerical_feature_cols=actual_numerical_cols_for_gnn
    )
    
    # Convert NetworkX graph and feature DataFrames into a PyTorch Geometric Data object
    # df_for_label_extraction is used here.
    pyg_data, node_id_map = create_pyg_data_object(
        nx_graph,                       # NetworkX graph
        processed_node_features_df,     # DataFrame with processed features (OHE categorical + scaled numerical)
        df_for_label_extraction,        # DataFrame for extracting labels (original or augmented if target is new)
        feature_names                   # List of feature column names in processed_node_features_df
    )
    
    print(f"PyG Data object created: {pyg_data}")
    print(f"Node features shape (x): {pyg_data.x.shape}")
    if hasattr(pyg_data, 'edge_attr') and pyg_data.edge_attr is not None:
        print(f"Edge attributes shape (edge_attr): {pyg_data.edge_attr.shape}")
    else:
        print("No edge attributes (edge_attr) found in pyg_data.")


    # --- Create Train/Validation/Test Masks ---
    # These masks are boolean tensors indicating which nodes belong to which set.
    num_nodes = pyg_data.num_nodes
    indices = np.arange(num_nodes)
    np.random.seed(42) # For reproducibility of splits
    np.random.shuffle(indices)
    
    if num_nodes == 0:
        print("Error: No nodes found in the graph. Exiting.")
        return

    # Define split points for 70% train, 15% validation, 15% test
    train_end_idx = int(0.7 * num_nodes)
    val_end_idx = int(0.85 * num_nodes)
    
    # Initialize masks as all False
    pyg_data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    pyg_data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    pyg_data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    # Assign nodes to masks based on shuffled indices
    pyg_data.train_mask[indices[:train_end_idx]] = True
    pyg_data.val_mask[indices[train_end_idx:val_end_idx]] = True
    pyg_data.test_mask[indices[val_end_idx:]] = True

    print(f"Train mask: {pyg_data.train_mask.sum().item()} nodes")
    print(f"Validation mask: {pyg_data.val_mask.sum().item()} nodes")
    print(f"Test mask: {pyg_data.test_mask.sum().item()} nodes")


    # --- GNN Encoder Initialization ---
    # Common arguments for all encoder types
    encoder_params = {
        'in_channels': pyg_data.x.shape[1],      # Number of input node features
        'hidden_channels': args.hidden_channels, # Size of hidden layers in encoder
        'out_channels': args.embedding_size,     # Output dimension of the encoder (embedding size)
        'num_layers': args.num_gnn_layers,       # Number of GNN layers
        'dropout_rate': args.dropout_rate        # Dropout rate
    }
    # Select and initialize the GNN encoder based on user input
    if args.gnn_model == 'gcn':
        encoder = GCNEncoder(**encoder_params)
    elif args.gnn_model == 'sage':
        encoder = SAGEEncoder(**encoder_params)
    elif args.gnn_model == 'gat':
        encoder_params['heads'] = args.gat_heads # Add GAT-specific 'heads' argument
        encoder = GATEncoder(**encoder_params)
    else:
        # This should not be reached due to argparse choices constraint
        raise ValueError(f"Unsupported GNN model: {args.gnn_model}")
    
    print(f"Initialized {args.gnn_model.upper()} Encoder: {encoder}")

    # --- Application Specific Logic ---
    if args.application == 'classification':
        # --- Node Classification ---
        print(f"\n--- Running Node Classification for target: {args.target_label} ---")
        
        # Construct the target label attribute name (e.g., 'y_industry')
        target_label_attribute_name = f'y_{args.target_label}'
        if not hasattr(pyg_data, target_label_attribute_name):
            available_y_attrs = [attr for attr in dir(pyg_data) if attr.startswith('y_')]
            raise ValueError(f"Target label attribute '{target_label_attribute_name}' not found in PyG data object. Available y attributes: {available_y_attrs}")
        
        # Determine number of classes using the label_encoders stored in pyg_data
        if not hasattr(pyg_data, 'label_encoders') or args.target_label not in pyg_data.label_encoders:
            raise ValueError(f"Label encoder for target '{args.target_label}' not found in pyg_data.label_encoders.")
        num_classes = len(pyg_data.label_encoders[args.target_label].classes_)
        
        # Initialize the NodeClassifier model, optimizer, and loss function
        classifier_model = NodeClassifier(encoder, num_classes=num_classes)
        optimizer = torch.optim.Adam(classifier_model.parameters(), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss()

        print(f"Training Node Classifier for {args.epochs} epochs...")
        train_node_classifier(classifier_model, pyg_data, args.target_label, optimizer, criterion, args.epochs, pyg_data_object=pyg_data)
        
        print("Evaluating Node Classifier...")
        accuracy = evaluate_node_classifier(classifier_model, pyg_data, args.target_label, 'test_mask')
        print(f"Test Accuracy for '{args.target_label}': {accuracy:.4f}")

        # --- Enhanced Output for 'is_high_future_demand' ---
        if args.target_label == 'is_high_future_demand':
            print("\n--- Establishments Predicted as High Future Demand (Test Set Insights) ---")
            classifier_model.eval() # Ensure model is in eval mode
            with torch.no_grad():
                all_logits = classifier_model(pyg_data) # Get logits for all nodes in pyg_data
                predictions = all_logits.argmax(dim=1)  # Get predictions (0 or 1)

            test_mask = getattr(pyg_data, 'test_mask', None)
            if test_mask is None or not test_mask.any():
                print("Warning: Test mask not found or empty. Cannot provide detailed test set insights.")
            else:
                output_data_list = []
                # Iterate over indices of nodes that are in the test set
                for node_idx_tensor in test_mask.nonzero(as_tuple=True)[0]:
                    node_idx = node_idx_tensor.item() # Convert tensor to int
                    
                    # Map PyG index back to original establishment ID
                    original_establishment_id = pyg_data.idx_to_node_id.get(node_idx, 'Unknown_ID')
                    
                    # Retrieve original node info from the potentially augmented nodes_df
                    # Use .loc with boolean indexing to find the row for the establishment_id
                    node_info_series = nodes_df.loc[nodes_df['establishment_id'] == original_establishment_id]
                    
                    if not node_info_series.empty:
                        node_info = node_info_series.iloc[0] # Get the first (and should be only) row
                        
                        # Get true label from pyg_data's y_is_high_future_demand attribute
                        true_label_val = getattr(pyg_data, f'y_{args.target_label}', None)
                        true_label = true_label_val[node_idx].item() if true_label_val is not None else 'N/A'
                        
                        output_data_list.append({
                            'establishment_id': original_establishment_id,
                            'name': node_info.get('name', 'N/A'),
                            'city': node_info.get('city', 'N/A'),
                            'net_inflow_recent': node_info.get('net_inflow_recent', 0), # Default to 0 if not found
                            'avg_monthly_net_inflow_recent': node_info.get('avg_monthly_net_inflow_recent', 0.0),
                            'true_label': true_label,
                            'predicted_label': predictions[node_idx].item() # Prediction for this node_idx
                        })
                    else:
                        print(f"Warning: Could not find details for establishment ID {original_establishment_id} (index {node_idx}) in nodes_df.")

                if output_data_list:
                    output_df = pd.DataFrame(output_data_list)
                    predicted_high_demand_df = output_df[output_df['predicted_label'] == 1] # Assuming 1 is True for 'is_high_future_demand'
                    
                    print(f"\nTop {min(20, len(predicted_high_demand_df))} Establishments Predicted as High Demand (from Test Set):")
                    print(predicted_high_demand_df.head(20))

                    if not predicted_high_demand_df.empty:
                        print("\n--- Summary: Count of Predicted High Demand Establishments by City (from Test Set) ---")
                        city_summary_df = predicted_high_demand_df.groupby('city').size().reset_index(name='count_predicted_high_demand')
                        print(city_summary_df.sort_values(by='count_predicted_high_demand', ascending=False))
                else:
                    print("No data to display for high demand predictions on the test set.")

    elif args.application == 'clustering':
        # --- Node Clustering ---
        print(f"\n--- Running Node Clustering with K-Means (k={args.n_clusters}) ---")
        
        print("Training GNN Autoencoder to learn node embeddings for clustering...")
        # For clustering, embeddings are often learned via an autoencoder framework.
        # The specified GNN encoder (`encoder`) is used as the encoder part of this autoencoder.
        autoencoder_for_clustering_embeddings = GNNAutoencoder(
            encoder=encoder, 
            decoder_hidden_dims=args.decoder_hidden_dims,    # Structure of the decoder
            reconstructed_feature_dim=pyg_data.x.shape[1], # AE aims to reconstruct original features
            dropout_rate=args.dropout_rate
        )
        ae_optimizer_clustering = torch.optim.Adam(autoencoder_for_clustering_embeddings.parameters(), lr=args.lr)
        ae_criterion_clustering = torch.nn.MSELoss() # MSE for reconstruction loss
        
        # Train the autoencoder
        train_gnn_autoencoder(autoencoder_for_clustering_embeddings, pyg_data, ae_optimizer_clustering, ae_criterion_clustering, args.epochs)
        
        print("Extracting node embeddings from the trained autoencoder's encoder part...")
        # Use the encoder component (which has been trained as part of the autoencoder) to get embeddings
        embeddings_for_clustering = get_node_embeddings(autoencoder_for_clustering_embeddings.encoder, pyg_data)
        
        print(f"Performing K-Means clustering with n_clusters={args.n_clusters}...")
        cluster_labels, silhouette_avg = cluster_nodes_kmeans(embeddings_for_clustering, args.n_clusters, random_state=42)
        
        print(f"Clustering complete. Number of nodes per cluster: {np.bincount(cluster_labels) if cluster_labels.size > 0 else 'N/A'}")
        if silhouette_avg is not None:
            print(f"Silhouette Score: {silhouette_avg:.4f}")
        
        # Display sample of clustering results
        if hasattr(pyg_data, 'establishment_ids') and pyg_data.establishment_ids and cluster_labels.size > 0:
            results_df = pd.DataFrame({
                'establishment_id': pyg_data.establishment_ids[:len(cluster_labels)], # Ensure alignment if not all nodes were clustered
                'cluster_id': cluster_labels
            })
            print("\nClustering Results (first 10 rows):")
            print(results_df.head(10))
            # Example: Save results to CSV (requires uncommenting save function import)
            # save_data_to_csv(nx_graph, results_df, ..., "nodes_with_clusters.csv", ...)


    elif args.application == 'anomaly_detection':
        # --- Anomaly Detection ---
        print("\n--- Running Anomaly Detection using GNN Autoencoder ---")
        
        # Initialize GNN Autoencoder for anomaly detection
        # The encoder's output (embedding_size) is the input to the decoder part.
        autoencoder_for_anomaly_detection = GNNAutoencoder(
            encoder=encoder, 
            decoder_hidden_dims=args.decoder_hidden_dims,    # Structure of the decoder
            reconstructed_feature_dim=pyg_data.x.shape[1], # AE reconstructs original features
            dropout_rate=args.dropout_rate
        )
        ae_optimizer_anomaly = torch.optim.Adam(autoencoder_for_anomaly_detection.parameters(), lr=args.lr)
        ae_criterion_anomaly = torch.nn.MSELoss() # MSE for reconstruction loss

        print(f"Training GNN Autoencoder for anomaly detection ({args.epochs} epochs)...")
        train_gnn_autoencoder(autoencoder_for_anomaly_detection, pyg_data, ae_optimizer_anomaly, ae_criterion_anomaly, args.epochs)
        
        print("Calculating reconstruction errors for each node...")
        reconstruction_errors = get_reconstruction_errors(autoencoder_for_anomaly_detection, pyg_data)
        
        # Identify and display top N anomalous nodes
        if hasattr(pyg_data, 'establishment_ids') and pyg_data.establishment_ids:
            anomalous_nodes = get_anomalies_by_error(reconstruction_errors, pyg_data.establishment_ids, top_n=args.anomaly_top_n)
            
            # For tasks like 'Detecting fraudulent claims' or 'Identifying shell companies',
            # examine these anomalous establishments. Look for:
            # - Mismatches between declared size/industry and transfer activity (requires manual cross-referencing with transfer data).
            # - Unusual feature combinations (e.g., an industry type that doesn't fit the city or size).
            # - Establishments with very high error scores that are also isolated or part of small, tightly connected suspicious clusters (requires further graph analysis).
            print(f"\nTop {args.anomaly_top_n} Anomalous Establishments (by reconstruction error):")
            
            # Assuming 'nodes_df' is the DataFrame holding original node attributes and is indexed by 'establishment_id'
            # If not, adjust the lookup accordingly.
            # Make sure nodes_df is available in this scope. It's loaded as 'nodes_df' at the beginning of main().
            # We might need to set 'establishment_id' as index for quick lookup if it's not already.
            original_nodes_df_indexed = nodes_df.set_index('establishment_id')

            for est_id, error_score in anomalous_nodes:
                try:
                    node_details = original_nodes_df_indexed.loc[est_id]
                    name = node_details.get('name', 'N/A')
                    industry = node_details.get('industry', 'N/A')
                    city = node_details.get('city', 'N/A')
                    size_category = node_details.get('size_category', 'N/A')
                    print(f"- ID: {est_id}, Name: {name}, Industry: {industry}, City: {city}, Size: {size_category}, Error: {error_score:.4f}")
                except KeyError:
                    # This might happen if an ID from pyg_data.establishment_ids is not in original_nodes_df_indexed
                    # Or if get_anomalies_by_error returns IDs not present in the original DataFrame.
                    print(f"- ID: {est_id}, Error: {error_score:.4f} (Details not found in original nodes_df)")
        else:
            print("Could not retrieve establishment IDs to report anomalies by ID. Showing errors by index if available.")
            # Fallback: print top errors by index if node IDs are not available
            sorted_error_indices = np.argsort(reconstruction_errors)[::-1]
            for i in range(min(args.anomaly_top_n, len(reconstruction_errors))):
                idx = sorted_error_indices[i]
                print(f"- Node Index: {idx}, Reconstruction Error: {reconstruction_errors[idx]:.4f}")
    else:
        # This case should ideally not be reached due to argparse `choices` constraint
        print(f"Application '{args.application}' is not recognized or implemented.")


if __name__ == '__main__':
    main()
import numpy as np
import pandas as pd
from gnn_applications import (
    load_establishment_data,
    preprocess_node_features,
    create_pyg_data_object,
    GCNEncoder,
    SAGEEncoder,
    GATEncoder,
    NodeClassifier,
    train_node_classifier,
    evaluate_node_classifier,
    get_node_embeddings,
    cluster_nodes_kmeans,
    GNNAutoencoder,
    train_gnn_autoencoder,
    get_reconstruction_errors,
    get_anomalies_by_error
)
# from data import save_data_to_csv # For potentially saving results - uncomment if needed

# Default parameters (can be overridden by argparse)
DEFAULT_NODES_FILE = "epfo_establishments_nodes.csv"
DEFAULT_EDGES_FILE = "epfo_transfers_edges.csv"
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_EPOCHS = 100 # Reduced for quick testing; can be increased
DEFAULT_HIDDEN_CHANNELS = 64
DEFAULT_EMBEDDING_SIZE = 32 # Output size of GNN encoders

def main():
    parser = argparse.ArgumentParser(description="Run GNN Applications on EPFO Data")

    # General arguments
    parser.add_argument('--nodes_file', type=str, default=DEFAULT_NODES_FILE, help="Path to nodes CSV file.")
    parser.add_argument('--edges_file', type=str, default=DEFAULT_EDGES_FILE, help="Path to edges CSV file.")
    parser.add_argument('--gnn_model', type=str, choices=['gcn', 'sage', 'gat'], required=True, help="GNN model to use.")
    parser.add_argument('--application', type=str, choices=['classification', 'clustering', 'anomaly_detection'], required=True, help="Application to run.")
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate.")
    parser.add_argument('--hidden_channels', type=int, default=DEFAULT_HIDDEN_CHANNELS, help="Number of hidden units in GNN layers.")
    parser.add_argument('--embedding_size', type=int, default=DEFAULT_EMBEDDING_SIZE, help="Dimension of node embeddings output by the encoder.")
    parser.add_argument('--num_gnn_layers', type=int, default=2, help="Number of GNN layers in encoders.")
    parser.add_argument('--dropout_rate', type=float, default=0.5, help="Dropout rate.")
    parser.add_argument('--gat_heads', type=int, default=8, help="Number of attention heads for GAT model.")


    # Arguments for Classification
    # Arguments for feature engineering
    parser.add_argument('--num_recent_months', type=int, default=6, help='Number of recent months for flow feature calculation.')
    parser.add_argument('--high_demand_quantile', type=float, default=0.8, help='Quantile for net inflow to define high-demand label.')

    # Arguments for Classification
    parser.add_argument('--target_label', type=str, default='industry', choices=['industry', 'city', 'size_category', 'is_high_future_demand'], help="Target label for node classification.")

    # Arguments for Clustering
    parser.add_argument('--n_clusters', type=int, default=5, help="Number of clusters for K-Means.")

    # Arguments for Anomaly Detection (Autoencoder)
    # Example: --decoder_hidden_dims 64  (for a single hidden layer in decoder)
    # Example: --decoder_hidden_dims 64 32 (for two hidden layers in decoder)
    parser.add_argument('--decoder_hidden_dims', nargs='+', type=int, default=[DEFAULT_HIDDEN_CHANNELS], help=f"List of hidden dimensions for autoencoder's decoder. Default: [{DEFAULT_HIDDEN_CHANNELS}]")
    parser.add_argument('--anomaly_top_n', type=int, default=10, help="Number of top anomalies to report.")

    args = parser.parse_args()

    print("Loading and preprocessing data...")
    nx_graph, nodes_df, edges_df = load_establishment_data(args.nodes_file, args.edges_file) # Capture edges_df

    # Data Augmentation (Feature Engineering)
    original_nodes_df_for_pyg = nodes_df.copy() # Save a snapshot before potential augmentation

    if edges_df is not None and not edges_df.empty:
        print(f"Calculating flow features using last {args.num_recent_months} months and {args.high_demand_quantile} quantile...")
        try:
            nodes_df = calculate_establishment_flow_features(
                nodes_df, 
                edges_df, 
                num_recent_months=args.num_recent_months,
                high_demand_threshold_quantile=args.high_demand_quantile
            )
            print("Flow features calculated and 'is_high_future_demand' label generated.")
            if 'is_high_future_demand' in nodes_df.columns:
                 nodes_df['is_high_future_demand'] = nodes_df['is_high_future_demand'].astype(bool)
        except ValueError as e:
            print(f"Warning: Could not calculate flow features: {e}. Proceeding without them.")
            nodes_df = original_nodes_df_for_pyg.copy() # Revert to original if error
    else:
        print("Warning: edges_df not available or empty. Skipping flow feature calculation.")
        nodes_df = original_nodes_df_for_pyg.copy() # Use original if no edges

    # Preprocess Node Features for GNN
    numerical_cols_for_gnn = ['total_members_in_recent', 'total_members_out_recent', 'net_inflow_recent', 'avg_monthly_net_inflow_recent']
    actual_numerical_cols = [col for col in numerical_cols_for_gnn if col in nodes_df.columns]
    
    if actual_numerical_cols:
        print(f"Using numerical columns for GNN: {actual_numerical_cols}")
    else:
        print("No numerical flow features will be used for GNN.")

    # Decide which nodes_df to use for label extraction in create_pyg_data_object
    # If target is 'is_high_future_demand', it must be in the df passed for label extraction.
    df_for_label_extraction = nodes_df.copy() if args.target_label == 'is_high_future_demand' else original_nodes_df_for_pyg.copy()

    processed_node_features_df, feature_names = preprocess_node_features(
        nodes_df.copy(), # Pass potentially augmented nodes_df for feature creation
        numerical_feature_cols=actual_numerical_cols
    )
    
    # create_pyg_data_object returns data, node_id_map
    pyg_data, node_id_map = create_pyg_data_object(
        nx_graph, 
        processed_node_features_df, 
        df_for_label_extraction, # Use the appropriate df for labels
        feature_names
    ) 
    
    print(f"PyG Data object created: {pyg_data}")
    print(f"Node features shape: {pyg_data.x.shape}")
    if hasattr(pyg_data, 'edge_attr') and pyg_data.edge_attr is not None:
        print(f"Edge attributes shape: {pyg_data.edge_attr.shape}")
    else:
        print("No edge attributes (edge_attr) found in pyg_data.")


    # Create train/val/test masks
    num_nodes = pyg_data.num_nodes
    indices = np.arange(num_nodes)
    np.random.seed(42) # for reproducibility
    np.random.shuffle(indices)
    
    # Check if any nodes exist
    if num_nodes == 0:
        print("Error: No nodes found in the graph. Exiting.")
        return

    train_end = int(0.7 * num_nodes)
    val_end = int(0.85 * num_nodes)
    
    pyg_data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    pyg_data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    pyg_data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    pyg_data.train_mask[indices[:train_end]] = True
    pyg_data.val_mask[indices[train_end:val_end]] = True
    pyg_data.test_mask[indices[val_end:]] = True

    print(f"Train mask: {pyg_data.train_mask.sum().item()} nodes")
    print(f"Validation mask: {pyg_data.val_mask.sum().item()} nodes")
    print(f"Test mask: {pyg_data.test_mask.sum().item()} nodes")


    # Initialize GNN Encoder
    encoder_args = {
        'in_channels': pyg_data.x.shape[1],
        'hidden_channels': args.hidden_channels,
        'out_channels': args.embedding_size, # Encoder outputs embeddings of this size
        'num_layers': args.num_gnn_layers,
        'dropout_rate': args.dropout_rate
    }
    if args.gnn_model == 'gcn':
        encoder = GCNEncoder(**encoder_args)
    elif args.gnn_model == 'sage':
        encoder = SAGEEncoder(**encoder_args)
    elif args.gnn_model == 'gat':
        encoder_args['heads'] = args.gat_heads 
        encoder = GATEncoder(**encoder_args)
    else:
        raise ValueError(f"Unsupported GNN model: {args.gnn_model}")
    
    print(f"Initialized {args.gnn_model.upper()} Encoder: {encoder}")

    # --- Application Specific Logic ---
    if args.application == 'classification':
        print(f"\n--- Running Node Classification for target: {args.target_label} ---")
        
        target_label_attr = f'y_{args.target_label}'
        if not hasattr(pyg_data, target_label_attr):
            raise ValueError(f"Target label attribute '{target_label_attr}' not found in PyG data object. Available y attributes: {[attr for attr in dir(pyg_data) if attr.startswith('y_')]}")
        
        # Determine num_classes from the pyg_data object's mapping
        num_classes = len(getattr(pyg_data, f'{target_label_attr}_mapping'))
        
        classifier_model = NodeClassifier(encoder, num_classes=num_classes) # Encoder's out_channels is args.embedding_size
        optimizer = torch.optim.Adam(classifier_model.parameters(), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss()

        print(f"Training Node Classifier for {args.epochs} epochs...")
        # Pass the full pyg_data object for label mapping access if needed by train/eval functions
        train_node_classifier(classifier_model, pyg_data, args.target_label, optimizer, criterion, args.epochs, pyg_data_object=pyg_data)
        
        print("Evaluating Node Classifier...")
        accuracy = evaluate_node_classifier(classifier_model, pyg_data, args.target_label, 'test_mask')
        print(f"Test Accuracy for '{args.target_label}': {accuracy:.4f}")

        # Enhanced Output for 'is_high_future_demand'
        if args.target_label == 'is_high_future_demand':
            print("\n--- Establishments Predicted as High Future Demand (Test Set) ---")
            classifier_model.eval()
            with torch.no_grad():
                all_logits = classifier_model(pyg_data) # Use pyg_data which has all nodes
                predictions = all_logits.argmax(dim=1)

            test_mask = getattr(pyg_data, 'test_mask', None)
            if test_mask is None or not test_mask.any():
                print("Warning: Test mask not available or empty. Cannot provide detailed test insights.")
            else:
                output_data = []
                # Iterate over nodes in the test set
                for node_pyg_idx_tensor in test_mask.nonzero(as_tuple=True)[0]:
                    node_pyg_idx = node_pyg_idx_tensor.item()
                    original_id = pyg_data.idx_to_node_id.get(node_pyg_idx, "Unknown_ID")
                    
                    # Fetch details from the (potentially augmented) nodes_df
                    node_details_series = nodes_df[nodes_df['establishment_id'] == original_id]
                    if not node_details_series.empty:
                        node_details = node_details_series.iloc[0]
                        true_label_tensor = getattr(pyg_data, f'y_{args.target_label}', None)
                        true_label_val = true_label_tensor[node_pyg_idx].item() if true_label_tensor is not None else 'N/A'

                        output_data.append({
                            'establishment_id': original_id,
                            'name': node_details.get('name', 'N/A'),
                            'city': node_details.get('city', 'N/A'),
                            'net_inflow_recent': node_details.get('net_inflow_recent', 'N/A'),
                            'avg_monthly_net_inflow_recent': node_details.get('avg_monthly_net_inflow_recent', 'N/A'),
                            'true_label': true_label_val,
                            'predicted_label': predictions[node_pyg_idx].item()
                        })
                    else:
                        print(f"Warning: Could not find details for node ID {original_id} (PyG index {node_pyg_idx}).")
                
                if output_data:
                    output_df = pd.DataFrame(output_data)
                    predicted_high_demand_df = output_df[output_df['predicted_label'] == 1] # Assuming 1 means True
                    
                    print(f"\nTop {min(20, len(predicted_high_demand_df))} Establishments Predicted as High Demand (Test Set):")
                    print(predicted_high_demand_df.head(20))

                    if not predicted_high_demand_df.empty:
                        print("\n--- Cities with Predicted High Demand Establishments (Test Set) ---")
                        city_summary = predicted_high_demand_df.groupby('city').size().reset_index(name='count_predicted_high_demand')
                        print(city_summary.sort_values(by='count_predicted_high_demand', ascending=False))
                else:
                    print("No data to display for high demand predictions (Test Set).")


    elif args.application == 'clustering':
        print(f"\n--- Running Node Clustering with K-Means (k={args.n_clusters}) ---")
        
        print("Training GNN Autoencoder to get meaningful embeddings for clustering...")
        # The encoder passed to GNNAutoencoder will be trained as part of the autoencoder.
        # Its output dimension is args.embedding_size.
        autoencoder_for_clustering = GNNAutoencoder(
            encoder=encoder, 
            decoder_hidden_dims=args.decoder_hidden_dims, 
            reconstructed_feature_dim=pyg_data.x.shape[1], # Reconstruct original features
            dropout_rate=args.dropout_rate
        )
        ae_optimizer_clustering = torch.optim.Adam(autoencoder_for_clustering.parameters(), lr=args.lr)
        ae_criterion_clustering = torch.nn.MSELoss()
        
        train_gnn_autoencoder(autoencoder_for_clustering, pyg_data, ae_optimizer_clustering, ae_criterion_clustering, args.epochs)
        
        print("Extracting node embeddings from the trained autoencoder's encoder part...")
        # Use the encoder part (which was trained) from the autoencoder
        embeddings = get_node_embeddings(autoencoder_for_clustering.encoder, pyg_data)
        
        print(f"Performing K-Means clustering with n_clusters={args.n_clusters}...")
        cluster_labels, silhouette = cluster_nodes_kmeans(embeddings, args.n_clusters, random_state=42)
        
        print(f"Clustering complete. Number of nodes per cluster: {np.bincount(cluster_labels) if cluster_labels.size > 0 else 'N/A'}")
        if silhouette is not None:
            print(f"Silhouette Score: {silhouette:.4f}")
        
        if hasattr(pyg_data, 'establishment_ids') and pyg_data.establishment_ids and cluster_labels.size > 0:
            results_df = pd.DataFrame({
                'establishment_id': pyg_data.establishment_ids, 
                'cluster_id': cluster_labels
            })
            print("\nClustering Results (first 10 rows):")
            print(results_df.head(10))
            # Example save:
            # from data import save_graph_data # Assuming save_graph_data can save nodes DataFrame
            # save_graph_data(nx_graph, results_df, edges_df, "nodes_with_clusters.csv", "edges_data.csv")


    elif args.application == 'anomaly_detection':
        print("\n--- Running Anomaly Detection using GNN Autoencoder ---")
        
        # Encoder's out_channels (embedding_size) is the input to the decoder part of GNNAutoencoder
        autoencoder_for_anomaly = GNNAutoencoder(
            encoder=encoder, 
            decoder_hidden_dims=args.decoder_hidden_dims,
            reconstructed_feature_dim=pyg_data.x.shape[1], # AE reconstructs original features
            dropout_rate=args.dropout_rate
        )
        ae_optimizer_anomaly = torch.optim.Adam(autoencoder_for_anomaly.parameters(), lr=args.lr)
        ae_criterion_anomaly = torch.nn.MSELoss()

        print(f"Training GNN Autoencoder for {args.epochs} epochs...")
        train_gnn_autoencoder(autoencoder_for_anomaly, pyg_data, ae_optimizer_anomaly, ae_criterion_anomaly, args.epochs)
        
        print("Calculating reconstruction errors...")
        errors = get_reconstruction_errors(autoencoder_for_anomaly, pyg_data)
        
        if hasattr(pyg_data, 'establishment_ids') and pyg_data.establishment_ids:
            anomalies = get_anomalies_by_error(errors, pyg_data.establishment_ids, top_n=args.anomaly_top_n)
            
            # For tasks like 'Detecting fraudulent claims' or 'Identifying shell companies',
            # examine these anomalous establishments. Look for:
            # - Mismatches between declared size/industry and transfer activity (requires manual cross-referencing with transfer data).
            # - Unusual feature combinations (e.g., an industry type that doesn't fit the city or size).
            # - Establishments with very high error scores that are also isolated or part of small, tightly connected suspicious clusters (requires further graph analysis).
            print(f"\nTop {args.anomaly_top_n} Anomalous Establishments (by reconstruction error):")

            # Use the 'nodes_df' DataFrame loaded at the start of main()
            # Set 'establishment_id' as index for efficient lookup
            original_nodes_df_indexed = nodes_df.set_index('establishment_id')

            for est_id, error in anomalies:
                try:
                    node_details = original_nodes_df_indexed.loc[est_id]
                    name = node_details.get('name', 'N/A')
                    industry = node_details.get('industry', 'N/A')
                    city = node_details.get('city', 'N/A')
                    size_category = node_details.get('size_category', 'N/A')
                    print(f"- ID: {est_id}, Name: {name}, Industry: {industry}, City: {city}, Size: {size_category}, Error: {error:.4f}")
                except KeyError:
                    # Fallback if est_id is not found in the indexed DataFrame
                    print(f"- ID: {est_id}, Error: {error:.4f} (Details not found in original nodes_df)")
        else:
            print("Could not retrieve establishment IDs to report anomalies.")

    else:
        # This case should not be reached due to argparse choices constraint
        print(f"Application '{args.application}' is not recognized.")


if __name__ == '__main__':
    main()
