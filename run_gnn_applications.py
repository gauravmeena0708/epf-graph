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


    # Arguments specific to Node Classification
    parser.add_argument('--target_label', type=str, default='industry', choices=['industry', 'city', 'size_category'], help="Target node attribute for classification.")

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
    nx_graph, nodes_df, _ = load_establishment_data(args.nodes_file, args.edges_file)
    # Preprocess node features (e.g., one-hot encoding categorical attributes)
    # A copy of nodes_df is made to prevent modifications to the original DataFrame during preprocessing.
    node_features_df, feature_names = preprocess_node_features(nodes_df.copy()) 
    
    # Convert NetworkX graph and feature DataFrames into a PyTorch Geometric Data object
    # Another copy of nodes_df is used for extracting original labels without interference from preprocessing.
    pyg_data, node_id_map = create_pyg_data_object(nx_graph, node_features_df, nodes_df.copy(), feature_names) 
    
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
        
        # Determine number of classes from the label mapping stored in pyg_data
        # (e.g., pyg_data.y_industry_mapping)
        num_classes = len(getattr(pyg_data, f'{target_label_attribute_name}_mapping'))
        
        # Initialize the NodeClassifier model, optimizer, and loss function
        classifier_model = NodeClassifier(encoder, num_classes=num_classes)
        optimizer = torch.optim.Adam(classifier_model.parameters(), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss()

        print(f"Training Node Classifier for {args.epochs} epochs...")
        train_node_classifier(classifier_model, pyg_data, args.target_label, optimizer, criterion, args.epochs, pyg_data_object=pyg_data)
        
        print("Evaluating Node Classifier...")
        accuracy = evaluate_node_classifier(classifier_model, pyg_data, args.target_label, 'test_mask')
        print(f"Test Accuracy for '{args.target_label}': {accuracy:.4f}")

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
            print(f"\nTop {args.anomaly_top_n} Anomalous Establishments (by reconstruction error):")
            for est_id, error_score in anomalous_nodes:
                print(f"- Establishment ID: {est_id}, Reconstruction Error: {error_score:.4f}")
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
    parser.add_argument('--target_label', type=str, default='industry', choices=['industry', 'city', 'size_category'], help="Target label for node classification.")

    # Arguments for Clustering
    parser.add_argument('--n_clusters', type=int, default=5, help="Number of clusters for K-Means.")

    # Arguments for Anomaly Detection (Autoencoder)
    # Example: --decoder_hidden_dims 64  (for a single hidden layer in decoder)
    # Example: --decoder_hidden_dims 64 32 (for two hidden layers in decoder)
    parser.add_argument('--decoder_hidden_dims', nargs='+', type=int, default=[DEFAULT_HIDDEN_CHANNELS], help=f"List of hidden dimensions for autoencoder's decoder. Default: [{DEFAULT_HIDDEN_CHANNELS}]")
    parser.add_argument('--anomaly_top_n', type=int, default=10, help="Number of top anomalies to report.")

    args = parser.parse_args()

    print("Loading and preprocessing data...")
    nx_graph, nodes_df, _ = load_establishment_data(args.nodes_file, args.edges_file)
    node_features_df, feature_names = preprocess_node_features(nodes_df.copy()) 
    
    # create_pyg_data_object returns data, node_id_map
    pyg_data, node_id_map = create_pyg_data_object(nx_graph, node_features_df, nodes_df.copy(), feature_names) 
    
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
            print(f"\nTop {args.anomaly_top_n} Anomalous Establishments (by reconstruction error):")
            for est_id, error in anomalies:
                print(f"- Establishment ID: {est_id}, Reconstruction Error: {error:.4f}")
        else:
            print("Could not retrieve establishment IDs to report anomalies.")

    else:
        # This case should not be reached due to argparse choices constraint
        print(f"Application '{args.application}' is not recognized.")


if __name__ == '__main__':
    main()
