# Core PyTorch and PyTorch Geometric imports
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import to_networkx, from_networkx, negative_sampling # Add other PyG utils as needed

# Data manipulation and machine learning imports
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, accuracy_score, silhouette_score
# Add other sklearn utilities as needed, e.g., for anomaly detection or other metrics

# Standard Python libraries
import random
from collections import Counter

# Assuming data.py is in the same directory or accessible via PYTHONPATH
from data import load_graph_data # For loading graph data from CSV files


def load_establishment_data(nodes_filepath="epfo_establishments_nodes.csv", edges_filepath="epfo_transfers_edges.csv"):
    """
    Loads graph data (NetworkX graph, nodes DataFrame, edges DataFrame) from specified CSV files.

    This function serves as a wrapper around `data.load_graph_data` to simplify loading
    the specific establishment and transfer data files.

    Args:
        nodes_filepath (str, optional): Path to the CSV file containing node information.
                                        Defaults to "epfo_establishments_nodes.csv".
        edges_filepath (str, optional): Path to the CSV file containing edge information.
                                        Defaults to "epfo_transfers_edges.csv".

    Returns:
        tuple:
            - nx_graph (networkx.Graph): The loaded graph as a NetworkX object.
            - nodes_df (pd.DataFrame): DataFrame containing node attributes.
            - edges_df (pd.DataFrame): DataFrame containing edge attributes.
    """
    nx_graph, nodes_df, edges_df = load_graph_data(nodes_filepath, edges_filepath)
    return nx_graph, nodes_df, edges_df


def preprocess_node_features(nodes_df):
    """
    Preprocesses node features by applying one-hot encoding to categorical attributes.

    Categorical features 'industry', 'city', and 'size_category' are one-hot encoded.
    The 'establishment_id' column is preserved.

    Args:
        nodes_df (pd.DataFrame): DataFrame containing node information, including 'establishment_id'.
                                 It is expected to have 'industry', 'city', 'size_category' columns.

    Returns:
        tuple:
            - features_df (pd.DataFrame): DataFrame with 'establishment_id' and one-hot encoded features.
                                           Original categorical columns are replaced by their one-hot encoded counterparts.
            - feature_cols (list): List of strings representing the names of the generated feature columns
                                   (i.e., all columns in `features_df` except 'establishment_id').
    Raises:
        ValueError: If 'establishment_id' column is missing or if any of the expected
                    categorical feature columns ('industry', 'city', 'size_category') are missing.
    """
    if 'establishment_id' not in nodes_df.columns:
        raise ValueError("nodes_df must contain 'establishment_id' column.")

    # Set establishment_id as index for easy mapping later, and to avoid it being processed as a feature
    # Make a copy to avoid SettingWithCopyWarning if nodes_df is a slice
    nodes_df_indexed = nodes_df.copy().set_index('establishment_id')

    categorical_features = ['industry', 'city', 'size_category']
    
    # Ensure all categorical features are present
    missing_cols = [col for col in categorical_features if col not in nodes_df_indexed.columns]
    if missing_cols:
        raise ValueError(f"Missing categorical columns in nodes_df: {missing_cols}")

    # Apply one-hot encoding using pandas.get_dummies
    # This will create new columns for each category within each feature, prefixed accordingly.
    features_df_encoded = pd.get_dummies(nodes_df_indexed, columns=categorical_features, prefix=categorical_features)
    
    # Reset index so 'establishment_id' becomes a column again
    features_df_encoded = features_df_encoded.reset_index()
    
    # The feature columns are all columns except 'establishment_id'
    # These are the names of the columns that will form the node feature matrix 'x'.
    feature_cols = [col for col in features_df_encoded.columns if col != 'establishment_id']
    
    return features_df_encoded, feature_cols


def create_pyg_data_object(nx_graph, processed_node_features_df, original_nodes_df, feature_cols):
    """
    Creates a PyTorch Geometric Data object from a NetworkX graph and processed node features.

    This function handles:
    1.  Mapping 'establishment_id' to continuous integer indices.
    2.  Constructing the node feature tensor `x` from `processed_node_features_df`.
    3.  Constructing the edge index tensor `edge_index` and edge attribute tensor `edge_attr` (weights).
    4.  Extracting and encoding labels ('industry', 'city', 'size_category') from `original_nodes_df`
        and storing them as `y_<label_name>` attributes on the Data object.
    5.  Storing mapping information (node IDs, label encodings) on the Data object.

    Args:
        nx_graph (networkx.Graph): The input graph. Node IDs are expected to be 'establishment_id'.
        processed_node_features_df (pd.DataFrame): DataFrame with 'establishment_id' and one-hot encoded features.
                                                   The columns listed in `feature_cols` will form the node features.
        original_nodes_df (pd.DataFrame): The original nodes DataFrame, used to extract labels.
                                          Must contain 'establishment_id', and label columns like
                                          'industry', 'city', 'size_category'.
        feature_cols (list): List of column names in `processed_node_features_df` that constitute the features.

    Returns:
        tuple:
            - data (torch_geometric.data.Data): A PyG Data object ready for GNN models.
            - node_id_map (dict): A dictionary mapping original 'establishment_id' to integer indices (0 to N-1).
    Raises:
        ValueError: If 'establishment_id' is missing from input DataFrames.
    """
    if 'establishment_id' not in processed_node_features_df.columns:
        raise ValueError("processed_node_features_df must contain 'establishment_id' column.")
    if 'establishment_id' not in original_nodes_df.columns:
        raise ValueError("original_nodes_df must contain 'establishment_id' column.")

    # Create a mapping from establishment_id to a continuous integer index (0 to N-1)
    # It's crucial that graph_nodes are the same IDs present in the feature and original node DataFrames.
    graph_nodes_original_ids = list(nx_graph.nodes()) # These are the 'establishment_id's
    node_id_map = {node_id: i for i, node_id in enumerate(graph_nodes_original_ids)}
    
    # Align processed_node_features_df with the graph node order for feature matrix 'x'
    # Set 'establishment_id' as index for efficient lookup using .reindex()
    features_df_indexed = processed_node_features_df.set_index('establishment_id')
    # Reorder rows according to graph_nodes_original_ids and select only feature_cols
    aligned_features = features_df_indexed.reindex(graph_nodes_original_ids)[feature_cols]
    
    # Handle cases where nodes in the graph might not have features in processed_node_features_df
    if aligned_features.isnull().values.any():
        # Identify nodes that are in the graph but for which features are missing (NaN after reindex)
        missing_feature_nodes = aligned_features[aligned_features.isnull().any(axis=1)].index.tolist()
        print(f"Warning: Nodes {missing_feature_nodes} are in the graph but missing features. Filling with zeros.")
        # Fill NaN values with 0. This is a common strategy but might not be optimal for all cases.
        aligned_features = aligned_features.fillna(0) 

    # Convert the aligned feature DataFrame to a PyTorch tensor
    x = torch.tensor(aligned_features.values, dtype=torch.float)

    # Create edge_index and edge_attr (edge features, typically weights)
    edge_list = []
    edge_attributes_list = []
    for u_orig, v_orig, edge_data_dict in nx_graph.edges(data=True):
        # Map original node IDs (establishment_id) to their integer indices
        if u_orig in node_id_map and v_orig in node_id_map: # Ensure both nodes are in our map
            u_mapped, v_mapped = node_id_map[u_orig], node_id_map[v_orig]
            edge_list.append((u_mapped, v_mapped))
            # Assume 'weight' attribute exists for edges, as per data.py's typical output.
            # Default to 1.0 if no 'weight' attribute is found on an edge.
            edge_attributes_list.append(edge_data_dict.get('weight', 1.0))
        # else:
            # print(f"Warning: Edge ({u_orig}, {v_orig}) contains nodes not in node_id_map. Skipping this edge.")


    if not edge_list: # Handle graphs with no edges or where all edges were skipped
        edge_index = torch.empty((2, 0), dtype=torch.long) # Shape [2, num_edges]
        edge_attr = torch.empty((0, 1), dtype=torch.float) # Shape [num_edges, num_edge_features]
    else:
        # Convert edge list to PyTorch tensor for edge_index (shape [2, num_edges])
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        # Convert edge attributes to PyTorch tensor (shape [num_edges, num_edge_features])
        edge_attr = torch.tensor(edge_attributes_list, dtype=torch.float).unsqueeze(1) # Make it [num_edges, 1]

    # Create the PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # Store original labels (e.g., industry, city) on the Data object
    # Align original_nodes_df with the graph node order for labels
    original_nodes_df_indexed = original_nodes_df.set_index('establishment_id')
    aligned_original_nodes_for_labels = original_nodes_df_indexed.reindex(graph_nodes_original_ids)

    label_column_names = ['industry', 'city', 'size_category']
    for col_name in label_column_names:
        if col_name in aligned_original_nodes_for_labels:
            le = LabelEncoder()
            # Handle potential NaNs in label columns before encoding, treat them as an 'Unknown' category
            # Ensure labels are string type for consistent encoding
            labels_str = aligned_original_nodes_for_labels[col_name].fillna('Unknown').astype(str)
            encoded_labels = le.fit_transform(labels_str)
            
            # Store encoded labels as y_<label_name> (e.g., data.y_industry)
            setattr(data, f'y_{col_name}', torch.tensor(encoded_labels, dtype=torch.long))
            # Store the mapping from encoded integer back to original string label for interpretation
            setattr(data, f'y_{col_name}_mapping', {i: cls_name for i, cls_name in enumerate(le.classes_)})
        # else:
            # print(f"Warning: Label column '{col_name}' not found in original_nodes_df.")

    # Store mapping information and original IDs on the Data object for convenience
    data.node_id_map = node_id_map # Maps 'establishment_id' to 0-based integer index
    data.idx_to_node_id = {v: k for k, v in node_id_map.items()} # Maps integer index back to 'establishment_id'
    data.establishment_ids = graph_nodes_original_ids # List of 'establishment_id's in the order of the graph nodes

    return data, node_id_map


# GNN Encoder Architectures

class GCNEncoder(torch.nn.Module):
    """
    Graph Convolutional Network (GCN) Encoder.

    This encoder uses a stack of GCNConv layers to generate node embeddings.
    ReLU activation and Dropout are applied between layers.

    Args:
        in_channels (int): Number of input features per node.
        hidden_channels (int): Number of hidden units in intermediate GCNConv layers.
        out_channels (int): Dimension of the output node embeddings.
        num_layers (int, optional): Number of GCNConv layers. Defaults to 2.
        dropout_rate (float, optional): Dropout probability. Defaults to 0.5.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout_rate=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.num_layers = num_layers
        self.out_channels = out_channels # Store for easy access by downstream modules

        if num_layers < 1:
            raise ValueError("Number of layers must be at least 1.")

        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels))
        else:
            # First layer: input_features -> hidden_channels
            self.convs.append(GCNConv(in_channels, hidden_channels))
            # Intermediate layers: hidden_channels -> hidden_channels
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            # Final layer: hidden_channels -> output_embedding_dimension
            self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass of the GCN encoder.

        Args:
            x (torch.Tensor): Node feature matrix of shape [num_nodes, in_channels].
            edge_index (torch.Tensor): Graph connectivity in COO format with shape [2, num_edges].
            edge_weight (torch.Tensor, optional): Edge weights or attributes. Shape [num_edges].
                                                   Defaults to None.

        Returns:
            torch.Tensor: Node embeddings of shape [num_nodes, out_channels].
        """
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight=edge_weight)
            if i < self.num_layers - 1:  # Apply ReLU and Dropout to all but the last layer
                x = F.relu(x)
                x = self.dropout(x)
        return x


class SAGEEncoder(torch.nn.Module):
    """
    GraphSAGE (Sample and Aggregate) Encoder.

    This encoder uses a stack of SAGEConv layers to generate node embeddings.
    ReLU activation and Dropout are applied between layers.

    Args:
        in_channels (int): Number of input features per node.
        hidden_channels (int): Number of hidden units in intermediate SAGEConv layers.
        out_channels (int): Dimension of the output node embeddings.
        num_layers (int, optional): Number of SAGEConv layers. Defaults to 2.
        dropout_rate (float, optional): Dropout probability. Defaults to 0.5.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout_rate=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.num_layers = num_layers
        self.out_channels = out_channels # Store for easy access

        if num_layers < 1:
            raise ValueError("Number of layers must be at least 1.")

        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, out_channels))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        """
        Forward pass of the SAGE encoder.

        Args:
            x (torch.Tensor): Node feature matrix of shape [num_nodes, in_channels].
            edge_index (torch.Tensor): Graph connectivity in COO format with shape [2, num_edges].

        Returns:
            torch.Tensor: Node embeddings of shape [num_nodes, out_channels].
        """
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x


class GATEncoder(torch.nn.Module):
    """
    Graph Attention Network (GAT) Encoder.

    This encoder uses a stack of GATConv layers, incorporating multi-head attention.
    ELU activation and Dropout are applied between layers. The final layer typically uses
    a single head and no concatenation to produce the final embedding dimension.

    Args:
        in_channels (int): Number of input features per node.
        hidden_channels (int): Number of hidden units per head in intermediate GATConv layers.
                               The effective number of hidden units will be hidden_channels * heads.
        out_channels (int): Dimension of the output node embeddings.
        num_layers (int, optional): Number of GATConv layers. Defaults to 2.
        dropout_rate (float, optional): Dropout probability for features between layers
                                        and for attention coefficients within GATConv. Defaults to 0.5.
        heads (int, optional): Number of attention heads in GATConv layers (except possibly the last).
                               Defaults to 8.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout_rate=0.5, heads=8):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        # Dropout for features between layers. GATConv has its own internal dropout for attention weights.
        self.dropout = torch.nn.Dropout(dropout_rate) 
        self.num_layers = num_layers
        self.heads = heads
        self.out_channels = out_channels # Store for easy access

        if num_layers < 1:
            raise ValueError("Number of layers must be at least 1.")

        current_in_channels = in_channels

        if num_layers == 1:
            # If only one layer, it directly maps input to output, usually with 1 head and no concatenation.
            # The GATConv's internal dropout is applied to attention coefficients.
            self.convs.append(GATConv(current_in_channels, out_channels, heads=1, concat=False, dropout=dropout_rate))
        else:
            # First layer: input_features -> hidden_channels * heads
            # GATConv's internal dropout applies to attention coefficients.
            self.convs.append(GATConv(current_in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout_rate))
            current_in_channels = hidden_channels * heads # Output dimension of the first layer

            # Intermediate layers: (hidden_channels * heads) -> (hidden_channels * heads)
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(current_in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout_rate))
                # current_in_channels remains hidden_channels * heads for subsequent intermediate layers
            
            # Final layer: (hidden_channels * heads) -> output_embedding_dimension
            # Typically uses 1 head and no concatenation for the final output.
            # GATConv's internal dropout applies to attention coefficients.
            self.convs.append(GATConv(current_in_channels, out_channels, heads=1, concat=False, dropout=dropout_rate))


    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass of the GAT encoder.

        Args:
            x (torch.Tensor): Node feature matrix of shape [num_nodes, in_channels].
            edge_index (torch.Tensor): Graph connectivity in COO format with shape [2, num_edges].
            edge_attr (torch.Tensor, optional): Edge attributes for weighted attention, if the GATConv
                                                layer is configured to use them (e.g., via `edge_dim`
                                                parameter in GATConv constructor). Shape [num_edges, edge_features].
                                                Defaults to None.

        Returns:
            torch.Tensor: Node embeddings of shape [num_nodes, out_channels].
        """
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            if i < self.num_layers - 1:  # Apply ELU and feature Dropout to all but the last layer
                x = F.elu(x)
                x = self.dropout(x) # This is the feature dropout between GAT layers
        return x


# Node Classification Components

class NodeClassifier(torch.nn.Module):
    """
    Node Classification Model.

    This model takes a GNN encoder, obtains node embeddings, and then passes these
    embeddings through a linear layer to produce classification logits.

    Args:
        encoder (torch.nn.Module): An instantiated GNN encoder (e.g., GCNEncoder, SAGEEncoder, GATEncoder).
                                   The encoder must have an `out_channels` attribute or its last
                                   convolutional layer (`encoder.convs[-1]`) must have one,
                                   indicating the dimension of embeddings it produces.
        num_classes (int): The number of classes for classification.
    """
    def __init__(self, encoder: torch.nn.Module, num_classes: int):
        super().__init__()
        self.encoder = encoder
        
        # Determine the output dimension of the provided encoder
        encoder_out_channels = None
        if hasattr(encoder, 'out_channels'): # Check if encoder itself has 'out_channels' (set in my GNNEncoders)
            encoder_out_channels = encoder.out_channels
        elif hasattr(encoder, 'convs') and encoder.convs and hasattr(encoder.convs[-1], 'out_channels'):
            # Fallback for encoders where out_channels might be on the last conv layer
            encoder_out_channels = encoder.convs[-1].out_channels
        
        if encoder_out_channels is None:
            raise ValueError("Could not determine out_channels from the provided encoder. Ensure encoder has an 'out_channels' attribute or its last layer does.")

        # Linear layer for classification: maps from embedding dimension to number of classes
        self.classifier_head = torch.nn.Linear(encoder_out_channels, num_classes)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass for node classification.

        Obtains node embeddings using the specified GNN encoder, then passes them
        through a linear classification head.

        Args:
            data (torch_geometric.data.Data): The input graph data, containing `data.x` (node features),
                                              `data.edge_index` (connectivity), and optionally
                                              `data.edge_attr` (edge attributes/weights).

        Returns:
            torch.Tensor: Logits for each node, shape [num_nodes, num_classes].
        Raises:
            NotImplementedError: If the provided encoder type is not GCNEncoder, GATEncoder, or SAGEEncoder.
        """
        # Get node embeddings from the encoder
        # data.edge_attr is assumed to be the edge weights/features if present
        edge_feature_to_pass = data.edge_attr if hasattr(data, 'edge_attr') else None

        if isinstance(self.encoder, GCNEncoder):
            # GCNEncoder expects 'edge_weight' argument for edge attributes
            embeddings = self.encoder(data.x, data.edge_index, edge_weight=edge_feature_to_pass)
        elif isinstance(self.encoder, GATEncoder):
            # GATEncoder expects 'edge_attr' argument for edge attributes
            embeddings = self.encoder(data.x, data.edge_index, edge_attr=edge_feature_to_pass)
        elif isinstance(self.encoder, SAGEEncoder):
            # SAGEEncoder's basic PyG form does not use edge attributes in its forward pass signature
            embeddings = self.encoder(data.x, data.edge_index)
        else:
            raise NotImplementedError(f"Encoder type {type(self.encoder).__name__} not specifically handled in NodeClassifier.")

        # Pass embeddings through the linear classification head
        logits = self.classifier_head(embeddings)
        return logits


def train_node_classifier(model: NodeClassifier, data: Data, 
                          target_label_name: str, 
                          optimizer: torch.optim.Optimizer, 
                          criterion: torch.nn.Module, 
                          n_epochs: int = 100,
                          pyg_data_object: Data = None): 
    """
    Trains a NodeClassifier model.

    The function assumes the `data` object contains:
    - Node features `data.x` and graph structure `data.edge_index`.
    - True labels as `data.y_<target_label_name>`.
    - A boolean mask `data.train_mask` indicating nodes to use for training.
    - Optionally, `data.val_mask` for validation during training.

    Args:
        model (NodeClassifier): The node classification model to train.
        data (Data): The PyG Data object containing graph, features, labels, and masks.
        target_label_name (str): The base name of the target label attribute on the `data` object
                                 (e.g., 'industry', which corresponds to `data.y_industry`).
        optimizer (torch.optim.Optimizer): The optimizer for training.
        criterion (torch.nn.Module): The loss function (e.g., CrossEntropyLoss).
        n_epochs (int, optional): Number of training epochs. Defaults to 100.
        pyg_data_object (Data, optional): The full PyG Data object, primarily for accessing
                                          label mappings if needed for more complex scenarios.
                                          Currently, `data` itself is expected to have necessary attributes.
                                          Defaults to None.

    Returns:
        NodeClassifier: The trained model.
    Raises:
        ValueError: If the target label attribute (e.g., `data.y_industry`) is not found in `data`.
    """
    # Retrieve the true labels for the target task
    y_true_attr_name = f'y_{target_label_name}'
    y_true = getattr(data, y_true_attr_name, None)
    if y_true is None:
        raise ValueError(f"Target label attribute '{y_true_attr_name}' not found in data object.")

    # Determine masks for training and validation
    train_mask = getattr(data, 'train_mask', None)
    val_mask = getattr(data, 'val_mask', None)

    if train_mask is None:
        print("Warning: 'train_mask' not found in data. Defaulting to train on all nodes.")
        # Create a default train_mask: all nodes.
        # Assumes labels are dense. If labels can be missing (e.g., NaN or -1 before encoding),
        # this should be more sophisticated (e.g., mask out nodes with invalid labels).
        train_mask = torch.ones(data.num_nodes, dtype=torch.bool, device=data.x.device)
    
    if not train_mask.any():
        print("Warning: train_mask is empty or all False. No nodes to train on. Returning untrained model.")
        return model 

    for epoch in range(n_epochs):
        model.train() # Set model to training mode
        optimizer.zero_grad() # Clear gradients
        
        logits = model(data) # Forward pass
        
        # Calculate loss only on the training nodes
        loss = criterion(logits[train_mask], y_true[train_mask])
        
        loss.backward() # Backpropagate
        optimizer.step() # Update weights
        
        # Print training progress (loss, and validation metrics if val_mask is available)
        log_message = f"Epoch {epoch+1}/{n_epochs}, Training Loss: {loss.item():.4f}"

        if val_mask is not None and val_mask.any():
            model.eval() # Set model to evaluation mode for validation
            with torch.no_grad(): # Disable gradient calculations for validation
                val_logits = model(data)
                val_loss = criterion(val_logits[val_mask], y_true[val_mask])
                predicted_labels_val = val_logits[val_mask].argmax(dim=1)
                # Ensure labels are on CPU for sklearn metrics
                val_acc = accuracy_score(y_true[val_mask].cpu().numpy(), predicted_labels_val.cpu().numpy())
                log_message += f", Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc:.4f}"
        
        print(log_message)

    return model


def evaluate_node_classifier(model: NodeClassifier, data: Data, 
                             target_label_name: str, 
                             test_mask_name: str = 'test_mask') -> float:
    """
    Evaluates the NodeClassifier model on a test set.

    Args:
        model (NodeClassifier): The trained node classification model.
        data (Data): The PyG Data object containing graph, features, labels, and the test mask.
        target_label_name (str): The base name of the target label attribute (e.g., 'industry').
        test_mask_name (str, optional): The name of the boolean mask attribute on `data`
                                        indicating nodes to use for testing. Defaults to 'test_mask'.

    Returns:
        float: The accuracy of the model on the specified test set. Returns `float('nan')`
               if the test mask is not found or is empty.
    Raises:
        ValueError: If the target label attribute (e.g., `data.y_industry`) is not found in `data`.
    """
    model.eval() # Set model to evaluation mode
    
    y_true_attr_name = f'y_{target_label_name}'
    y_true = getattr(data, y_true_attr_name, None)
    if y_true is None:
        raise ValueError(f"Target label attribute '{y_true_attr_name}' not found in data object.")

    test_mask = getattr(data, test_mask_name, None)
    if test_mask is None or not test_mask.any():
        print(f"Warning: Test mask '{test_mask_name}' not found in data or is empty. Cannot evaluate.")
        return float('nan')

    with torch.no_grad(): # Disable gradient calculations
        logits = model(data) # Forward pass
        
    # Get predictions for the test set
    predicted_labels = logits[test_mask].argmax(dim=1)
    
    # Ensure true labels and predicted labels are on CPU for scikit-learn metrics
    true_labels_test = y_true[test_mask].cpu().numpy()
    predicted_labels_test = predicted_labels.cpu().numpy()
    
    acc = accuracy_score(true_labels_test, predicted_labels_test)
    
    return acc


# Node Clustering Components

def get_node_embeddings(encoder_model: torch.nn.Module, data: Data) -> np.ndarray:
    """
    Generates node embeddings using a trained GNN encoder model.

    The encoder model is set to evaluation mode, and computations are performed
    without gradient tracking.

    Args:
        encoder_model (torch.nn.Module): The trained GNN encoder (e.g., GCNEncoder, SAGEEncoder, GATEncoder).
                                         This could also be the encoder part of a trained GNNAutoencoder.
        data (Data): The PyTorch Geometric Data object containing node features (`data.x`),
                     graph structure (`data.edge_index`), and optionally `data.edge_attr`.

    Returns:
        np.ndarray: A NumPy array of node embeddings, shape [num_nodes, embedding_dim].
    Raises:
        TypeError: If the provided `encoder_model` type is not GCNEncoder, GATEncoder, or SAGEEncoder.
    """
    encoder_model.eval() # Set the encoder to evaluation mode
    with torch.no_grad(): # Disable gradient calculations
        edge_feature_to_pass = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        # Call the appropriate forward method based on encoder type
        if isinstance(encoder_model, GCNEncoder):
            embeddings = encoder_model(data.x, data.edge_index, edge_weight=edge_feature_to_pass)
        elif isinstance(encoder_model, GATEncoder):
            embeddings = encoder_model(data.x, data.edge_index, edge_attr=edge_feature_to_pass)
        elif isinstance(encoder_model, SAGEEncoder):
            embeddings = encoder_model(data.x, data.edge_index)
        else:
            raise TypeError(f"Unknown or unhandled encoder type: {type(encoder_model).__name__} passed to get_node_embeddings.")
            
    # Detach embeddings from computation graph and move to CPU, then convert to NumPy array
    return embeddings.detach().cpu().numpy()


def cluster_nodes_kmeans(embeddings: np.ndarray, n_clusters: int, random_state: int = 42) -> tuple[np.ndarray, float | None]:
    """
    Clusters node embeddings using K-Means and calculates the silhouette score.

    Args:
        embeddings (np.ndarray): A NumPy array of node embeddings, shape [n_samples, n_features].
        n_clusters (int): The desired number of clusters for K-Means.
        random_state (int, optional): Seed for KMeans reproducibility. Defaults to 42.

    Returns:
        tuple:
            - cluster_labels (np.ndarray): Cluster labels assigned by KMeans for each node.
                                           Returns an empty array if KMeans fitting fails.
            - silhouette_avg (float | None): The silhouette score. Returns `None` if the score cannot be computed
                                             (e.g., `n_clusters <= 1`, `n_clusters >= n_samples`,
                                             KMeans fails, or all samples are assigned to a single cluster).
    Raises:
        TypeError: If `embeddings` is not a NumPy array.
        ValueError: If `embeddings` is not 2D, or `n_clusters` is not positive.
    """
    if not isinstance(embeddings, np.ndarray):
        raise TypeError("Embeddings must be a NumPy array.")
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array (n_samples, n_features).")
    if n_clusters <= 0:
        raise ValueError("Number of clusters (n_clusters) must be greater than 0.")

    # Initialize KMeans with n_init='auto' to suppress future warnings and use optimal number of initializations.
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    
    try:
        cluster_labels = kmeans.fit_predict(embeddings)
    except Exception as e:
        print(f"KMeans fitting failed: {e}")
        # Return empty labels and None score if KMeans fails.
        return np.array([]), None

    silhouette_avg = None
    # Silhouette score is only defined if number of labels is 2 <= n_labels <= n_samples - 1
    # Also check if the number of unique clusters found is appropriate for silhouette score.
    num_unique_labels = len(np.unique(cluster_labels))

    if 1 < num_unique_labels < embeddings.shape[0] and 1 < n_clusters < embeddings.shape[0]:
        try:
            silhouette_avg = silhouette_score(embeddings, cluster_labels)
        except ValueError as e:
            # This can happen if, e.g., KMeans results in a single cluster despite n_clusters > 1
            print(f"Could not compute silhouette score (n_clusters={n_clusters}, n_samples={embeddings.shape[0]}, unique_labels={num_unique_labels}): {e}")
            silhouette_avg = None 
    else:
        # Print informative message if silhouette score is not computed due to constraints.
        reason = ""
        if num_unique_labels <= 1: reason += f"found {num_unique_labels} unique cluster(s) "
        if n_clusters <= 1 : reason += f"n_clusters ({n_clusters}) <= 1 "
        if n_clusters >= embeddings.shape[0]: reason += f"n_clusters ({n_clusters}) >= n_samples ({embeddings.shape[0]}) "
        
        print(f"Silhouette score not computed or invalid: {reason.strip()}.")

    return cluster_labels, silhouette_avg


# GNN Autoencoder for Anomaly Detection

class GNNAutoencoder(torch.nn.Module):
    """
    Graph Neural Network Autoencoder.

    This model uses a GNN encoder to compress node features into embeddings,
    and a GNN-based decoder to reconstruct the original node features from these embeddings.
    The type of GNN layers used in the decoder mirrors those of the provided encoder.

    Args:
        encoder (torch.nn.Module): An instantiated GNN encoder (e.g., GCNEncoder, SAGEEncoder, GATEncoder).
                                   Its `out_channels` attribute (or that of its last conv layer)
                                   determines the latent embedding dimension.
        decoder_hidden_dims (list[int]): A list of integers specifying the number of hidden units
                                         in each intermediate layer of the decoder.
        reconstructed_feature_dim (int): The dimension of the original node features that the
                                         autoencoder aims to reconstruct. This should match
                                         `encoder.in_channels` (or `data.x.shape[1]`).
        dropout_rate (float, optional): Dropout probability for decoder layers. Defaults to 0.5.
    """
    def __init__(self, encoder: torch.nn.Module, decoder_hidden_dims: list, 
                 reconstructed_feature_dim: int, dropout_rate: float = 0.5):
        super().__init__()
        self.encoder = encoder

        # Determine encoder's output embedding dimension (latent space dimension)
        encoder_embedding_dim = None
        if hasattr(encoder, 'out_channels'):
            encoder_embedding_dim = encoder.out_channels
        elif hasattr(encoder, 'convs') and encoder.convs and hasattr(encoder.convs[-1], 'out_channels'):
            encoder_embedding_dim = encoder.convs[-1].out_channels
        
        if encoder_embedding_dim is None:
            raise ValueError("Could not determine embedding dimension from the provided encoder for GNNAutoencoder.")

        self.decoder_layers = torch.nn.ModuleList()
        self.decoder_dropout_layers = torch.nn.ModuleList()
        
        current_dim = encoder_embedding_dim # Input dimension for the first decoder layer
        # Define the sequence of output dimensions for decoder layers
        all_decoder_output_dims = list(decoder_hidden_dims) + [reconstructed_feature_dim]

        for i, target_dim in enumerate(all_decoder_output_dims):
            is_last_layer = (i == len(all_decoder_output_dims) - 1)
            
            # Choose decoder layer type based on the encoder type
            if isinstance(self.encoder, (GCNEncoder, SAGEEncoder)):
                conv_class = GCNConv if isinstance(self.encoder, GCNEncoder) else SAGEConv
                conv = conv_class(current_dim, target_dim)
            elif isinstance(self.encoder, GATEncoder):
                # For GAT decoder layers, use heads=1 and concat=False for simplicity and direct dimension mapping.
                # GATConv's internal dropout applies to attention; self.decoder_dropout_layers applies to features.
                gat_layer_dropout = dropout_rate if not is_last_layer else 0.0 # No GAT internal dropout on last layer
                conv = GATConv(current_dim, target_dim, heads=1, concat=False, dropout=gat_layer_dropout)
            else:
                raise ValueError(f"Unsupported encoder type for autoencoder decoder: {type(self.encoder).__name__}")
            
            self.decoder_layers.append(conv)
            
            if not is_last_layer: # No dropout after the final output layer
                self.decoder_dropout_layers.append(torch.nn.Dropout(dropout_rate))
            
            current_dim = target_dim # Update current_dim for the next decoder layer

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass of the GNN Autoencoder.

        Encodes node features to a latent representation, then decodes them back
        to reconstruct the original node features.

        Args:
            data (torch_geometric.data.Data): Input graph data, must contain `data.x`, `data.edge_index`,
                                              and optionally `data.edge_attr`.

        Returns:
            torch.Tensor: Reconstructed node features, shape [num_nodes, reconstructed_feature_dim].
        Raises:
            NotImplementedError: If the encoder type is not GCNEncoder, GATEncoder, or SAGEEncoder.
        """
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

        # 1. Encode: Get latent node embeddings (z)
        if isinstance(self.encoder, GCNEncoder):
            z = self.encoder(x, edge_index, edge_weight=edge_attr)
        elif isinstance(self.encoder, GATEncoder):
            z = self.encoder(x, edge_index, edge_attr=edge_attr)
        elif isinstance(self.encoder, SAGEEncoder):
            z = self.encoder(x, edge_index)
        else:
            raise NotImplementedError(f"Encoding for {type(self.encoder).__name__} not implemented in GNNAutoencoder.")

        # 2. Decode: Reconstruct features from latent embeddings
        # The variable 'z' will be transformed by each decoder layer
        for i, layer in enumerate(self.decoder_layers):
            is_last_layer = (i == len(self.decoder_layers) - 1)
            
            # Pass through the GNN convolutional layer
            if isinstance(self.encoder, GCNEncoder): # Assuming decoder mirrors encoder type for conv
                z = layer(z, edge_index, edge_weight=edge_attr) 
            elif isinstance(self.encoder, SAGEEncoder):
                z = layer(z, edge_index) 
            elif isinstance(self.encoder, GATEncoder):
                z = layer(z, edge_index, edge_attr=edge_attr)
            
            if not is_last_layer:
                # Apply activation function
                if isinstance(self.encoder, GATEncoder): # GAT typically uses ELU
                    z = F.elu(z)
                else: # GCN, SAGE typically use ReLU
                    z = F.relu(z)
                # Apply dropout (feature dropout)
                z = self.decoder_dropout_layers[i](z)
            # No activation or dropout on the final output layer for reconstruction
            
        return z # This is x_reconstructed


def train_gnn_autoencoder(model: GNNAutoencoder, data: Data, 
                          optimizer: torch.optim.Optimizer, 
                          criterion: torch.nn.Module, # e.g., torch.nn.MSELoss()
                          n_epochs: int = 100):
    """
    Trains a GNNAutoencoder model.

    The goal is to minimize the reconstruction error between original node features
    and features reconstructed by the autoencoder.

    Args:
        model (GNNAutoencoder): The GNN autoencoder model to train.
        data (Data): The PyG Data object containing graph, features (`data.x`),
                     and optionally `data.train_mask`.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        criterion (torch.nn.Module): The loss function, typically Mean Squared Error (MSE)
                                     for reconstruction tasks (e.g., `torch.nn.MSELoss()`).
        n_epochs (int, optional): Number of training epochs. Defaults to 100.

    Returns:
        GNNAutoencoder: The trained model.
    """
    # Use train_mask if available, otherwise train on all nodes (common for unsupervised autoencoders)
    train_mask = getattr(data, 'train_mask', None)
    if train_mask is None:
        print("Warning: 'train_mask' not found in data. Training autoencoder on all nodes.")
        # Ensure mask is on the same device as features if creating a default one
        train_mask = torch.ones(data.num_nodes, dtype=torch.bool, device=data.x.device) 
    
    if not train_mask.any():
        print("Warning: train_mask is empty or all False. No nodes to train autoencoder on. Returning untrained model.")
        return model

    for epoch in range(n_epochs):
        model.train() # Set model to training mode
        optimizer.zero_grad() # Clear gradients
        
        x_reconstructed = model(data) # Forward pass to get reconstructed features
        
        # Calculate loss only on the training nodes (or all nodes if no mask)
        # Ensure target (data.x) and input (x_reconstructed) are aligned if using mask
        loss = criterion(x_reconstructed[train_mask], data.x[train_mask])
        
        loss.backward() # Backpropagate
        optimizer.step() # Update weights
        
        # Print training progress periodically
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch+1}/{n_epochs}, Autoencoder Training Loss: {loss.item():.4f}")
            
    return model


def get_reconstruction_errors(model: GNNAutoencoder, data: Data) -> np.ndarray:
    """
    Calculates per-node reconstruction errors using a trained GNNAutoencoder.

    The reconstruction error for each node is typically the Mean Squared Error (MSE)
    between its original features and its reconstructed features.

    Args:
        model (GNNAutoencoder): The trained GNN autoencoder model.
        data (Data): The PyG Data object containing original node features (`data.x`)
                     and graph structure.

    Returns:
        np.ndarray: A NumPy array of per-node reconstruction errors.
    Raises:
        ValueError: If the shape of original features and reconstructed features mismatch.
    """
    model.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculations
        x_reconstructed = model(data) # Get reconstructed features
    
    if data.x.shape != x_reconstructed.shape:
        raise ValueError(f"Original features shape {data.x.shape} and reconstructed features shape {x_reconstructed.shape} mismatch. Cannot calculate errors.")

    # Calculate Mean Squared Error per node (sum of squared differences across features, then mean)
    # errors = torch.sum((data.x - x_reconstructed)**2, dim=1) # Sum of squared errors
    errors = torch.mean((data.x - x_reconstructed)**2, dim=1) # Mean of squared errors
    
    return errors.detach().cpu().numpy()


def get_anomalies_by_error(reconstruction_errors: np.ndarray, 
                           node_ids: list,
                           top_n: int = 10) -> list[tuple[any, float]]:
    """
    Identifies the top_n nodes with the highest reconstruction errors.

    Args:
        reconstruction_errors (np.ndarray): A NumPy array of per-node reconstruction errors.
                                            Order must correspond to `node_ids`.
        node_ids (list): List of original node identifiers (e.g., establishment_ids).
                         Order must correspond to `reconstruction_errors`.
        top_n (int, optional): Number of top anomalies to return. Defaults to 10.

    Returns:
        list[tuple[any, float]]: A list of tuples, where each tuple is (node_id, error_score),
                                 sorted by error_score in descending order, containing the top_n anomalies.
    Raises:
        ValueError: If the length of `reconstruction_errors` and `node_ids` do not match.
    """
    if len(reconstruction_errors) != len(node_ids):
        raise ValueError("Length of reconstruction_errors and node_ids must match.")

    # Combine node_ids with their reconstruction errors
    node_errors = list(zip(node_ids, reconstruction_errors))
    
    # Sort by error in descending order (highest error first)
    node_errors.sort(key=lambda item: item[1], reverse=True)
    
    # Return the top_n anomalies
    return node_errors[:top_n]
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import to_networkx, from_networkx, negative_sampling # Add other PyG utils as needed

import networkx as nx
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, accuracy_score, silhouette_score
# Add other sklearn utilities as needed, e.g., for anomaly detection or other metrics

# Standard Python libraries
import random
from collections import Counter

# Assuming data.py is in the same directory or accessible via PYTHONPATH
from data import load_graph_data


def load_establishment_data(nodes_filepath="epfo_establishments_nodes.csv", edges_filepath="epfo_transfers_edges.csv"):
    """
    Loads graph data (NetworkX graph, nodes DataFrame, edges DataFrame) from specified CSV files.

    Args:
        nodes_filepath (str): Path to the CSV file containing node information.
        edges_filepath (str): Path to the CSV file containing edge information.

    Returns:
        tuple: nx_graph (networkx.Graph), nodes_df (pd.DataFrame), edges_df (pd.DataFrame)
    """
    nx_graph, nodes_df, edges_df = load_graph_data(nodes_filepath, edges_filepath)
    return nx_graph, nodes_df, edges_df


def preprocess_node_features(nodes_df):
    """
    Preprocesses node features by applying one-hot encoding to categorical attributes.

    Args:
        nodes_df (pd.DataFrame): DataFrame containing node information, including 'establishment_id'.
                                 Expected to have 'industry', 'city', 'size_category' columns.

    Returns:
        tuple: 
            - features_df (pd.DataFrame): DataFrame with 'establishment_id' and one-hot encoded features.
            - feature_cols (list): List of strings representing the names of the feature columns.
    """
    if 'establishment_id' not in nodes_df.columns:
        raise ValueError("nodes_df must contain 'establishment_id' column.")

    # Set establishment_id as index for easy mapping later, and to avoid it being processed as a feature
    nodes_df_indexed = nodes_df.set_index('establishment_id')

    categorical_features = ['industry', 'city', 'size_category']
    
    # Ensure all categorical features are present
    missing_cols = [col for col in categorical_features if col not in nodes_df_indexed.columns]
    if missing_cols:
        raise ValueError(f"Missing categorical columns in nodes_df: {missing_cols}")

    # Apply one-hot encoding
    features_df = pd.get_dummies(nodes_df_indexed, columns=categorical_features, prefix=categorical_features)
    
    # Get the names of the new feature columns
    # Exclude original columns that were not one-hot encoded if any remain,
    # though in this setup, all except 'establishment_id' (which is index) are either categorical or dropped.
    # If there were other numerical features, they would be preserved.
    # For now, we assume only categorical are present besides id.
    
    # Reset index so 'establishment_id' becomes a column again
    features_df = features_df.reset_index()
    
    # The feature columns are all columns except 'establishment_id'
    feature_cols = [col for col in features_df.columns if col != 'establishment_id']
    
    # If there were other non-categorical features in the original nodes_df_indexed that should be features,
    # they would already be in features_df.
    # For example, if 'age' was a numerical column, it would be carried over.
    # feature_cols would then be pd.get_dummies_columns + original_numerical_columns.

    return features_df, feature_cols


def create_pyg_data_object(nx_graph, processed_node_features_df, original_nodes_df, feature_cols):
    """
    Creates a PyTorch Geometric Data object from a NetworkX graph and processed node features.

    Args:
        nx_graph (networkx.Graph): The input graph.
        processed_node_features_df (pd.DataFrame): DataFrame with 'establishment_id' and one-hot encoded features.
        original_nodes_df (pd.DataFrame): The original nodes DataFrame, used to extract labels.
                                          Must contain 'establishment_id', 'industry', 'city', 'size_category'.
        feature_cols (list): List of column names that constitute the features in processed_node_features_df.

    Returns:
        torch_geometric.data.Data: A PyG Data object.
        dict: node_id_map ({establishment_id: integer_index})
    """
    if 'establishment_id' not in processed_node_features_df.columns:
        raise ValueError("processed_node_features_df must contain 'establishment_id' column.")
    if 'establishment_id' not in original_nodes_df.columns:
        raise ValueError("original_nodes_df must contain 'establishment_id' column.")

    # Create a mapping from establishment_id to a continuous integer index (0 to N-1)
    # Important: Ensure all nodes in nx_graph are present in processed_node_features_df and original_nodes_df
    # For simplicity, we assume nx_graph.nodes() are the establishment_ids.
    # If nx_graph might have nodes not in dataframes, filtering or error handling is needed.
    
    graph_nodes = list(nx_graph.nodes())
    node_id_map = {node_id: i for i, node_id in enumerate(graph_nodes)}
    
    num_nodes = len(graph_nodes)

    # Align processed_node_features_df with the graph node order
    # Set 'establishment_id' as index for quick lookup
    features_df_indexed = processed_node_features_df.set_index('establishment_id')
    # Reindex based on graph_nodes order and select feature_cols
    # If an establishment_id from graph_nodes is not in features_df_indexed, it will result in NaNs.
    # This needs careful handling, e.g., ensuring all graph nodes have features.
    # For now, assume all nodes in nx_graph are in processed_node_features_df.
    aligned_features = features_df_indexed.reindex(graph_nodes)[feature_cols]
    
    # Check for missing features after alignment (nodes in graph but not in features_df)
    if aligned_features.isnull().values.any():
        missing_nodes = aligned_features[aligned_features.isnull().any(axis=1)].index.tolist()
        # print(f"Warning: Nodes {missing_nodes} are in the graph but missing features. Filling with zeros.")
        # Option: fill NaNs, or raise error, or ensure data integrity upstream.
        # For now, let's fill with zeros, though this might not be ideal for all cases.
        aligned_features = aligned_features.fillna(0) 


    x = torch.tensor(aligned_features.values, dtype=torch.float)

    # Create edge_index
    # Edges in nx_graph can be (u, v) or (u, v, data_dict)
    # We need to map u and v to their integer indices
    edge_list = []
    edge_attributes = []
    for u, v, data in nx_graph.edges(data=True):
        if u in node_id_map and v in node_id_map: # Ensure both nodes are in the map
            edge_list.append((node_id_map[u], node_id_map[v]))
            # Assuming 'weight' attribute exists for edges, as per data.py
            edge_attributes.append(data.get('weight', 1.0)) # Default to 1.0 if no weight
        # else:
            # print(f"Warning: Edge ({u}, {v}) contains nodes not in node_id_map. Skipping this edge.")


    if not edge_list:
        # Handle case with no edges or all edges skipped
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0,1), dtype=torch.float) # Assuming edge_attr is 1D per edge
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attributes, dtype=torch.float).unsqueeze(1) # Make it [num_edges, 1]

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # Store original labels
    # Align original_nodes_df with the graph node order
    original_nodes_df_indexed = original_nodes_df.set_index('establishment_id')
    aligned_original_nodes = original_nodes_df_indexed.reindex(graph_nodes)

    label_cols = ['industry', 'city', 'size_category']
    for col in label_cols:
        if col in aligned_original_nodes:
            # Convert string labels to numerical representation (e.g., using LabelEncoder)
            # then to tensor. For PyG, these are often stored directly on the data object.
            le = LabelEncoder()
            # Handle potential NaNs in label columns before encoding
            labels = aligned_original_nodes[col].fillna('Unknown').astype(str)
            encoded_labels = le.fit_transform(labels)
            setattr(data, f'y_{col}', torch.tensor(encoded_labels, dtype=torch.long))
            # Store the mapping from encoded label to original string label
            setattr(data, f'y_{col}_mapping', {i: cls for i, cls in enumerate(le.classes_)})
        # else:
            # print(f"Warning: Label column '{col}' not found in original_nodes_df.")


    # Add node_id_map to the data object for easy reference
    data.node_id_map = node_id_map
    # Also store the reverse map if needed
    data.idx_to_node_id = {v: k for k, v in node_id_map.items()}
    
    # Store establishment_ids in their mapped order
    data.establishment_ids = graph_nodes


    return data, node_id_map


# GNN Encoder Architectures

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout_rate=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("Number of layers must be at least 1.")

        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels))
        else:
            # First layer
            self.convs.append(GCNConv(in_channels, hidden_channels))
            # Intermediate layers
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            # Final layer
            self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index, edge_weight=None):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight=edge_weight)
            if i < self.num_layers - 1:  # Apply ReLU and Dropout to all but the last layer
                x = F.relu(x)
                x = self.dropout(x)
        return x


class SAGEEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout_rate=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("Number of layers must be at least 1.")

        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, out_channels))
        else:
            # First layer
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            # Intermediate layers
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            # Final layer
            self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i < self.num_layers - 1:  # Apply ReLU and Dropout to all but the last layer
                x = F.relu(x)
                x = self.dropout(x)
        return x


class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout_rate=0.5, heads=8):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        # Dropout for features between layers. GATConv has its own internal dropout for attention weights.
        self.dropout = torch.nn.Dropout(dropout_rate) 
        self.num_layers = num_layers
        self.heads = heads

        if num_layers < 1:
            raise ValueError("Number of layers must be at least 1.")

        current_in_channels = in_channels

        if num_layers == 1:
            # Final layer directly maps to out_channels, no intermediate hidden_channels * heads
            # For the final layer in GAT, it's common to use heads=1 and concat=False
            # The GATConv dropout is for attention coefficients, self.dropout is for features.
            self.convs.append(GATConv(current_in_channels, out_channels, heads=1, concat=False, dropout=dropout_rate))
        else:
            # First layer
            # The GATConv dropout is for attention coefficients.
            self.convs.append(GATConv(current_in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout_rate))
            current_in_channels = hidden_channels * heads # Output of first layer

            # Intermediate layers
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(current_in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout_rate))
                current_in_channels = hidden_channels * heads
            
            # Final layer
            # The GATConv dropout is for attention coefficients.
            self.convs.append(GATConv(current_in_channels, out_channels, heads=1, concat=False, dropout=dropout_rate))


    def forward(self, x, edge_index, edge_attr=None): # GATConv uses edge_attr for weighted attention
        for i in range(self.num_layers):
            # Pass edge_attr to GATConv layers
            x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            if i < self.num_layers - 1:  # Apply ELU and Dropout to all but the last layer
                x = F.elu(x)
                x = self.dropout(x) # This is the feature dropout between layers
        return x


# Node Classification Components

class NodeClassifier(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, num_classes: int):
        super().__init__()
        self.encoder = encoder
        
        # Ensure the encoder has an out_channels attribute (from its own GNN layers)
        if not hasattr(encoder, 'convs') or not encoder.convs: # Basic check, assumes convs list
             raise ValueError("Encoder does not seem to have layers or out_channels defined properly.")
        
        # Determine encoder's output channels.
        # This depends on how out_channels is stored in the encoder.
        # Assuming the last conv layer in encoder.convs has an 'out_channels' attribute
        # or the encoder itself has an 'out_channels' attribute set.
        encoder_out_channels = None
        if hasattr(encoder, 'out_channels'): # If encoder itself stores it
            encoder_out_channels = encoder.out_channels
        elif hasattr(encoder.convs[-1], 'out_channels'): # Last layer of encoder
            encoder_out_channels = encoder.convs[-1].out_channels
        else: # Fallback for GAT which might have heads and concat affecting final size before last layer
             # For GAT, the final layer is GATConv(..., out_channels, heads=1, concat=False, ...)
             # So its out_channels is the direct output feature size.
             # For other encoders, it's simpler.
             # This logic might need refinement based on actual encoder attribute names.
            if isinstance(encoder, GATEncoder):
                 # The GATEncoder's last conv layer (self.convs[-1]) is designed to output `out_channels`
                 # as specified in its constructor.
                 encoder_out_channels = encoder.convs[-1].out_channels
            else: # Default for GCN, SAGE
                 encoder_out_channels = encoder.convs[-1].out_channels


        if encoder_out_channels is None:
            raise ValueError("Could not determine out_channels from the provided encoder.")

        self.classifier_head = torch.nn.Linear(encoder_out_channels, num_classes)

    def forward(self, data: Data) -> torch.Tensor:
        # Get node embeddings
        # data.edge_attr is assumed to be the edge weights/features
        edge_feature_to_pass = data.edge_attr if hasattr(data, 'edge_attr') else None

        if isinstance(self.encoder, GCNEncoder):
            # GCNEncoder expects 'edge_weight'
            embeddings = self.encoder(data.x, data.edge_index, edge_weight=edge_feature_to_pass)
        elif isinstance(self.encoder, GATEncoder):
            # GATEncoder expects 'edge_attr'
            embeddings = self.encoder(data.x, data.edge_index, edge_attr=edge_feature_to_pass)
        elif isinstance(self.encoder, SAGEEncoder):
            # SAGEEncoder's basic form does not use edge attributes in its forward pass
            embeddings = self.encoder(data.x, data.edge_index)
        else:
            # Fallback or error for unknown/unhandled encoder type
            # Or try a generic call if the encoder has a standard signature (x, edge_index)
            # For now, raise an error if specific handling isn't defined.
            raise NotImplementedError(f"Encoder type {type(self.encoder).__name__} not specifically handled in NodeClassifier.")

        # Pass these embeddings through the classification head
        logits = self.classifier_head(embeddings)
        return logits


def train_node_classifier(model: NodeClassifier, data: Data, 
                          target_label_name: str, 
                          optimizer: torch.optim.Optimizer, 
                          criterion: torch.nn.Module, 
                          n_epochs: int = 100,
                          pyg_data_object: Data = None): # pyg_data_object for potential future use, not strictly needed now
    """
    Trains a NodeClassifier model.
    Assumes data object has 'train_mask' and optionally 'val_mask'.
    If not, trains on all nodes with available labels.
    """
    y_true = getattr(data, f'y_{target_label_name}', None)
    if y_true is None:
        raise ValueError(f"Target label 'y_{target_label_name}' not found in data object.")

    # Determine masks
    train_mask = getattr(data, 'train_mask', None)
    val_mask = getattr(data, 'val_mask', None)

    if train_mask is None:
        print("Warning: 'train_mask' not found in data. Training on all nodes.")
        # Create a default train_mask: all nodes that have a valid label
        # This assumes labels are dense. If labels can be missing (e.g. NaN or -1 before encoding),
        # this should be more sophisticated. Given labels are already encoded, we assume all are valid.
        train_mask = torch.ones(data.num_nodes, dtype=torch.bool)
    
    if not train_mask.any():
        print("Warning: train_mask is empty or all False. No nodes to train on.")
        return model # Or raise error

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        
        logits = model(data)
        
        loss = criterion(logits[train_mask], y_true[train_mask])
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{n_epochs}, Training Loss: {loss.item():.4f}", end="")

        if val_mask is not None and val_mask.any():
            model.eval()
            with torch.no_grad():
                val_logits = model(data)
                val_loss = criterion(val_logits[val_mask], y_true[val_mask])
                predicted_labels_val = val_logits[val_mask].argmax(dim=1)
                val_acc = accuracy_score(y_true[val_mask].cpu().numpy(), predicted_labels_val.cpu().numpy())
                print(f", Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc:.4f}")
        else:
            print() # Newline if no validation

    return model


def evaluate_node_classifier(model: NodeClassifier, data: Data, 
                             target_label_name: str, 
                             test_mask_name: str = 'test_mask') -> float:
    """
    Evaluates the NodeClassifier model on a test set.
    """
    model.eval()
    
    y_true = getattr(data, f'y_{target_label_name}', None)
    if y_true is None:
        raise ValueError(f"Target label 'y_{target_label_name}' not found in data object.")

    test_mask = getattr(data, test_mask_name, None)
    if test_mask is None or not test_mask.any():
        print(f"Warning: '{test_mask_name}' not found in data or is empty. Cannot evaluate.")
        return float('nan')

    with torch.no_grad():
        logits = model(data)
        
    predicted_labels = logits[test_mask].argmax(dim=1)
    
    # Ensure y_true for the test set is on CPU for sklearn metrics
    true_labels_test = y_true[test_mask].cpu().numpy()
    predicted_labels_test = predicted_labels.cpu().numpy()
    
    acc = accuracy_score(true_labels_test, predicted_labels_test)
    
    return acc


# Node Clustering Components

def get_node_embeddings(encoder_model: torch.nn.Module, data: Data) -> np.ndarray:
    """
    Generates node embeddings using a trained GNN encoder model.

    Args:
        encoder_model (torch.nn.Module): The trained GNN encoder (e.g., GCNEncoder, SAGEEncoder, GATEncoder).
        data (Data): The PyTorch Geometric Data object containing node features (data.x) 
                     and graph structure (data.edge_index, data.edge_attr).

    Returns:
        np.ndarray: A NumPy array of node embeddings.
    """
    encoder_model.eval()
    with torch.no_grad():
        edge_feature_to_pass = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        if isinstance(encoder_model, GCNEncoder):
            embeddings = encoder_model(data.x, data.edge_index, edge_weight=edge_feature_to_pass)
        elif isinstance(encoder_model, GATEncoder):
            embeddings = encoder_model(data.x, data.edge_index, edge_attr=edge_feature_to_pass)
        elif isinstance(encoder_model, SAGEEncoder):
            embeddings = encoder_model(data.x, data.edge_index)
        else:
            raise TypeError(f"Unknown or unhandled encoder type: {type(encoder_model).__name__} passed to get_node_embeddings.")
            
    return embeddings.detach().cpu().numpy()


def cluster_nodes_kmeans(embeddings: np.ndarray, n_clusters: int, random_state: int = 42) -> tuple[np.ndarray, float | None]:
    """
    Clusters node embeddings using K-Means and calculates the silhouette score.

    Args:
        embeddings (np.ndarray): A NumPy array of node embeddings.
        n_clusters (int): The desired number of clusters.
        random_state (int, optional): Seed for KMeans reproducibility. Defaults to 42.

    Returns:
        tuple:
            - np.ndarray: Cluster labels assigned by KMeans.
            - float | None: The silhouette score. Returns None if the score cannot be computed 
                            (e.g., n_clusters <= 1 or n_clusters >= number of samples, or if KMeans fails).
    """
    if not isinstance(embeddings, np.ndarray):
        raise TypeError("Embeddings must be a NumPy array.")
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array (n_samples, n_features).")
    if n_clusters <= 0:
        raise ValueError("Number of clusters (n_clusters) must be greater than 0.")


    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    
    try:
        cluster_labels = kmeans.fit_predict(embeddings)
    except Exception as e:
        print(f"KMeans fitting failed: {e}")
        # Depending on how critical this is, one might re-raise or return specific error indicators
        # For now, return empty labels and None score if KMeans fails.
        return np.array([]), None


    silhouette_avg = None
    # Silhouette score is only defined if number of labels is 2 <= n_labels <= n_samples - 1
    if 1 < n_clusters < embeddings.shape[0]:
        try:
            silhouette_avg = silhouette_score(embeddings, cluster_labels)
        except ValueError as e:
            # This can happen if, e.g., KMeans results in a single cluster
            # despite n_clusters > 1 (e.g. due to degenerate data or low n_samples)
            print(f"Could not compute silhouette score (n_clusters={n_clusters}, n_samples={embeddings.shape[0]}): {e}")
            silhouette_avg = None 
            # Note: If cluster_labels contains only one unique value, silhouette_score raises ValueError.
            # We check this explicitly:
            if len(np.unique(cluster_labels)) < 2 :
                 print(f"Silhouette score cannot be calculated with less than 2 unique clusters. Found {len(np.unique(cluster_labels))} unique cluster(s).")

    elif n_clusters <= 1 or n_clusters >= embeddings.shape[0]:
        print(f"Silhouette score not computed: n_clusters ({n_clusters}) must be > 1 and < n_samples ({embeddings.shape[0]}).")

    return cluster_labels, silhouette_avg


# GNN Autoencoder for Anomaly Detection

class GNNAutoencoder(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, decoder_hidden_dims: list, 
                 reconstructed_feature_dim: int, dropout_rate: float = 0.5):
        super().__init__()
        self.encoder = encoder

        # Determine encoder's output embedding dimension
        encoder_embedding_dim = None
        if hasattr(encoder, 'out_channels'): # If encoder itself stores it (e.g. a custom attribute)
            encoder_embedding_dim = encoder.out_channels
        elif hasattr(encoder.convs[-1], 'out_channels'): # Last GCN/SAGE layer
            encoder_embedding_dim = encoder.convs[-1].out_channels
        elif isinstance(encoder, GATEncoder): # GATEncoder's last layer is specific
             encoder_embedding_dim = encoder.convs[-1].out_channels
        
        if encoder_embedding_dim is None:
            raise ValueError("Could not determine embedding dimension from the provided encoder.")

        self.decoder_layers = torch.nn.ModuleList()
        self.decoder_dropout_layers = torch.nn.ModuleList()
        
        current_dim = encoder_embedding_dim
        all_decoder_output_dims = list(decoder_hidden_dims) + [reconstructed_feature_dim]

        for i, target_dim in enumerate(all_decoder_output_dims):
            is_last_layer = (i == len(all_decoder_output_dims) - 1)
            
            if isinstance(self.encoder, (GCNEncoder, SAGEEncoder)): # Assuming GCN and SAGE decoders are similar
                conv_class = GCNConv if isinstance(self.encoder, GCNEncoder) else SAGEConv
                conv = conv_class(current_dim, target_dim)
            elif isinstance(self.encoder, GATEncoder):
                # For GAT decoder layers, use heads=1 and concat=False for simplicity.
                # The GATConv's internal dropout can be used.
                conv = GATConv(current_dim, target_dim, heads=1, concat=False, dropout=dropout_rate if not is_last_layer else 0.0)
            else:
                raise ValueError(f"Unsupported encoder type for autoencoder decoder: {type(self.encoder).__name__}")
            
            self.decoder_layers.append(conv)
            
            if not is_last_layer:
                self.decoder_dropout_layers.append(torch.nn.Dropout(dropout_rate))
            
            current_dim = target_dim

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

        # Encode
        if isinstance(self.encoder, GCNEncoder):
            z = self.encoder(x, edge_index, edge_weight=edge_attr)
        elif isinstance(self.encoder, GATEncoder):
            z = self.encoder(x, edge_index, edge_attr=edge_attr)
        elif isinstance(self.encoder, SAGEEncoder):
            z = self.encoder(x, edge_index)
        else:
            raise NotImplementedError(f"Encoding for {type(self.encoder).__name__} not implemented in GNNAutoencoder.")

        # Decode
        for i, layer in enumerate(self.decoder_layers):
            is_last_layer = (i == len(self.decoder_layers) - 1)
            
            if isinstance(self.encoder, GCNEncoder):
                z = layer(z, edge_index, edge_weight=edge_attr) # GCNConv might use edge_weight
            elif isinstance(self.encoder, SAGEEncoder):
                z = layer(z, edge_index) # SAGEConv typically doesn't use edge_attr in basic form
            elif isinstance(self.encoder, GATEncoder):
                z = layer(z, edge_index, edge_attr=edge_attr) # GATConv uses edge_attr
            
            if not is_last_layer:
                if isinstance(self.encoder, GATEncoder):
                    z = F.elu(z)
                else: # GCN, SAGE
                    z = F.relu(z)
                z = self.decoder_dropout_layers[i](z)
            # No activation or dropout on the final output layer for reconstruction
            
        return z


def train_gnn_autoencoder(model: GNNAutoencoder, data: Data, 
                          optimizer: torch.optim.Optimizer, 
                          criterion: torch.nn.Module, # e.g., torch.nn.MSELoss()
                          n_epochs: int = 100):
    """
    Trains a GNNAutoencoder model.
    """
    train_mask = getattr(data, 'train_mask', None)
    if train_mask is None:
        print("Warning: 'train_mask' not found in data. Training on all nodes for autoencoder.")
        train_mask = torch.ones(data.num_nodes, dtype=torch.bool, device=data.x.device) # Ensure mask is on same device
    
    if not train_mask.any():
        print("Warning: train_mask is empty or all False. No nodes to train on.")
        return model

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        
        x_reconstructed = model(data)
        
        loss = criterion(x_reconstructed[train_mask], data.x[train_mask])
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch+1}/{n_epochs}, Training Loss: {loss.item():.4f}")
            
    return model


def get_reconstruction_errors(model: GNNAutoencoder, data: Data) -> np.ndarray:
    """
    Calculates per-node reconstruction errors using a trained GNNAutoencoder.
    """
    model.eval()
    with torch.no_grad():
        x_reconstructed = model(data)
    
    if data.x.shape != x_reconstructed.shape:
        raise ValueError(f"Original features shape {data.x.shape} and reconstructed features shape {x_reconstructed.shape} mismatch.")

    # Mean Squared Error per node
    errors = torch.mean((data.x - x_reconstructed)**2, dim=1)
    
    return errors.detach().cpu().numpy()


def get_anomalies_by_error(reconstruction_errors: np.ndarray, 
                           node_ids: list,  # e.g., data.establishment_ids or list(data.node_id_map.keys())
                           top_n: int = 10) -> list[tuple[any, float]]:
    """
    Identifies the top_n nodes with the highest reconstruction errors.

    Args:
        reconstruction_errors (np.ndarray): Per-node reconstruction errors.
        node_ids (list): List of original node identifiers, in the same order as errors.
        top_n (int): Number of top anomalies to return.

    Returns:
        list[tuple[any, float]]: List of (node_id, error_score) for the top_n anomalies.
    """
    if len(reconstruction_errors) != len(node_ids):
        raise ValueError("Length of reconstruction_errors and node_ids must match.")

    # Combine errors with node_ids
    node_errors = list(zip(node_ids, reconstruction_errors))
    
    # Sort by error in descending order
    node_errors.sort(key=lambda x: x[1], reverse=True)
    
    return node_errors[:top_n]
