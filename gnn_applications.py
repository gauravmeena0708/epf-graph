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
    Preprocesses node features by applying one-hot encoding to specified categorical attributes.

    Only 'industry', 'city', and 'size_category' are one-hot encoded. Other columns like 'name'
    are excluded from the final feature set to ensure a purely numeric feature matrix.
    The 'establishment_id' column is preserved for mapping.

    Args:
        nodes_df (pd.DataFrame): DataFrame containing node information, including 'establishment_id'.
                                 It is expected to have 'industry', 'city', 'size_category' columns.

    Returns:
        tuple:
            - features_df_final (pd.DataFrame): DataFrame with 'establishment_id' and one-hot encoded features.
                                                This DataFrame only contains numeric features suitable for tensor conversion.
            - feature_names (list): List of strings representing the names of the generated one-hot encoded feature columns.
    Raises:
        ValueError: If 'establishment_id' column is missing or if any of the expected
                    categorical feature columns ('industry', 'city', 'size_category') are missing.
    """
    if 'establishment_id' not in nodes_df.columns:
        raise ValueError("nodes_df must contain 'establishment_id' column.")

    categorical_to_encode = ['industry', 'city', 'size_category']

    # Ensure all specified categorical features are present in the input nodes_df
    missing_cols = [col for col in categorical_to_encode if col not in nodes_df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected categorical columns in nodes_df for one-hot encoding: {missing_cols}")

    # Select only the establishment_id and the categorical columns for one-hot encoding.
    # This explicitly excludes other columns (like 'name') from being part of the feature matrix.
    df_for_encoding = nodes_df[['establishment_id'] + categorical_to_encode].copy()
    
    # Set 'establishment_id' as index before get_dummies to align features correctly
    df_for_encoding_indexed = df_for_encoding.set_index('establishment_id')

    # Apply one-hot encoding. The original columns listed in categorical_to_encode are dropped
    # and replaced by their one-hot encoded counterparts.
    features_one_hot_df = pd.get_dummies(df_for_encoding_indexed, columns=categorical_to_encode, prefix=categorical_to_encode)
    
    # Reset index so 'establishment_id' becomes a column again.
    # features_df_final will contain 'establishment_id' and only the numeric one-hot encoded columns.
    features_df_final = features_one_hot_df.reset_index()
    
    # The feature names are all columns in features_df_final except 'establishment_id'.
    # These are the names of the columns that will form the node feature matrix 'x'.
    feature_names = [col for col in features_df_final.columns if col != 'establishment_id']
    
    return features_df_final, feature_names


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
                                                   This DataFrame should only contain numeric features and the ID.
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

    graph_nodes_original_ids = list(nx_graph.nodes()) 
    node_id_map = {node_id: i for i, node_id in enumerate(graph_nodes_original_ids)}
    
    features_df_indexed = processed_node_features_df.set_index('establishment_id')
    
    # Ensure feature_cols only contains columns present in features_df_indexed
    # This is a safeguard, though preprocess_node_features should ensure this.
    valid_feature_cols = [col for col in feature_cols if col in features_df_indexed.columns]
    if len(valid_feature_cols) != len(feature_cols):
        print(f"Warning: Some feature_cols were not found in processed_node_features_df. Using valid subset.")
        # This situation ideally shouldn't happen if preprocess_node_features is correct.
    
    aligned_features_df = features_df_indexed.reindex(graph_nodes_original_ids)[valid_feature_cols]
    
    if aligned_features_df.isnull().values.any():
        missing_feature_nodes = aligned_features_df[aligned_features_df.isnull().any(axis=1)].index.tolist()
        print(f"Warning: Nodes {missing_feature_nodes} are in the graph but missing features after reindexing. Filling with zeros.")
        aligned_features_df = aligned_features_df.fillna(0) 

    # Convert the aligned feature DataFrame values to a NumPy array and then to a PyTorch tensor
    # Ensure the underlying NumPy array is of a numeric type before tensor conversion.
    # pd.get_dummies typically creates uint8, which is fine. fillna(0) also maintains numeric.
    try:
        # Explicitly convert to a numeric type like float32 if there's any doubt,
        # though pandas should handle this with get_dummies and fillna(0).
        # The .values attribute gives a NumPy array.
        feature_values_np = aligned_features_df.values.astype(np.float32) 
        x = torch.tensor(feature_values_np, dtype=torch.float)
    except Exception as e:
        print(f"Error converting features to tensor. DataFrame dtypes: {aligned_features_df.dtypes}")
        raise e


    edge_list = []
    edge_attributes_list = []
    for u_orig, v_orig, edge_data_dict in nx_graph.edges(data=True):
        if u_orig in node_id_map and v_orig in node_id_map: 
            u_mapped, v_mapped = node_id_map[u_orig], node_id_map[v_orig]
            edge_list.append((u_mapped, v_mapped))
            edge_attributes_list.append(edge_data_dict.get('weight', 1.0))

    if not edge_list: 
        edge_index = torch.empty((2, 0), dtype=torch.long) 
        edge_attr = torch.empty((0, 1), dtype=torch.float) 
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attributes_list, dtype=torch.float).unsqueeze(1) 

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    original_nodes_df_indexed = original_nodes_df.set_index('establishment_id')
    aligned_original_nodes_for_labels = original_nodes_df_indexed.reindex(graph_nodes_original_ids)

    label_column_names = ['industry', 'city', 'size_category']
    data.label_encoders = {}

    for col_name in label_column_names:
        if col_name in aligned_original_nodes_for_labels:
            le = LabelEncoder()
            labels_str = aligned_original_nodes_for_labels[col_name].fillna('Unknown').astype(str)
            encoded_labels = le.fit_transform(labels_str)
            
            setattr(data, f'y_{col_name}', torch.tensor(encoded_labels, dtype=torch.long))
            setattr(data, f'y_{col_name}_mapping', {i: cls_name for i, cls_name in enumerate(le.classes_)})
            data.label_encoders[col_name] = le

    data.node_id_map = node_id_map 
    data.idx_to_node_id = {v: k for k, v in node_id_map.items()} 
    data.establishment_ids = graph_nodes_original_ids 

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
        
        if epoch % 10 == 0 or epoch == n_epochs -1 : # Print every 10 epochs or last epoch
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
        
        if reason: # Only print if there's a specific reason identified
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