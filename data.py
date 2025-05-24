import networkx as nx
import pandas as pd
import random

# --- Sample Data Definitions (as before) ---
DEFAULT_ESTABLISHMENTS_DATA = {
    "EST001": {"name": "Alpha Tech Solutions", "industry": "Information Technology", "size_category": "Large", "city": "Bangalore"},
    "EST002": {"name": "Beta Manufacturing Co.", "industry": "Manufacturing", "size_category": "Medium", "city": "Pune"},
    "EST003": {"name": "Gamma Logistics", "industry": "Logistics", "size_category": "Small", "city": "Chennai"},
    "EST004": {"name": "Delta Innovations", "industry": "Information Technology", "size_category": "Medium", "city": "Hyderabad"},
    "EST005": {"name": "Epsilon Consulting", "industry": "Consulting", "size_category": "Large", "city": "Mumbai"},
    "EST006": {"name": "Zeta Pharma", "industry": "Pharmaceuticals", "size_category": "Large", "city": "Ahmedabad"},
    "EST007": {"name": "Omega IT Services", "industry": "Information Technology", "size_category": "Small", "city": "Bangalore"},
}
DEFAULT_TRANSFERS_DATA = [
    # Month 1
    ("EST001", "EST004", 35, 1), # IT to IT (different city)
    ("EST001", "EST007", 15, 1), # IT to IT (same city)
    ("EST002", "EST003", 20, 1), # Manufacturing to Logistics
    # Month 2
    ("EST004", "EST001", 10, 2), # IT back to IT
    ("EST005", "EST001", 25, 2), # Consulting to IT
    ("EST003", "EST002", 5, 2),  # Logistics back to Manufacturing
    # Month 3
    ("EST006", "EST004", 30, 3), # Pharma to IT
    ("EST001", "EST005", 12, 3), # IT to Consulting
    # Month 4
    ("EST002", "EST006", 18, 4), # Manufacturing to Pharma
    ("EST007", "EST004", 8, 4),  # IT to IT (different city, small to medium)
]

def get_sample_data():
    """Returns sample establishments, transfers, and corresponding DataFrames."""
    # Node List DataFrame from establishments_data
    node_list_data = []
    for est_id, attributes in DEFAULT_ESTABLISHMENTS_DATA.items():
        node_info = {'establishment_id': est_id}
        node_info.update(attributes)
        node_list_data.append(node_info)
    nodes_df = pd.DataFrame(node_list_data)

    # Edge List DataFrame from transfers_data
    edge_list_data = []
    # Now expecting (from_est, to_est, members, month)
    for from_est, to_est, members, month in DEFAULT_TRANSFERS_DATA:
        edge_info = {
            'source_establishment_id': from_est,
            'target_establishment_id': to_est,
            'members_transferred': members,
            'month': month # new field
        }
        edge_list_data.append(edge_info)
    edges_df = pd.DataFrame(edge_list_data)
    
    return DEFAULT_ESTABLISHMENTS_DATA, DEFAULT_TRANSFERS_DATA, nodes_df, edges_df

def create_graph_from_datasets(establishments_data_dict, transfers_data_list):
    """
    Creates a NetworkX graph from the provided data structures.
    The 'month' attribute from transfers_data_list is not added to graph edges here by default.
    """
    G = nx.DiGraph()
    for est_id, attributes in establishments_data_dict.items():
        G.add_node(est_id, **attributes)
    # transfers_data_list items are now (from_est, to_est, members, month)
    # We only use the first 3 for basic graph structure here.
    for from_est, to_est, members, _month in transfers_data_list: # _month is ignored for graph structure
        if G.has_node(from_est) and G.has_node(to_est):
            G.add_edge(from_est, to_est, weight=members) # Add weight
        else:
            print(f"Warning: Could not add edge ({from_est} -> {to_est}). One or both nodes do not exist.")
    return G

def save_data_to_csv(nodes_df, edges_df, nodes_csv_file="epfo_establishments_nodes.csv", edges_csv_file="epfo_transfers_edges.csv"):
    """Saves node and edge DataFrames to CSV files."""
    nodes_df.to_csv(nodes_csv_file, index=False)
    print(f"Node data saved to: {nodes_csv_file}")
    edges_df.to_csv(edges_csv_file, index=False)
    print(f"Edge data saved to: {edges_csv_file}")

def save_graph_to_gexf(G, gexf_file="epfo_transfers_graph.gexf"):
    """Saves the graph to a GEXF file."""
    try:
        nx.write_gexf(G, gexf_file)
        print(f"Graph data saved in GEXF format to: {gexf_file}")
    except Exception as e:
        print(f"Error writing GEXF file: {e}")

def load_graph_data(nodes_filepath="epfo_establishments_nodes.csv", edges_filepath="epfo_transfers_edges.csv"):
    """Loads graph data from CSV files and reconstructs the graph."""
    nodes_df = pd.read_csv(nodes_filepath)
    edges_df = pd.read_csv(edges_filepath)

    G = nx.DiGraph()

    # Add nodes with attributes
    for _, row in nodes_df.iterrows():
        node_id = row['establishment_id']
        attrs = row.drop('establishment_id').to_dict()
        G.add_node(node_id, **attrs)

    # Add edges with weights. 'month' column, if present in CSV's edges_df,
    # is not used for basic graph edge attributes here but is part of the returned edges_df.
    for _, row in edges_df.iterrows():
        G.add_edge(row['source_establishment_id'], 
                   row['target_establishment_id'], 
                   weight=row['members_transferred'])
    
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges from CSV.")
    return G, nodes_df, edges_df

if __name__ == "__main__":
    # This block now serves for testing data.py itself or for initial data generation
    print("--- data.py executed directly ---")
    
    # 1. Get sample data
    establishments, transfers, nodes_df, edges_df = get_sample_data()
    print(f"Sample establishments: {len(establishments)}, Sample transfers: {len(transfers)}")
    print("Sample nodes DataFrame head:\n", nodes_df.head())
    print("Sample edges DataFrame head:\n", edges_df.head())

    # 2. Create graph from this sample data
    G_sample = create_graph_from_datasets(establishments, transfers)
    print(f"Created graph from sample data: {G_sample.number_of_nodes()} nodes, {G_sample.number_of_edges()} edges.")

    # 3. Save the sample data to CSV and GEXF as an example run
    # This is useful if you want to generate these files once
    save_data_to_csv(nodes_df, edges_df)
    save_graph_to_gexf(G_sample)

    # 4. Example of loading data back
    try:
        G_loaded, _, _ = load_graph_data()
        print(f"Successfully loaded graph from CSV: {G_loaded.number_of_nodes()} nodes, {G_loaded.number_of_edges()} edges.")
    except FileNotFoundError:
        print("CSV files not found. Run once to generate them if needed.")
    
    # Matplotlib visualization part from original data.py can be moved here
    # or to a dedicated visualization module/function if complex.
    # For now, keeping it commented out from __main__ to avoid auto-plotting on import.
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 10))
    # pos = nx.spring_layout(G_sample, k=0.5, iterations=50)
    # nx.draw_networkx_nodes(G_sample, pos, node_size=2500, node_color="skyblue", alpha=0.9)
    # nx.draw_networkx_edges(G_sample, pos, arrowstyle="->", arrowsize=15, edge_color="gray", width=1.5, connectionstyle='arc3,rad=0.1')
    # nx.draw_networkx_labels(G_sample, pos, font_size=9, font_weight="bold")
    # edge_labels = nx.get_edge_attributes(G_sample, 'weight')
    # nx.draw_networkx_edge_labels(G_sample, pos, edge_labels=edge_labels, font_color='red', font_size=8)
    # plt.title("EPFO Member Transfers Between Establishments (Sample from data.py)", fontsize=15)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    print("\n--- data.py direct execution finished ---")