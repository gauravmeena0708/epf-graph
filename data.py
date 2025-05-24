import networkx as nx
import matplotlib.pyplot as plt # For basic visualization
import pandas as pd
import random # For generating some sample data

# --- 1. Define Sample Establishments (Nodes) ---
# Each establishment has an ID and some attributes.
establishments_data = {
    "EST001": {"name": "Alpha Tech Solutions", "industry": "Information Technology", "size_category": "Large", "city": "Bangalore"},
    "EST002": {"name": "Beta Manufacturing Co.", "industry": "Manufacturing", "size_category": "Medium", "city": "Pune"},
    "EST003": {"name": "Gamma Logistics", "industry": "Logistics", "size_category": "Small", "city": "Chennai"},
    "EST004": {"name": "Delta Innovations", "industry": "Information Technology", "size_category": "Medium", "city": "Hyderabad"},
    "EST005": {"name": "Epsilon Consulting", "industry": "Consulting", "size_category": "Large", "city": "Mumbai"},
    "EST006": {"name": "Zeta Pharma", "industry": "Pharmaceuticals", "size_category": "Large", "city": "Ahmedabad"},
    "EST007": {"name": "Omega IT Services", "industry": "Information Technology", "size_category": "Small", "city": "Bangalore"},
}

# --- 2. Define Sample Transfers (Edges with Weights) ---
# Format: (from_establishment_id, to_establishment_id, number_of_members_transferred)
transfers_data = [
    ("EST001", "EST004", 35), # IT to IT (different city)
    ("EST001", "EST007", 15), # IT to IT (same city)
    ("EST002", "EST003", 20), # Manufacturing to Logistics
    ("EST004", "EST001", 10), # IT back to IT
    ("EST005", "EST001", 25), # Consulting to IT
    ("EST003", "EST002", 5),  # Logistics back to Manufacturing
    ("EST006", "EST004", 30), # Pharma to IT
    ("EST001", "EST005", 12), # IT to Consulting
    ("EST002", "EST006", 18), # Manufacturing to Pharma
    ("EST007", "EST004", 8),  # IT to IT (different city, small to medium)
]

# --- 3. Create the Graph using NetworkX ---
G = nx.DiGraph() # Create a Directed Graph

# Add nodes with their attributes
for est_id, attributes in establishments_data.items():
    G.add_node(est_id, **attributes)
    # print(f"Added node: {est_id} with attributes {attributes}") # For debugging

# Add edges with weights (number of members transferred)
for from_est, to_est, members in transfers_data:
    if G.has_node(from_est) and G.has_node(to_est):
        G.add_edge(from_est, to_est, weight=members, label=str(members)) # Add weight and label for visualization
        # print(f"Added edge: {from_est} -> {to_est} with weight {members}") # For debugging
    else:
        print(f"Warning: Could not add edge ({from_est} -> {to_est}). One or both nodes do not exist.")

# --- 4. Basic Graph Information (Optional: Print to console) ---
print("--- Graph Summary ---")
print(f"Number of establishments (nodes): {G.number_of_nodes()}")
print(f"Number of transfer routes (edges): {G.number_of_edges()}")

# You can uncomment these to see details in the console:
# print("\n--- Establishments (Nodes) ---")
# for node, data in G.nodes(data=True):
#     print(f"ID: {node}, Attributes: {data}")
#
# print("\n--- Transfers (Edges) ---")
# for source, target, data in G.edges(data=True):
#     print(f"From: {source}, To: {target}, Members Transferred: {data['weight']}")

# --- 5. Generate Sample Data Files ---

# 5.1. Node List CSV
node_list_data = []
for node, attrs in G.nodes(data=True):
    node_info = {'establishment_id': node}
    node_info.update(attrs)
    node_list_data.append(node_info)

nodes_df = pd.DataFrame(node_list_data)
nodes_csv_file = "epfo_establishments_nodes.csv"
nodes_df.to_csv(nodes_csv_file, index=False)
print(f"\nNode data (establishments) saved to: {nodes_csv_file}")

# 5.2. Edge List CSV
edge_list_data = []
for source, target, attrs in G.edges(data=True):
    edge_info = {
        'source_establishment_id': source,
        'target_establishment_id': target,
        'members_transferred': attrs.get('weight', 0) # Get weight, default to 0 if not found
    }
    edge_list_data.append(edge_info)

edges_df = pd.DataFrame(edge_list_data)
edges_csv_file = "epfo_transfers_edges.csv"
edges_df.to_csv(edges_csv_file, index=False)
print(f"Edge data (transfers) saved to: {edges_csv_file}")

# 5.3. GEXF File (for Gephi and other graph visualization tools)
gexf_file = "epfo_transfers_graph.gexf"
try:
    nx.write_gexf(G, gexf_file)
    print(f"Graph data saved in GEXF format to: {gexf_file}")
except Exception as e:
    print(f"Error writing GEXF file: {e}")
    print("Please ensure you have the 'lxml' library installed if you encounter GEXF writing issues (pip install lxml).")


# --- 6. Simple Visualization (Optional: using Matplotlib) ---
# This is a very basic visualization. For complex graphs, tools like Gephi are better.
# You might need to install matplotlib: pip install matplotlib

# plt.figure(figsize=(12, 10))
# pos = nx.spring_layout(G, k=0.5, iterations=50) # k adjusts spacing, iterations for convergence

# # Draw nodes
# nx.draw_networkx_nodes(G, pos, node_size=2500, node_color="skyblue", alpha=0.9)

# # Draw edges
# nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15, edge_color="gray", width=1.5, connectionstyle='arc3,rad=0.1')

# # Draw node labels
# nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")

# # Draw edge labels (weights)
# edge_labels = nx.get_edge_attributes(G, 'weight')
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)

# plt.title("EPFO Member Transfers Between Establishments (Sample)", fontsize=15)
# plt.axis('off') # Turn off the axis
# plt.tight_layout()
# # To save the plot:
# # plot_file = "epfo_transfers_plot.png"
# # plt.savefig(plot_file)
# # print(f"\nBasic graph plot saved to: {plot_file}")
# plt.show()

print("\n--- Script Finished ---")