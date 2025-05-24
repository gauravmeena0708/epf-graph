# graph_analysis.py
# This file will contain functions for various graph analysis insights.
import networkx as nx
import pandas as pd
from collections import Counter # Keep Counter if used elsewhere, otherwise can be removed
import community as community_louvain # For community detection

# This function can be removed if create_graph_from_datasets from data.py is used consistently.
# For now, keeping it as it might be used by other parts or for specific tests.
def get_graph_from_data(establishments_data, transfers_data):
    """
    Creates a NetworkX DiGraph from establishment and transfer data.
    """
    G = nx.DiGraph()
    for est_id, attributes in establishments_data.items():
        G.add_node(est_id, **attributes)
    
    for from_est, to_est, members in transfers_data:
        if G.has_node(from_est) and G.has_node(to_est):
            G.add_edge(from_est, to_est, weight=members)
        else:
            print(f"Warning: Could not add edge ({from_est} -> {to_est}). One or both nodes do not exist.")
    return G

def get_node_attributes_df(G, node_ids, columns_prefix=""):
    """Helper to get node attributes as a DataFrame."""
    data = []
    for node_id in node_ids:
        attrs = G.nodes[node_id]
        data.append({
            f'{columns_prefix}establishment_id': node_id,
            f'{columns_prefix}name': attrs.get('name', 'N/A'),
            f'{columns_prefix}industry': attrs.get('industry', 'N/A'),
            f'{columns_prefix}city': attrs.get('city', 'N/A'),
            f'{columns_prefix}size_category': attrs.get('size_category', 'N/A')
        })
    return pd.DataFrame(data).set_index(f'{columns_prefix}establishment_id')

# --- Establishment-Level Insights ---
def get_top_source_establishments(G, top_n=5):
    """
    Calculates out-degree and total members leaving for each node.
    Returns a DataFrame sorted by total_members_out.
    """
    out_degrees = dict(G.out_degree())
    total_members_out = dict(G.out_degree(weight='weight'))
    
    data = []
    for node_id in G.nodes():
        data.append({
            'establishment_id': node_id,
            'name': G.nodes[node_id].get('name', node_id),
            'out_degree': out_degrees.get(node_id, 0),
            'total_members_out': total_members_out.get(node_id, 0)
        })
    
    df = pd.DataFrame(data)
    return df.sort_values(by='total_members_out', ascending=False).head(top_n)

def get_top_destination_establishments(G, top_n=5):
    """
    Calculates in-degree and total members joining for each node.
    Returns a DataFrame sorted by total_members_in.
    """
    in_degrees = dict(G.in_degree())
    total_members_in = dict(G.in_degree(weight='weight'))
    
    data = []
    for node_id in G.nodes():
        data.append({
            'establishment_id': node_id,
            'name': G.nodes[node_id].get('name', node_id),
            'in_degree': in_degrees.get(node_id, 0),
            'total_members_in': total_members_in.get(node_id, 0)
        })
        
    df = pd.DataFrame(data)
    return df.sort_values(by='total_members_in', ascending=False).head(top_n)

def get_net_gainers_losers(G):
    """
    Calculates net change (Total members in - Total members out) for each establishment.
    Returns a DataFrame sorted by net_change.
    """
    total_members_in = dict(G.in_degree(weight='weight'))
    total_members_out = dict(G.out_degree(weight='weight'))
    
    data = []
    for node_id in G.nodes():
        members_in = total_members_in.get(node_id, 0)
        members_out = total_members_out.get(node_id, 0)
        data.append({
            'establishment_id': node_id,
            'name': G.nodes[node_id].get('name', node_id),
            'total_members_in': members_in,
            'total_members_out': members_out,
            'net_change': members_in - members_out
        })
        
    df = pd.DataFrame(data)
    return df.sort_values(by='net_change', ascending=False)

def get_hub_establishments(G, top_n=5):
    """
    Calculates betweenness and eigenvector centrality for each node.
    Returns a DataFrame sorted by betweenness_centrality.
    Handles potential PowerIterationFailedConvergence for eigenvector centrality.
    """
    if not G.nodes: return pd.DataFrame() # Handle empty graph
    betweenness_centrality = nx.betweenness_centrality(G, weight='weight', normalized=True)
    
    try:
        eigenvector_centrality = nx.eigenvector_centrality_numpy(G, weight='weight', max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        print("Warning: Eigenvector centrality did not converge. Results might be approximate or zero.")
        eigenvector_centrality = {node: 0.0 for node in G.nodes()} 
    except Exception as e: 
        print(f"Error calculating eigenvector centrality: {e}. Defaulting to 0.")
        eigenvector_centrality = {node: 0.0 for node in G.nodes()}

    data = []
    for node_id in G.nodes():
        data.append({
            'establishment_id': node_id,
            'name': G.nodes[node_id].get('name', node_id),
            'betweenness_centrality': betweenness_centrality.get(node_id, 0.0),
            'eigenvector_centrality': eigenvector_centrality.get(node_id, 0.0)
        })
        
    df = pd.DataFrame(data)
    return df.sort_values(by='betweenness_centrality', ascending=False).head(top_n)

def get_isolated_connected_establishments(G, min_degree_threshold=1):
    """
    Identifies isolated and connected establishments based on total degree.
    Returns two lists of establishment IDs: isolated_ids, connected_ids.
    """
    isolated_ids = []
    connected_ids = []
    
    if not G.nodes(): return isolated_ids, connected_ids

    total_degrees = dict(G.degree()) 
    
    for node_id in G.nodes():
        node_degree = total_degrees.get(node_id, 0)
        if node_degree <= min_degree_threshold:
            isolated_ids.append(node_id)
        else:
            connected_ids.append(node_id)
            
    return isolated_ids, connected_ids

# --- Transfer Pattern Insights ---

def get_dominant_transfer_routes(G, top_n=5):
    """
    Identifies edges with the highest 'weight' attribute.
    Returns a pandas DataFrame with columns: source_id, source_name, target_id, target_name, members_transferred.
    """
    if not G.edges: return pd.DataFrame()

    transfers = []
    for u, v, data in G.edges(data=True):
        transfers.append({
            'source_id': u,
            'source_name': G.nodes[u].get('name', u),
            'target_id': v,
            'target_name': G.nodes[v].get('name', v),
            'members_transferred': data.get('weight', 0)
        })
    
    df = pd.DataFrame(transfers)
    return df.sort_values(by='members_transferred', ascending=False).head(top_n)

def get_reciprocal_transfers(G, min_transfers_oneway=1):
    """
    Finds pairs of establishments (A, B) where there's a transfer from A to B AND from B to A.
    Returns a pandas DataFrame.
    """
    if not G.edges: return pd.DataFrame()
    
    reciprocal_pairs = []
    processed_pairs = set() 

    for u, v, data_uv in G.edges(data=True):
        if (v, u) in G.edges() and frozenset((u,v)) not in processed_pairs:
            data_vu = G.get_edge_data(v, u)
            members_uv = data_uv.get('weight', 0)
            members_vu = data_vu.get('weight', 0)

            if members_uv >= min_transfers_oneway and members_vu >= min_transfers_oneway:
                reciprocal_pairs.append({
                    'establishment_A_id': u,
                    'establishment_A_name': G.nodes[u].get('name', u),
                    'establishment_B_id': v,
                    'establishment_B_name': G.nodes[v].get('name', v),
                    'A_to_B_members': members_uv,
                    'B_to_A_members': members_vu
                })
            processed_pairs.add(frozenset((u,v)))
            
    return pd.DataFrame(reciprocal_pairs)

def _get_attribute_transfer_patterns(G, attribute_name, value_A, value_B=None):
    """Generic helper for industry, location, and size category patterns."""
    if not G.nodes: return G.subgraph([]), pd.DataFrame() if value_B is None else pd.DataFrame()

    transfers_summary = []

    if value_B is None: 
        nodes_A = [n for n, data in G.nodes(data=True) if data.get(attribute_name) == value_A]
        if not nodes_A: return G.subgraph([]), pd.DataFrame()
        
        subgraph = G.subgraph(nodes_A)
        
        for u, v, data in subgraph.edges(data=True):
            transfers_summary.append({
                'source_id': u,
                'source_name': G.nodes[u].get('name', u),
                f'source_{attribute_name}': G.nodes[u].get(attribute_name, 'N/A'),
                'target_id': v,
                'target_name': G.nodes[v].get('name', v),
                f'target_{attribute_name}': G.nodes[v].get(attribute_name, 'N/A'),
                'members_transferred': data.get('weight', 0)
            })
        summary_df = pd.DataFrame(transfers_summary)
        return subgraph, summary_df.sort_values(by='members_transferred', ascending=False)

    else: 
        for u, v, data in G.edges(data=True):
            attr_u = G.nodes[u].get(attribute_name)
            attr_v = G.nodes[v].get(attribute_name)
            if attr_u == value_A and attr_v == value_B:
                transfers_summary.append({
                    'source_id': u,
                    'source_name': G.nodes[u].get('name', u),
                    f'source_{attribute_name}': attr_u,
                    'target_id': v,
                    'target_name': G.nodes[v].get('name', v),
                    f'target_{attribute_name}': attr_v,
                    'members_transferred': data.get('weight', 0)
                })
        summary_df = pd.DataFrame(transfers_summary)
        return summary_df.sort_values(by='members_transferred', ascending=False)


def get_industry_transfer_patterns(G, industry_A, industry_B=None):
    """Analyzes transfers based on industry."""
    return _get_attribute_transfer_patterns(G, 'industry', industry_A, industry_B)

def get_location_transfer_patterns(G, location_A, location_B=None):
    """Analyzes transfers based on city (location)."""
    return _get_attribute_transfer_patterns(G, 'city', location_A, location_B)

def get_size_category_transfer_patterns(G, size_cat_A, size_cat_B=None):
    """Analyzes transfers based on size_category."""
    return _get_attribute_transfer_patterns(G, 'size_category', size_cat_A, size_cat_B)

# --- Network Structure & Community Insights ---

def detect_communities(G_undirected):
    """
    Detects communities using Louvain algorithm on an UNDIRECTED graph.
    Assigns 'community_id' attribute to each node in G_undirected.
    Returns a DataFrame (id, name, community_id) and the partition dictionary.
    The input graph G_undirected is modified in place.
    """
    if not G_undirected.nodes() or G_undirected.number_of_edges() == 0:
        print("Warning: Community detection requires a graph with nodes and edges.")
        return pd.DataFrame(columns=['establishment_id', 'name', 'community_id']), {}

    # Ensure the graph is undirected (Louvain typically expects this)
    if nx.is_directed(G_undirected):
        # This function now expects G_undirected to be already undirected.
        # If it can be directed, it should be converted before calling this.
        print("Warning: Louvain community detection is typically run on undirected graphs. Consider converting.")
    
    partition = community_louvain.best_partition(G_undirected, weight='weight')
    
    # Assign community_id to nodes in the graph G_undirected
    nx.set_node_attributes(G_undirected, partition, 'community_id')
    
    community_data = []
    for node_id, comm_id in partition.items():
        community_data.append({
            'establishment_id': node_id,
            'name': G_undirected.nodes[node_id].get('name', node_id), # Get name from graph attributes
            'community_id': comm_id
        })
    
    df = pd.DataFrame(community_data)
    return df.sort_values(by='community_id'), partition

def get_network_density(G):
    """Calculates the density of the graph G."""
    if not G.nodes(): return 0.0
    return nx.density(G)

def get_bridge_establishments(G_original_directed, communities_partition):
    """
    Identifies bridge establishments connecting different communities.
    G_original_directed: The original graph (can be directed) used to check edges.
    communities_partition: Dictionary mapping node_id to community_id.
    """
    if not G_original_directed.edges or not communities_partition:
        return pd.DataFrame(columns=['establishment_id', 'name', 'community_id', 'connected_communities_count', 'connected_to_communities'])

    bridge_nodes_data = []
    
    for node_id in G_original_directed.nodes():
        node_community = communities_partition.get(node_id)
        if node_community is None: continue # Node not in a community

        connected_external_communities = set()
        
        # Check outgoing edges
        for _, neighbor_id in G_original_directed.out_edges(node_id):
            neighbor_community = communities_partition.get(neighbor_id)
            if neighbor_community is not None and neighbor_community != node_community:
                connected_external_communities.add(neighbor_community)
        
        # Check incoming edges (optional, depends on definition of bridge)
        # For this task, we consider outgoing connections as primary indicators
        # for u, _ in G_original_directed.in_edges(node_id):
        #     u_community = communities_partition.get(u)
        #     if u_community is not None and u_community != node_community:
        #         connected_external_communities.add(u_community)


        if connected_external_communities:
            bridge_nodes_data.append({
                'establishment_id': node_id,
                'name': G_original_directed.nodes[node_id].get('name', node_id),
                'community_id': node_community,
                'connected_communities_count': len(connected_external_communities),
                'connected_to_communities': sorted(list(connected_external_communities))
            })
            
    df = pd.DataFrame(bridge_nodes_data)
    return df.sort_values(by='connected_communities_count', ascending=False)


if __name__ == '__main__':
    from data import get_sample_data, create_graph_from_datasets 

    print("--- graph_analysis.py executed directly for testing ---")
    
    establishments_data_dict, transfers_data_list, nodes_df_sample, edges_df_sample = get_sample_data()
    G_sample_directed = create_graph_from_datasets(establishments_data_dict, transfers_data_list) # Original Directed Graph
    
    print(f"Sample graph created: {G_sample_directed.number_of_nodes()} nodes, {G_sample_directed.number_of_edges()} edges.")
    if G_sample_directed.number_of_nodes() == 0:
        print("Sample graph is empty. Skipping analyses.")
    else:
        # --- Test Establishment-Level Insights ---
        print("\n--- Top Source Establishments ---")
        print(get_top_source_establishments(G_sample_directed, top_n=3))
        print("\n--- Top Destination Establishments ---")
        print(get_top_destination_establishments(G_sample_directed, top_n=3))
        print("\n--- Net Gainers/Losers ---")
        print(get_net_gainers_losers(G_sample_directed).head())
        print("\n--- Hub Establishments ---")
        try:
            print(get_hub_establishments(G_sample_directed, top_n=3))
        except Exception as e: print(f"Could not calculate hub establishments: {e}")
        print("\n--- Isolated vs. Connected Establishments ---")
        isolated, _ = get_isolated_connected_establishments(G_sample_directed, min_degree_threshold=0)
        print(f"Isolated IDs (degree 0): {isolated}")
        
        # --- Test Transfer Pattern Insights ---
        print("\n--- Dominant Transfer Routes ---")
        print(get_dominant_transfer_routes(G_sample_directed, top_n=3))
        print("\n--- Reciprocal Transfers (min 1 one way) ---")
        print(get_reciprocal_transfers(G_sample_directed, min_transfers_oneway=1))

        it_industry = "Information Technology"
        man_industry = "Manufacturing"
        city_bglr = "Bangalore"
        size_large = "Large"
        size_small = "Small"

        print(f"\n--- Industry Transfer Patterns (Intra-Industry: {it_industry}) ---")
        subgraph_it, intra_it_transfers_df = get_industry_transfer_patterns(G_sample_directed, industry_A=it_industry)
        if subgraph_it.number_of_nodes() > 0 : print(intra_it_transfers_df.head())
        else: print(intra_it_transfers_df)
        
        print(f"\n--- Industry Transfer Patterns (Inter-Industry: {man_industry} to {it_industry}) ---")
        print(get_industry_transfer_patterns(G_sample_directed, industry_A=man_industry, industry_B=it_industry).head())
        
        # --- Test Network Structure & Community Insights ---
        print("\n--- Network Density ---")
        density_directed = get_network_density(G_sample_directed)
        print(f"The density of the original directed graph is: {density_directed:.4f}")
        
        # For community detection and density of undirected version
        G_community_analysis = G_sample_directed.to_undirected()
        # Copy weights for Louvain if they exist in directed graph
        for u, v, data in G_sample_directed.edges(data=True):
            if G_community_analysis.has_edge(u, v):
                G_community_analysis[u][v]['weight'] = data.get('weight', 1)
        
        density_undirected = get_network_density(G_community_analysis)
        print(f"The density of the undirected graph (for community analysis) is: {density_undirected:.4f}")

        print("\n--- Community Detection (on Undirected Graph with Weights) ---")
        partition = {} # Initialize partition
        try:
            # Pass the graph G_community_analysis which is undirected and has weights
            # detect_communities will add 'community_id' to nodes of G_community_analysis
            communities_df, partition = detect_communities(G_community_analysis) 
            print(communities_df.head())
            
            # To use bridge detection on original G_sample_directed, we need community IDs on its nodes.
            # We can copy them from G_community_analysis if the node sets are identical.
            if partition:
                 nx.set_node_attributes(G_sample_directed, {node_id: G_community_analysis.nodes[node_id].get('community_id') 
                                                      for node_id in G_community_analysis.nodes()}, 'community_id')


        except community_louvain.LouvainArgumentError as lae:
            print(f"Louvain argument error: {lae}. This can happen with very small or disconnected graphs.")
        except Exception as e:
            print(f"Community detection failed: {e}")
            print("Ensure 'python-louvain' is installed and the graph is suitable (e.g., connected components).")

        if partition: 
            print("\n--- Bridge Establishments (on Original Directed Graph with Community Info) ---")
            # G_sample_directed now has 'community_id' attributes if partition was successful
            bridge_nodes_df = get_bridge_establishments(G_sample_directed, partition) 
            print(bridge_nodes_df.head())
        else:
            print("Skipping bridge establishments as no communities were detected or partition failed.")
            
    print("\n--- graph_analysis.py direct execution finished ---")
