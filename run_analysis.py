# run_analysis.py
import argparse
import pandas as pd # For print options
import networkx as nx # For type hinting or specific operations if needed

# Assuming data.py is in the same directory or accessible via PYTHONPATH
from data import get_sample_data, load_graph_data, create_graph_from_datasets #, save_data_to_csv, save_graph_to_gexf # Save functions are optional here

# Import all analysis functions from graph_analysis.py
from graph_analysis import (
    get_top_source_establishments,
    get_top_destination_establishments,
    get_net_gainers_losers,
    get_hub_establishments,
    get_isolated_connected_establishments,
    get_dominant_transfer_routes,
    get_reciprocal_transfers,
    get_industry_transfer_patterns,
    get_location_transfer_patterns,
    get_size_category_transfer_patterns,
    detect_communities,
    get_network_density,
    get_bridge_establishments,
    get_churn_analysis_by_attribute # Added import
)

def print_df(df, title=""):
    """Helper function to print DataFrames nicely."""
    if title:
        print(f"--- {title} ---")
    if df is not None and not df.empty:
        # For DataFrames with many columns, ensure all are shown.
        # .to_string() is generally good for console.
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(df) # Using default print for DataFrames, to_string can be too wide.
    else:
        print("No data to display or result is empty.")
    print("\n")

def main():
    parser = argparse.ArgumentParser(description="Run EPFO Graph Analysis")
    parser.add_argument(
        "--data_source",
        type=str,
        default="sample",
        choices=["sample", "csv"],
        help="Source of the data: 'sample' for generated data, 'csv' to load from default CSV files."
    )
    parser.add_argument(
        "--nodes_csv",
        type=str,
        default="epfo_establishments_nodes.csv",
        help="Path to nodes CSV file (used if data_source is 'csv')."
    )
    parser.add_argument(
        "--edges_csv",
        type=str,
        default="epfo_transfers_edges.csv",
        help="Path to edges CSV file (used if data_source is 'csv')."
    )
    parser.add_argument(
        "--run_establishment_insights",
        action="store_true",
        help="Run Establishment-Level Insights."
    )
    parser.add_argument(
        "--run_transfer_insights",
        action="store_true",
        help="Run Transfer Pattern Insights."
    )
    parser.add_argument(
        "--run_network_insights",
        action="store_true",
        help="Run Network Structure & Community Insights."
    )
    parser.add_argument(
        "--run_all_insights",
        action="store_true",
        help="Run all available insights."
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=5,
        help="Number of top items to display for relevant analyses."
    )
    parser.add_argument("--industry_A", type=str, help="Primary industry for pattern analysis")
    parser.add_argument("--industry_B", type=str, help="Secondary industry for inter-industry pattern analysis (optional)")
    parser.add_argument("--location_A", type=str, help="Primary location for pattern analysis")
    parser.add_argument("--location_B", type=str, help="Secondary location for inter-location pattern analysis (optional)")
    parser.add_argument("--size_A", type=str, help="Primary size category for pattern analysis")
    parser.add_argument("--size_B", type=str, help="Secondary size category for inter-size pattern analysis (optional)")

    # Arguments for Churn Analysis
    parser.add_argument(
        "--run_churn_analysis",
        action="store_true",
        help="Run churn analysis by specified attribute."
    )
    parser.add_argument(
        "--churn_attribute",
        type=str,
        default="industry",
        choices=["industry", "city", "size_category"],
        help="Attribute to group by for churn analysis (e.g., 'industry', 'city')."
    )
    # Argument for Criticality Analysis
    parser.add_argument(
        "--run_criticality_analysis",
        action="store_true",
        help="Run critical establishments analysis."
    )

    args = parser.parse_args()
    
    # Configure pandas display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000) # Adjust width for console
    pd.set_option('display.colheader_justify', 'left')


    print("Starting EPFO Analysis Orchestrator...")

    G = None
    nodes_df = None # Initialize nodes_df
    edges_df = None # Initialize edges_df

    if args.data_source == "sample":
        print("Using sample data.")
        # get_sample_data returns: establishments_data_dict, transfers_data_list, nodes_df, edges_df
        establishments_data_dict, transfers_data_list, nodes_df, edges_df = get_sample_data()
        G = create_graph_from_datasets(establishments_data_dict, transfers_data_list)
    elif args.data_source == "csv":
        print(f"Loading data from CSV files: {args.nodes_csv}, {args.edges_csv}")
        try:
            # load_graph_data returns: G, nodes_df, edges_df
            G, nodes_df, edges_df = load_graph_data(nodes_filepath=args.nodes_csv, edges_filepath=args.edges_csv)
        except FileNotFoundError:
            print(f"Error: One or both CSV files not found. Please check paths: {args.nodes_csv}, {args.edges_csv}")
            return
        except Exception as e:
            print(f"Error loading data from CSV: {e}")
            return

    if not G or G.number_of_nodes() == 0: # Check if G is None or empty
        print("Graph could not be loaded or generated, or is empty. Exiting.")
        return

    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    if nodes_df is not None: # Use the loaded nodes_df
        print_df(nodes_df.head(), "Sample Node Data (First 5 Rows from loaded nodes_df)")


    if args.run_all_insights:
        args.run_establishment_insights = True
        args.run_transfer_insights = True
        args.run_network_insights = True
        args.run_churn_analysis = True # Also include churn analysis if run_all is specified
        args.run_criticality_analysis = True # Also include criticality analysis if run_all is specified

    if args.run_establishment_insights:
        print("\n=== ESTABLISHMENT-LEVEL INSIGHTS ===")
        print_df(get_top_source_establishments(G, top_n=args.top_n), f"Top {args.top_n} Source Establishments")
        print_df(get_top_destination_establishments(G, top_n=args.top_n), f"Top {args.top_n} Destination Establishments")
        print_df(get_net_gainers_losers(G), "Net Gainers/Losers")
        try:
            print_df(get_hub_establishments(G, top_n=args.top_n), f"Top {args.top_n} Hub Establishments (Centrality)")
        except Exception as e:
            print(f"Could not calculate hub establishments: {e}")
        
        isolated, connected = get_isolated_connected_establishments(G, min_degree_threshold=0) # Threshold 0 for truly isolated
        print("--- Isolated vs. Connected ---")
        print(f"Isolated Establishments (degree 0): {len(isolated)} IDs - e.g., {isolated[:10] if isolated else 'None'}")
        # print(f"Connected Establishments (degree > 0): {len(connected)} IDs - e.g., {connected[:10] if connected else 'None'}") # Can be very long
        print("\n")


    if args.run_transfer_insights:
        print("\n=== TRANSFER PATTERN INSIGHTS ===")
        print_df(get_dominant_transfer_routes(G, top_n=args.top_n), f"Top {args.top_n} Dominant Transfer Routes")
        print_df(get_reciprocal_transfers(G, min_transfers_oneway=1), "Reciprocal Transfers (min 1 transfer each way)")

        # Industry patterns
        if args.industry_A:
            subgraph_industry, df_industry = get_industry_transfer_patterns(G, industry_A=args.industry_A, industry_B=args.industry_B)
            title = f"Intra-Industry Transfers: {args.industry_A}" if not args.industry_B else f"Inter-Industry Transfers: {args.industry_A} to {args.industry_B}"
            if subgraph_industry and subgraph_industry.number_of_nodes() > 0: print(f"Subgraph for {title} has {subgraph_industry.number_of_nodes()} nodes and {subgraph_industry.number_of_edges()} edges.")
            print_df(df_industry, title)
        else:
            print("Skipping industry pattern analysis: --industry_A not specified. Provide e.g., --industry_A \"Information Technology\"")

        # Location patterns
        if args.location_A:
            subgraph_loc, df_loc = get_location_transfer_patterns(G, location_A=args.location_A, location_B=args.location_B)
            title = f"Intra-Location Transfers: {args.location_A}" if not args.location_B else f"Inter-Location Transfers: {args.location_A} to {args.location_B}"
            if subgraph_loc and subgraph_loc.number_of_nodes() > 0: print(f"Subgraph for {title} has {subgraph_loc.number_of_nodes()} nodes and {subgraph_loc.number_of_edges()} edges.")
            print_df(df_loc, title)
        else:
            print("Skipping location pattern analysis: --location_A not specified. Provide e.g., --location_A Bangalore")

        # Size category patterns
        if args.size_A:
            subgraph_size, df_size = get_size_category_transfer_patterns(G, size_cat_A=args.size_A, size_cat_B=args.size_B)
            title = f"Intra-Size Category Transfers: {args.size_A}" if not args.size_B else f"Inter-Size Category Transfers: {args.size_A} to {args.size_B}"
            if subgraph_size and subgraph_size.number_of_nodes() > 0: print(f"Subgraph for {title} has {subgraph_size.number_of_nodes()} nodes and {subgraph_size.number_of_edges()} edges.")
            print_df(df_size, title)
        else:
            print("Skipping size category pattern analysis: --size_A not specified. Provide e.g., --size_A Large")


    if args.run_network_insights:
        print("\n=== NETWORK STRUCTURE & COMMUNITY INSIGHTS ===")
        print(f"--- Network Density (Original Graph) ---")
        print(f"Density: {get_network_density(G):.4f}\n")
        
        G_undirected_for_louvain = G.to_undirected()
        for u, v, data in G.edges(data=True): # Ensure weights are copied for weighted Louvain
            if G_undirected_for_louvain.has_edge(u,v):
                G_undirected_for_louvain[u][v]['weight'] = data.get('weight', 1.0) # Default weight 1 if not present
        
        # Also calculate density for the undirected version for comparison
        print(f"--- Network Density (Undirected Graph for Community Detection) ---")
        print(f"Density: {get_network_density(G_undirected_for_louvain):.4f}\n")

        try:
            print("--- Community Detection (Louvain Algorithm on Undirected Graph) ---")
            # detect_communities modifies G_undirected_for_louvain by adding 'community_id'
            communities_df, partition = detect_communities(G_undirected_for_louvain) 
            print_df(communities_df.head(), "Establishments with Community IDs (Sample)")
            
            if partition:
                # For bridge establishments, use the original graph G and the partition.
                # The 'community_id' attribute might not be on original G if G_undirected_for_louvain was a deep copy
                # that didn't share node attribute dictionaries.
                # get_bridge_establishments takes the partition map directly.
                print_df(get_bridge_establishments(G, partition), f"Top {args.top_n} Bridge Establishments")
            else:
                print("Skipping bridge establishments as no communities detected or partition is empty.")

        except ImportError:
            print("Community detection skipped: 'python-louvain' library not found. Please install it via pip install python-louvain.")
        except Exception as e:
            print(f"Error in community detection or bridge analysis: {e}")

    if args.run_churn_analysis:
        print(f"\n=== CHURN ANALYSIS BY {args.churn_attribute.upper()} ===")
        if nodes_df is None or edges_df is None:
            print(f"Error: Churn analysis requires nodes_df and edges_df. Data for '{args.data_source}' source might not have loaded them correctly.")
            if args.data_source == "sample":
                 print("Sample data loading should provide nodes_df and edges_df. Check get_sample_data().")
            elif args.data_source == "csv":
                 print(f"Ensure CSV files '{args.nodes_csv}' and '{args.edges_csv}' are present and correctly formatted.")
        elif args.churn_attribute not in nodes_df.columns:
            print(f"Error: Churn attribute '{args.churn_attribute}' not found in nodes_df columns. Available columns: {nodes_df.columns.tolist()}")
        else:
            try:
                churn_analysis_df = get_churn_analysis_by_attribute(G, nodes_df, edges_df, attribute_name=args.churn_attribute)
                print_df(churn_analysis_df, f"Churn Analysis by {args.churn_attribute}")
            except ValueError as ve:
                print(f"ValueError during churn analysis: {ve}")
            except Exception as e:
                print(f"An unexpected error occurred during churn analysis: {e}")

    if args.run_criticality_analysis:
        print(f"\n=== CRITICAL ESTABLISHMENTS REPORT (Top {args.top_n}) ===")
        if G is None or nodes_df is None:
            print("Error: Criticality analysis requires a graph (G) and nodes_df. Data might not have loaded correctly.")
        else:
            try:
                # 1. Get Hubs
                hub_df = get_hub_establishments(G, top_n=args.top_n)
                hub_ids_set = set(hub_df['establishment_id'])

                # 2. Calculate Total Outgoing Members
                outflow_data = []
                for node_id in G.nodes():
                    # Use G.nodes[node_id].get('name', node_id) to ensure name is included if available
                    out_degree_weighted = G.out_degree(node_id, weight='weight') # Sum of weights of out-edges
                    outflow_data.append({
                        'establishment_id': node_id,
                        'name': G.nodes[node_id].get('name', node_id), # Get name from graph node attribute
                        'total_members_out_calc': out_degree_weighted
                    })
                total_outflow_df = pd.DataFrame(outflow_data)
                
                # Identify Top N Outflow Establishments
                top_outflow_df = total_outflow_df.sort_values(by='total_members_out_calc', ascending=False).head(args.top_n)
                top_outflow_ids_set = set(top_outflow_df['establishment_id'])

                # 3. Iterate and Build Critical List
                critical_establishments_data = []
                processed_ids = set() # To avoid duplicate entries if a node meets multiple criteria at different stages

                # Merge nodes_df with hub_df and total_outflow_df to have all data in one place
                # Start with nodes_df and left merge others to keep all establishments
                merged_df = nodes_df.copy()
                # Merge hub details (centrality scores)
                merged_df = pd.merge(merged_df, hub_df[['establishment_id', 'betweenness_centrality', 'eigenvector_centrality']], on='establishment_id', how='left')
                # Merge total outflow calculated
                merged_df = pd.merge(merged_df, total_outflow_df[['establishment_id', 'total_members_out_calc']], on='establishment_id', how='left')

                # Fill NaN for centrality and outflow for nodes not in hub_df or total_outflow_df (though total_outflow_df should have all nodes)
                merged_df[['betweenness_centrality', 'eigenvector_centrality', 'total_members_out_calc']] = merged_df[['betweenness_centrality', 'eigenvector_centrality', 'total_members_out_calc']].fillna(0)


                for index, row in merged_df.iterrows():
                    reasons = []
                    node_id = row['establishment_id']
                    
                    is_hub = node_id in hub_ids_set
                    is_large = row.get('size_category') == 'Large'
                    is_high_outflow = node_id in top_outflow_ids_set
                    
                    if is_hub:
                        reasons.append("Hub")
                    if is_large:
                        reasons.append("Large Size")
                    if is_high_outflow:
                        reasons.append("High Outflow")
                    
                    if reasons: # If any criteria met
                        critical_establishments_data.append({
                            'establishment_id': node_id,
                            'name': row.get('name', 'N/A'),
                            'industry': row.get('industry', 'N/A'),
                            'city': row.get('city', 'N/A'),
                            'size_category': row.get('size_category', 'N/A'),
                            'betweenness_centrality': row.get('betweenness_centrality', 0.0),
                            'eigenvector_centrality': row.get('eigenvector_centrality', 0.0),
                            'total_members_out_calc': row.get('total_members_out_calc', 0),
                            'reason_for_criticality': ", ".join(reasons)
                        })
                
                critical_df = pd.DataFrame(critical_establishments_data)
                # Sort by a primary reason, e.g. number of reasons, then by outflow or centrality
                if not critical_df.empty:
                    critical_df['num_reasons'] = critical_df['reason_for_criticality'].apply(lambda x: len(x.split(', ')))
                    critical_df_sorted = critical_df.sort_values(by=['num_reasons', 'total_members_out_calc', 'betweenness_centrality'], ascending=[False, False, False])
                else:
                    critical_df_sorted = critical_df

                print_df(critical_df_sorted, f"Critical Establishments (Criteria: Top {args.top_n} Hub, Large Size, Top {args.top_n} Outflow)")

            except Exception as e:
                print(f"An unexpected error occurred during criticality analysis: {e}")


    print("\nAnalysis complete.")
    # Message about generated files removed as saving is optional and not explicitly done in this script version

if __name__ == "__main__":
    main()
