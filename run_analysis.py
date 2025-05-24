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
    get_bridge_establishments
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

    args = parser.parse_args()
    
    # Configure pandas display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000) # Adjust width for console
    pd.set_option('display.colheader_justify', 'left')


    print("Starting EPFO Analysis Orchestrator...")

    G = None
    nodes_df_for_info = None # To store nodes_df for printing sample info

    if args.data_source == "sample":
        print("Using sample data.")
        establishments_data_dict, transfers_data_list, nodes_df_for_info, _ = get_sample_data()
        G = create_graph_from_datasets(establishments_data_dict, transfers_data_list)
    elif args.data_source == "csv":
        print(f"Loading data from CSV files: {args.nodes_csv}, {args.edges_csv}")
        try:
            G, nodes_df_for_info, _ = load_graph_data(nodes_filepath=args.nodes_csv, edges_filepath=args.edges_csv)
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
    if nodes_df_for_info is not None:
        print_df(nodes_df_for_info.head(), "Sample Node Data (First 5 Rows from loaded nodes_df)")


    if args.run_all_insights:
        args.run_establishment_insights = True
        args.run_transfer_insights = True
        args.run_network_insights = True

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

    print("\nAnalysis complete.")
    # Message about generated files removed as saving is optional and not explicitly done in this script version

if __name__ == "__main__":
    main()
