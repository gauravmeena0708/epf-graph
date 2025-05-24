import pandas as pd
import numpy as np

def calculate_establishment_flow_features(nodes_df, edges_df, num_recent_months=6, high_demand_threshold_quantile=0.8):
    """
    Calculates establishment flow features and labels high-demand areas.

    Args:
        nodes_df (pd.DataFrame): DataFrame with establishment data (must include 'establishment_id').
        edges_df (pd.DataFrame): DataFrame with transfer data (must include 
                                 'source_establishment_id', 'target_establishment_id', 
                                 'members_transferred', 'month').
        num_recent_months (int): Number of recent months to consider for flow calculation.
        high_demand_threshold_quantile (float): Quantile to determine high net inflow for labeling.

    Returns:
        pd.DataFrame: Augmented nodes_df with new flow features and 'is_high_future_demand' label.
    """
    if 'month' not in edges_df.columns:
        raise ValueError("edges_df must contain a 'month' column.")
    if edges_df['month'].empty:
        raise ValueError("edges_df 'month' column is empty or contains no valid date data.")

    # Convert month to datetime if it's not already, to ensure proper sorting and filtering
    # Assuming month is in a format that can be converted, e.g., YYYY-MM or YYYYMM
    # For this example, let's assume it's sortable as is (e.g., integer YYYYMM or string YYYY-MM)
    # A more robust solution would parse it explicitly: pd.to_datetime(edges_df['month'], format='%Y%m')
    
    # Determine the most recent month
    # Ensure 'month' is treated numerically or lexicographically correctly for max()
    # If 'month' is like '2023-01', string comparison works. If YYYYMM, integer comparison works.
    try:
        # Attempt to convert to numeric if possible, to handle cases like YYYYMM
        edges_df['month_numeric'] = pd.to_numeric(edges_df['month'])
        most_recent_month_val = edges_df['month_numeric'].max()
        unique_months_sorted = sorted(edges_df['month_numeric'].unique())
    except ValueError:
        # Fallback if 'month' is not purely numeric (e.g., 'YYYY-MM')
        edges_df['month_numeric'] = edges_df['month'] # Keep as is if conversion fails
        most_recent_month_val = edges_df['month_numeric'].max()
        unique_months_sorted = sorted(edges_df['month_numeric'].unique())

    if pd.isna(most_recent_month_val):
        raise ValueError("Could not determine the most recent month from edges_df.")

    # Determine the start month for the recent period
    if len(unique_months_sorted) < num_recent_months:
        start_month_val = unique_months_sorted[0]
        print(f"Warning: Data has fewer than {num_recent_months} unique months. Using all available {len(unique_months_sorted)} months.")
    else:
        start_month_val = unique_months_sorted[-num_recent_months]

    # Filter edges_df for the recent period
    recent_edges_df = edges_df[edges_df['month_numeric'] >= start_month_val]

    # Calculate total members in (inflow) for each establishment in the recent period
    recent_inflow = recent_edges_df.groupby('target_establishment_id')['members_transferred'].sum().reset_index()
    recent_inflow.rename(columns={'members_transferred': 'total_members_in_recent', 
                                  'target_establishment_id': 'establishment_id'}, inplace=True)

    # Calculate total members out (outflow) for each establishment in the recent period
    recent_outflow = recent_edges_df.groupby('source_establishment_id')['members_transferred'].sum().reset_index()
    recent_outflow.rename(columns={'members_transferred': 'total_members_out_recent',
                                   'source_establishment_id': 'establishment_id'}, inplace=True)

    # Merge inflow and outflow with nodes_df
    nodes_df = pd.merge(nodes_df, recent_inflow, on='establishment_id', how='left')
    nodes_df = pd.merge(nodes_df, recent_outflow, on='establishment_id', how='left')

    # Fill NaNs with 0 for establishments with no recent activity
    nodes_df['total_members_in_recent'].fillna(0, inplace=True)
    nodes_df['total_members_out_recent'].fillna(0, inplace=True)

    # Calculate net inflow
    nodes_df['net_inflow_recent'] = nodes_df['total_members_in_recent'] - nodes_df['total_members_out_recent']
    
    # Calculate average monthly net inflow
    # Adjust num_recent_months if data has fewer unique months
    actual_num_months = len(unique_months_sorted) if len(unique_months_sorted) < num_recent_months else num_recent_months
    if actual_num_months == 0: # Avoid division by zero if no months are processed
        nodes_df['avg_monthly_net_inflow_recent'] = 0.0
    else:
        nodes_df['avg_monthly_net_inflow_recent'] = nodes_df['net_inflow_recent'] / actual_num_months


    # Define 'is_high_future_demand'
    # Consider only establishments with some activity or non-zero net inflow for quantile calculation
    active_establishments_net_inflow = nodes_df[nodes_df['net_inflow_recent'] != 0]['net_inflow_recent']
    
    if not active_establishments_net_inflow.empty:
        high_demand_threshold = active_establishments_net_inflow.quantile(high_demand_threshold_quantile)
        nodes_df['is_high_future_demand'] = nodes_df['net_inflow_recent'] >= high_demand_threshold
    else:
        # If no establishments have activity, or all net_inflows are 0, then none are high demand by this definition
        nodes_df['is_high_future_demand'] = False
        print("Warning: No establishments with non-zero net inflow found. 'is_high_future_demand' is False for all.")

    # Clean up temporary numeric month column if it was different from original
    if 'month_numeric' in edges_df.columns and edges_df['month_numeric'].dtype != edges_df['month'].dtype :
        edges_df.drop(columns=['month_numeric'], inplace=True)
        
    return nodes_df


if __name__ == '__main__':
    # --- Example Usage ---

    # Sample nodes_df
    sample_nodes_data = {
        'establishment_id': [f'E{i:03d}' for i in range(1, 11)],
        'name': [f'Company {chr(64+i)}' for i in range(1, 11)],
        'industry': ['Tech', 'Finance', 'Healthcare', 'Tech', 'Retail', 'Finance', 'Healthcare', 'Retail', 'Tech', 'Finance'] * 1,
        'city': ['CityA', 'CityB', 'CityA', 'CityC', 'CityB', 'CityC', 'CityA', 'CityB', 'CityC', 'CityA'] * 1,
        'size_category': ['Medium', 'Large', 'Small', 'Medium', 'Large', 'Small', 'Medium', 'Large', 'Small', 'Medium'] * 1
    }
    nodes_df_sample = pd.DataFrame(sample_nodes_data)

    # Sample edges_df (representing transfers over several months)
    # Using YYYYMM integer format for 'month'
    sample_edges_data = {
        'source_establishment_id': ['E001', 'E002', 'E001', 'E003', 'E004', 'E005', 'E002', 'E006', 'E007', 'E001', 'E008', 'E009', 'E010', 'E003', 'E005'],
        'target_establishment_id': ['E002', 'E003', 'E004', 'E001', 'E005', 'E006', 'E001', 'E007', 'E008', 'E010', 'E009', 'E001', 'E002', 'E004', 'E001'],
        'members_transferred': [10, 5, 8, 12, 15, 3, 7, 9, 6, 11, 4, 10, 8, 6, 9],
        'month': [
            202301, 202301, 202302, 202302, 202303, 202303, 
            202304, 202304, 202305, 202305, 202306, 202306,
            202307, 202307, 202308 # Most recent month
        ] 
    }
    edges_df_sample = pd.DataFrame(sample_edges_data)
    
    print("--- Initial Sample Data ---")
    print("Nodes DataFrame:")
    print(nodes_df_sample.head())
    print("\nEdges DataFrame:")
    print(edges_df_sample.head())

    # Parameters for the function
    num_recent_months_param = 3 # Consider last 3 months (June, July, August 2023)
    quantile_param = 0.75 # Top 25% net inflow considered high demand

    print(f"\n--- Calculating Flow Features (last {num_recent_months_param} months, {quantile_param*100}th percentile threshold) ---")
    
    # Call the function
    augmented_nodes_df = calculate_establishment_flow_features(
        nodes_df_sample.copy(), # Use a copy to avoid modifying the original sample df
        edges_df_sample.copy(), 
        num_recent_months=num_recent_months_param,
        high_demand_threshold_quantile=quantile_param
    )

    print("\n--- Augmented Nodes DataFrame (Head) ---")
    print(augmented_nodes_df.head(10))

    print("\n--- Feature Details ---")
    print(augmented_nodes_df[['establishment_id', 'total_members_in_recent', 'total_members_out_recent', 'net_inflow_recent', 'avg_monthly_net_inflow_recent', 'is_high_future_demand']])

    print("\n--- Counts of 'is_high_future_demand' ---")
    print(augmented_nodes_df['is_high_future_demand'].value_counts())

    # Example with fewer months in data than num_recent_months
    print("\n--- Example: Requesting more months than available ---")
    edges_short_df_sample = edges_df_sample[edges_df_sample['month'] >= 202307] # Only July, August
    augmented_nodes_short_df = calculate_establishment_flow_features(
        nodes_df_sample.copy(), 
        edges_short_df_sample.copy(),
        num_recent_months=6, # Request 6 months
        high_demand_threshold_quantile=quantile_param
    )
    print(augmented_nodes_short_df[['establishment_id', 'net_inflow_recent', 'avg_monthly_net_inflow_recent', 'is_high_future_demand']])
    print(augmented_nodes_short_df['is_high_future_demand'].value_counts())

    # Example with no activity for some nodes
    print("\n--- Example: Nodes with no activity ---")
    nodes_with_inactive_sample = pd.DataFrame({
        'establishment_id': [f'E{i:03d}' for i in range(1, 13)], # E011, E012 are new
        'name': [f'Company {chr(64+i)}' for i in range(1, 13)],
    })
    augmented_nodes_inactive_df = calculate_establishment_flow_features(
        nodes_with_inactive_sample.copy(), 
        edges_df_sample.copy(), # original edges, E011, E012 won't be in them
        num_recent_months=num_recent_months_param,
        high_demand_threshold_quantile=quantile_param
    )
    print(augmented_nodes_inactive_df[['establishment_id', 'net_inflow_recent', 'is_high_future_demand']].tail())

    # Example with no edges at all
    print("\n--- Example: No edges data ---")
    empty_edges_df = pd.DataFrame(columns=['source_establishment_id', 'target_establishment_id', 'members_transferred', 'month'])
    try:
        augmented_empty_edges_df = calculate_establishment_flow_features(
            nodes_df_sample.copy(), 
            empty_edges_df.copy(),
            num_recent_months=3
        )
        print(augmented_empty_edges_df[['establishment_id', 'net_inflow_recent', 'is_high_future_demand']])
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # Example where all net inflows are zero
    print("\n--- Example: All net inflows are zero ---")
    zero_flow_edges_data = {
        'source_establishment_id': ['E001', 'E002'],
        'target_establishment_id': ['E002', 'E001'],
        'members_transferred': [10, 10], # Net zero flow for E001 and E002
        'month': [202301, 202301] 
    }
    zero_flow_edges_df = pd.DataFrame(zero_flow_edges_data)
    augmented_zero_flow_df = calculate_establishment_flow_features(
        nodes_df_sample.copy(), 
        zero_flow_edges_df.copy(),
        num_recent_months=1
    )
    print(augmented_zero_flow_df[['establishment_id', 'net_inflow_recent', 'is_high_future_demand']])
    print(augmented_zero_flow_df['is_high_future_demand'].value_counts())

