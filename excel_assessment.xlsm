# Question 9: Distance Matrix Calculation

"""
import pandas as pd
import numpy as np

def calculate_distance_matrix(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Extract unique IDs and create a distance matrix
    unique_ids = pd.unique(df[['id_start', 'id_end']].values.ravel('K'))
    distance_matrix = pd.DataFrame(np.inf, index=unique_ids, columns=unique_ids)

    # Set diagonal to 0 (distance from a location to itself)
    np.fill_diagonal(distance_matrix.values, 0)

    # Populate the distance matrix with known distances
    for _, row in df.iterrows():
        distance_matrix.loc[row['id_start'], row['id_end']] = row['distance']
        distance_matrix.loc[row['id_end'], row['id_start']] = row['distance']  # Ensure symmetry

    # Floyd-Warshall algorithm for calculating cumulative distances
    for k in unique_ids:
        for i in unique_ids:
            for j in unique_ids:
                if distance_matrix.loc[i, k] + distance_matrix.loc[k, j] < distance_matrix.loc[i, j]:
                    distance_matrix.loc[i, j] = distance_matrix.loc[i, k] + distance_matrix.loc[k, j]

    return distance_matrix

# Example usage
if __name__ == "__main__":
    file_path = 'dataset-2.csv'  # Path to your CSV file
    distance_df = calculate_distance_matrix(file_path)
    print(distance_df)
"""



# Question 10: Unroll Distance Matrix


"""
import pandas as pd

def unroll_distance_matrix(distance_matrix):
    # Create an empty list to store the rows
    rows = []

    # Iterate through the distance matrix
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            if id_start != id_end:  # Exclude same id_start to id_end
                distance = distance_matrix.loc[id_start, id_end]
                rows.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})

    # Convert the list of rows to a DataFrame
    result_df = pd.DataFrame(rows)

    return result_df

# Example usage
if __name__ == "__main__":
    # Assuming distance_df is the DataFrame from Question 9
    distance_df = pd.DataFrame({
        'id_start': ['A', 'A', 'B', 'C'],
        'id_end': ['B', 'C', 'A', 'A'],
        'distance': [5, 10, 5, 10]
    }).set_index('id_start')  # Example matrix for demonstration

    # Unroll the distance matrix
    unrolled_df = unroll_distance_matrix(distance_df)
    print(unrolled_df)
"""


# Question 11: Finding IDs within Percentage Threshold


"""
import pandas as pd


def find_ids_within_ten_percentage_threshold(df, reference_id):
    # Filter the DataFrame for rows where id_start matches the reference_id
    filtered_df = df[df['id_start'] == reference_id]

    # Check if there are any distances for the reference_id
    if filtered_df.empty:
        return []

    # Calculate the average distance for the reference_id
    average_distance = filtered_df['distance'].mean()

    # Calculate the lower and upper bounds for the 10% threshold
    lower_bound = average_distance * 0.9
    upper_bound = average_distance * 1.1

    # Find all ids that fall within the 10% threshold
    within_threshold = df[(df['distance'] >= lower_bound) & (df['distance'] <= upper_bound)]

    # Get unique id_start values and sort them
    result_ids = sorted(within_threshold['id_start'].unique())

    return result_ids


# Example usage
if __name__ == "__main__":
    # Sample DataFrame
    data = {
        'id_start': [1, 1, 2, 2, 3],
        'id_end': [2, 3, 1, 3, 1],
        'distance': [10, 15, 10, 20, 30]
    }

    df = pd.DataFrame(data)

    # Finding IDs within 10% of the average distance for id_start = 1
    result = find_ids_within_ten_percentage_threshold(df, 1)
    print(result)
"""




# Question 12: Calculate Toll Rate

"""
import pandas as pd

def calculate_toll_rate(distance_df):
    
    Calculate toll rates based on vehicle types and add them as new columns to the DataFrame.

    Parameters:
    distance_df (pd.DataFrame): Input DataFrame containing distances between IDs.

    Returns:
    pd.DataFrame: Updated DataFrame with toll rate columns added.
    
    # Define the rate coefficients for each vehicle type
    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Calculate toll rates and add new columns to the DataFrame
    for vehicle, rate in rates.items():
        distance_df[vehicle] = distance_df['distance'] * rate

    return distance_df

# Example usage:
Assuming 'distance_matrix' is the DataFrame created from Question 10
distance_matrix = calculate_distance_matrix(dataset)  # This would be your previous function
toll_rates_df = calculate_toll_rate(distance_matrix)
print(toll_rates_df)
"""



# Question 13: Calculate Time-Based Toll Rates

"""
import pandas as pd
from datetime import time


def calculate_time_based_toll_rates(toll_rates_df):
    
    Calculate time-based toll rates and add relevant columns to the DataFrame.

    Parameters:
    toll_rates_df (pd.DataFrame): Input DataFrame containing toll rates for different vehicle types.

    Returns:
    pd.DataFrame: Updated DataFrame with time-based toll rates and additional time columns.
    
    # Define day names and corresponding time intervals
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Prepare lists to hold new columns
    start_days = []
    start_times = []
    end_days = []
    end_times = []

    # Iterate over each row to calculate time-based rates
    for index, row in toll_rates_df.iterrows():
        # Append day and time values based on the index
        day_index = index % 7  # Cycle through the week
        start_day = days[day_index]
        end_day = days[day_index]
        start_time = time(0, 0)  # Start at 00:00:00
        end_time = time(23, 59, 59)  # End at 23:59:59

        # Store the new values
        start_days.append(start_day)
        end_days.append(end_day)
        start_times.append(start_time)
        end_times.append(end_time)

        # Modify vehicle rates based on time intervals
        if day_index < 5:  # Weekdays (Monday - Friday)
            for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
                if row['distance'] <= 10:  # Example condition, adjust as needed
                    row[vehicle] *= 0.8  # Apply discount factor
                elif 10 < row['distance'] <= 18:
                    row[vehicle] *= 1.2  # Apply discount factor
                else:
                    row[vehicle] *= 0.8  # Apply discount factor
        else:  # Weekends (Saturday and Sunday)
            for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
                row[vehicle] *= 0.7  # Apply constant discount factor

    # Add new columns to the DataFrame
    toll_rates_df['start_day'] = start_days
    toll_rates_df['end_day'] = end_days
    toll_rates_df['start_time'] = start_times
    toll_rates_df['end_time'] = end_times

    return toll_rates_df

# Example usage:
toll_rates_df = calculate_toll_rate(distance_matrix)  # This would be from Question 12
time_based_toll_df = calculate_time_based_toll_rates(toll_rates_df)
print(time_based_toll_df)

"""
