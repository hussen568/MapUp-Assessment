# Question 1: Reverse List by N Elements

"""
def reverse_list_by_n_elements(lst, n):
    result = []
    length = len(lst)

    # Iterate over the list in steps of n
    for i in range(0, length, n):
        # Get the current group of n elements
        group = []

        # Collect the next n elements (or fewer if at the end)
        for j in range(i, min(i + n, length)):
            group.append(lst[j])

        # Reverse the collected group manually
        for k in range(len(group) - 1, -1, -1):
            result.append(group[k])

    return result


# Example usage:
if __name__ == "__main__":
    print(reverse_list_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8], 3))  # Output: [3, 2, 1, 6, 5, 4, 8, 7]
    print(reverse_list_by_n_elements([1, 2, 3, 4, 5], 2))  # Output: [2, 1, 4, 3, 5]
    print(reverse_list_by_n_elements([10, 20, 30, 40, 50, 60, 70], 4))  # Output: [40, 30, 20, 10, 70, 60, 50]

"""



# Question 2: Lists & Dictionaries

"""
def group_strings_by_length(strings):
    length_dict = {}

    # Group strings by their length
    for string in strings:
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(string)

    # Sort the dictionary by keys (lengths)
    sorted_length_dict = dict(sorted(length_dict.items()))

    return sorted_length_dict


# Example usage:
if __name__ == "__main__":
    print(group_strings_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))
    # Output: {3: ['bat', 'car', 'dog'], 4: ['bear'], 5: ['apple'], 8: ['elephant']}

    print(group_strings_by_length(["one", "two", "three", "four"]))
    # Output: {3: ['one', 'two'], 4: ['four'], 5: ['three']}
"""



# Question 3: Flatten a Nested Dictionary

"""
def flatten_dictionary(nested_dict, parent_key='', sep='.'):
    items = {}

    for k, v in nested_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dictionary(v, new_key, sep=sep))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.update(flatten_dictionary(item, f"{new_key}[{i}]", sep=sep))
                else:
                    items[f"{new_key}[{i}]"] = item
        else:
            items[new_key] = v

    return items


# Example usage:
if __name__ == "__main__":
    input_data = {
        "road": {
            "name": "Highway 1",
            "length": 350,
            "sections": [
                {
                    "id": 1,
                    "condition": {
                        "pavement": "good",
                        "traffic": "moderate"
                    }
                }
            ]
        }
    }

    flattened = flatten_dictionary(input_data)
    print(flattened)
    # Expected Output:
    # {
    #     "road.name": "Highway 1",
    #     "road.length": 350,
    #     "road.sections[0].id": 1,
    #     "road.sections[0].condition.pavement": "good",
    #     "road.sections[0].condition.traffic": "moderate"
    # }
"""



# Question 4: Generate Unique Permutations

"""
from itertools import permutations


def unique_permutations(nums):
    # Use a set to store unique permutations
    return list(map(list, set(permutations(nums))))


# Example usage:
if __name__ == "__main__":
    input_data = [1, 1, 2]
    result = unique_permutations(input_data)

    # Sort the result for consistent output order
    result.sort()

    print(result)
    # Expected Output:
    # [
    #     [1, 1, 2],
    #     [1, 2, 1],
    #     [2, 1, 1]
    # ]

"""



# Question 5: Find All Dates in a Text

"""
import re


def find_all_dates(text):
    # Regular expression patterns for the different date formats
    patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',  # dd-mm-yyyy
        r'\b\d{2}/\d{2}/\d{4}\b',  # mm/dd/yyyy
        r'\b\d{4}\.\d{2}\.\d{2}\b'  # yyyy.mm.dd
    ]

    # Combine all patterns into one
    combined_pattern = '|'.join(patterns)

    # Find all matches in the input text
    return re.findall(combined_pattern, text)


# Example usage:
if __name__ == "__main__":
    text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
    dates = find_all_dates(text)
    print(dates)
    # Expected Output: ["23-08-1994", "08/23/1994", "1994.08.23"]
"""



# Question 6: Decode Polyline, Convert to DataFrame with Distances

"""
pip install polyline pandas

import polyline
import pandas as pd
import numpy as np


def haversine(coord1, coord2):
    # Haversine formula to calculate the distance between two latitude/longitude pairs
    R = 6371000  # Radius of the Earth in meters
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c  # Distance in meters


def decode_polyline(polyline_str):
    # Decode the polyline string to a list of (latitude, longitude) tuples
    coordinates = polyline.decode(polyline_str)

    # Create a DataFrame from the coordinates
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])

    # Calculate distances
    distances = [0]  # First distance is 0
    for i in range(1, len(df)):
        distance = haversine((df.iloc[i - 1]['latitude'], df.iloc[i - 1]['longitude']),
                             (df.iloc[i]['latitude'], df.iloc[i]['longitude']))
        distances.append(distance)

    # Add distances to DataFrame
    df['distance'] = distances

    return df


# Example usage
if __name__ == "__main__":
    polyline_str = "m`~uFjzcv@c@h@y@c@h@Qh@q@w@c@k@k@k@b@A"  # Example polyline
    df = decode_polyline(polyline_str)
    print(df)
"""




# Question 7: Matrix Rotation and Transformation

"""
def rotate_matrix(matrix):
    n = len(matrix)

    # Step 1: Rotate the matrix by 90 degrees clockwise
    rotated_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]

    # Step 2: Replace each element with the sum of its row and column (excluding itself)
    final_matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i])
            col_sum = sum(rotated_matrix[k][j] for k in range(n))
            final_matrix[i][j] = row_sum + col_sum - rotated_matrix[i][j]

    return final_matrix


# Example usage
if __name__ == "__main__":
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    transformed_matrix = rotate_matrix(matrix)
    print(transformed_matrix)

"""


# Question 8: Time Check

"""
import pandas as pd


def check_time_completeness(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Convert start and end times to datetime
    df['start'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])

    # Create a multi-index based on id and id_2
    grouped = df.groupby(['id', 'id_2'])

    # Function to check completeness for each group
    def check_group(group):
        # Get the start and end times for the group
        start_times = group['start'].dt.date.unique()
        end_times = group['end'].dt.date.unique()

        # Check if all 7 days are present
        all_days = pd.date_range(start=start_times.min(), end=start_times.max(), freq='D').date
        has_all_days = len(set(all_days) - set(start_times)) == 0

        # Check if the timestamps cover a full 24-hour period
        full_24_hour_coverage = (group['start'].min() <= group['start'].min().normalize() + pd.Timedelta(hours=0)) and \
                                (group['end'].max() >= group['end'].max().normalize() + pd.Timedelta(hours=23,
                                                                                                     minutes=59,
                                                                                                     seconds=59))

        return has_all_days and full_24_hour_coverage

    # Apply the check to each group and return a boolean series
    result = grouped.apply(check_group)
    result.index = pd.MultiIndex.from_tuples(result.index)

    return result


# Example usage
if __name__ == "__main__":
    file_path = 'dataset-1.csv'  # Path to your CSV file
    completeness_result = check_time_completeness(file_path)
    print(completeness_result)

"""

