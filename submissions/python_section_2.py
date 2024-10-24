import pandas as pd
from datetime import time, timedelta


def calculate_distance_matrix_from_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    
    distance_dict = {}
    
    for _, row in df.iterrows():
        start = row['id_start']
        end = row['id_end']
        distance = row['distance']
        
        if start not in distance_dict:
            distance_dict[start] = {}
        if end not in distance_dict:
            distance_dict[end] = {}
        
        if end in distance_dict[start]:
            distance_dict[start][end] = min(distance_dict[start][end], distance)
            distance_dict[end][start] = min(distance_dict[end][start], distance)
        else:
            distance_dict[start][end] = distance
            distance_dict[end][start] = distance
    
    ids = list(distance_dict.keys())
    matrix_size = len(ids)
    
    distance_matrix = pd.DataFrame(0, index=ids, columns=ids)
    
    for start in ids:
        for end in ids:
            if start != end and end in distance_dict[start]:
                cumulative_distance = 0
                visited = set()
                stack = [(start, cumulative_distance)]
                
                while stack:
                    current, dist = stack.pop()
                    if current == end:
                        distance_matrix.at[start, end] = dist
                        break
                    visited.add(current)
                    for neighbor in distance_dict[current]:
                        if neighbor not in visited:
                            stack.append((neighbor, dist + distance_dict[current][neighbor]))
    
    return distance_matrix

#Example:
# result_matrix = calculate_distance_matrix_from_csv('dataset-2.csv')
# Write your logic here
# The code reads a dataset containing distances between toll locations identified by unique IDs. It constructs a dictionary to store these distances, ensuring that each distance is recorded in both directions to maintain symmetry. After populating the dictionary, the code creates a DataFrame where each ID is represented as both a row and a column. The diagonal entries are set to zero, indicating no distance from a location to itself. For any two different IDs, the code calculates cumulative distances, allowing for indirect routes if necessary. This results in a complete distance matrix that reflects all known distances between the toll locations, making it easy to analyze travel distances across the network.

   


def unroll_distance_matrix(distance_matrix: pd.DataFrame) -> pd.DataFrame:
    unrolled_data = []

    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            if id_start != id_end:  
                distance = distance_matrix.at[id_start, id_end]
                unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})
    # Write your logic here
    # The unroll_distance_matrix function converts a square distance matrix into a long-format DataFrame. It iterates through each unique pair of IDs, excluding pairs where the IDs are the same. For each valid pair, it retrieves the distance from the matrix and stores this information in a list. Finally, this list is transformed into a new DataFrame with columns id_start, id_end, and distance, making it easier to analyze distances between different toll locations.
    return pd.DataFrame(unrolled_data)
#Example:
#result_df = unroll_distance_matrix(distance_matrix)



def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: int) -> list:
    avg_distance = df[df['id_start'] == reference_id]['distance'].mean()
    
    if np.isnan(avg_distance):
        return []

    lower_bound = avg_distance * 0.9
    upper_bound = avg_distance * 1.1

    ids_within_threshold = df[(df['distance'] >= lower_bound) & (df['distance'] <= upper_bound)]['id_start']

    return sorted(ids_within_threshold.unique())
# Write your logic here
# The function find_ids_within_ten_percentage_threshold calculates the average distance for a specified reference ID from a DataFrame containing distances between toll locations. It determines the range of distances that fall within 10% above and below this average. The function then filters the DataFrame to find all unique IDs from the id_start column that have distances within this range, returning them as a sorted list. This allows users to easily identify toll locations that are similar in distance to the reference location.
# Example: 
# df = pd.read_csv('unrolled_distance_matrix.csv')
# result_ids = find_ids_within_ten_percentage_threshold(df, 1001400)





def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    moto_rate = 0.8
    car_rate = 1.2
    rv_rate = 1.5
    bus_rate = 2.2
    truck_rate = 3.6

    df['moto'] = df['distance'] * moto_rate
    df['car'] = df['distance'] * car_rate
    df['rv'] = df['distance'] * rv_rate
    df['bus'] = df['distance'] * bus_rate
    df['truck'] = df['distance'] * truck_rate

    return df


# Wrie your logic here
# The calculate_toll_rate function computes toll rates for various vehicle types based on distances in a DataFrame. It multiplies the distance by specific coefficients for each vehicle type: 0.8 for motorcycles, 1.2 for cars, 1.5 for RVs, 2.2 for buses, and 3.6 for trucks. The function adds five new columns to the DataFrame, each representing the toll rate for a vehicle type, resulting in a comprehensive view of toll costs associated with different vehicles and distances.
   






def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    time_ranges = [
        (time(0, 0), time(10, 0), 0.8),   # Weekdays: 00:00 to 10:00
        (time(10, 0), time(18, 0), 1.2),  # Weekdays: 10:00 to 18:00
        (time(18, 0), time(23, 59, 59), 0.8) # Weekdays: 18:00 to 23:59
    ]
    
    for day in days:
        for start_time, end_time, discount in time_ranges:
            for index, row in df.iterrows():
                if row['start_day'] in days[:5]:  # Weekdays
                    df.at[index, 'moto'] *= discount
                    df.at[index, 'car'] *= discount
                    df.at[index, 'rv'] *= discount
                    df.at[index, 'bus'] *= discount
                    df.at[index, 'truck'] *= discount

                else:  # Weekends
                    df.at[index, 'moto'] *= 0.7
                    df.at[index, 'car'] *= 0.7
                    df.at[index, 'rv'] *= 0.7
                    df.at[index, 'bus'] *= 0.7
                    df.at[index, 'truck'] *= 0.7

                # Add start and end day/time columns
                df.at[index, 'start_day'] = day
                df.at[index, 'start_time'] = start_time
                df.at[index, 'end_day'] = day
                df.at[index, 'end_time'] = end_time

    return df

# Example:
# result_df = calculate_time_based_toll_rates(df)
# Write your logic here
# The calculate_time_based_toll_rates function adjusts toll rates based on time intervals throughout the week. It applies specific discount factors for weekdays and weekends: weekdays have varying discounts depending on the time of day, while weekends receive a constant discount. The function adds columns for start and end days and times, covering a full 24-hour period across all seven days for each unique (id_start, id_end) pair, allowing for comprehensive toll cost analysis influenced by time.
    

