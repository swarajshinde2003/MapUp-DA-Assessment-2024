from typing import Dict, List
import re
import pandas as pd
import polyline
import numpy as np
from math import radians, sin, cos, sqrt, atan2


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    result = []
    length = len(lst)
    
    for i in range(0, length, n): 
        group = []
        for j in range(min(n, length - i)):  
            group.append(lst[i + j])
        for k in range(len(group)-1, -1, -1):  
            result.append(group[k])
    return result

#Input given by user 
lst = list(map(int, input("Enter the list elements separated by spaces:- ").split()))
n = int(input("Enter the value of n:- "))

print(reverse_by_n_elements(lst, n))




def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    result = {}
    for string in lst:
        length = len(string)
        if length not in result:
            result[length] = []
        result[length].append(string)
    
    return dict(sorted(result.items()))  

#Input given by user 
lst = input("Enter a list of strings separated by spaces: ").split()
output = group_by_length(lst)
print(output)


def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    def _flatten_dict(d, parent_key=''):
        flattened = {}
        
        for key, value in d.items():
            new_key = parent_key + sep + key if parent_key else key
            if isinstance(value, dict):
                flattened.update(_flatten_dict(value, new_key))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    flattened.update(_flatten_dict(item, f"{new_key}[{i}]"))
            else:
                flattened[new_key] = value
        
        return flattened
    return _flatten_dict(nested_dict)

#Example
nested_dict = {
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

flattened_dict = flatten_dict(nested_dict)
print(flattened_dict)


def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    def backtrack(start: int):
        if start == len(nums):
            result.append(nums[:])
            return
        seen=set()
        for i in range(start, len(nums)):
            if nums[i] in seen:
                continue
            seen.add(nums[i])
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)
            nums[start], nums[i] = nums[i], nums[start]
    
    nums.sort()
    result = []
    backtrack(0)
    return result
#Input given by user
input_list = input("Enter a list of integers: ")
input_list = [int(x.strip()) for x in input_list.split() if x.strip().isdigit()]
output = unique_permutations(input_list)
print(output)


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    pattern = r'\b(\d{2}-\d{2}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}\.\d{2}\.\d{2})\b'
    dates = re.findall(pattern, text)
    return dates

#Input given by user
input_text = input("Enter a string containing dates: ")
output_dates = find_all_dates(input_text)
print(output_dates)



def haversine(lat1, lon1, lat2, lon2):
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    R = 6371000 
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c
def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    decoded_coords = polyline.decode(polyline_str)
    df = pd.DataFrame(decoded_coords, columns=['latitude', 'longitude'])
    df['distance'] = 0.0
    
    for i in range(1, len(df)):
        lat1, lon1 = df.loc[i - 1, 'latitude'], df.loc[i - 1, 'longitude']
        lat2, lon2 = df.loc[i, 'latitude'], df.loc[i, 'longitude']
        df.loc[i, 'distance'] = haversine(lat1, lon1, lat2, lon2)
    
    return df

#Example
polyline_str = '_p~iF~ps|U_ulLnnqC_mqNvxq`@'
df = polyline_to_dataframe(polyline_str)
print(df)




def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    n = len(matrix)
    rotated_matrix = list(zip(*matrix[::-1]))
    rotated_matrix = [list(row) for row in rotated_matrix]
    result = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[x][j] for x in range(n)) - rotated_matrix[i][j]
            result[i][j] = row_sum + col_sum
    
    return result

matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
output = rotate_and_multiply_matrix(matrix)
for row in output:
    print(row)
    


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    df['start'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])
    df.set_index(['id', 'id_2'], inplace=True)
    grouped = df.groupby(level=[0, 1]).agg({'start': ['min', 'max'], 'end': ['min', 'max']})
    incorrect_timestamps = pd.Series(index=grouped.index, dtype=bool)

    for (id_val, id_2_val), group in grouped.iterrows():
        start_min = group[('start', 'min')]
        end_max = group[('end', 'max')]
        week_span = (end_max - start_min).days >= 6
        day_start = start_min.replace(hour=0, minute=0, second=0)
        day_end = start_min.replace(hour=23, minute=59, second=59)
        full_day_coverage = any((group['start'] <= day_end) & (group['end'] >= day_start))
        incorrect_timestamps[(id_val, id_2_val)] = not (week_span and full_day_coverage)

    return incorrect_timestamps
