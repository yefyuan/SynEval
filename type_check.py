#!/usr/bin/env python3

import pandas as pd
import numpy as np

def determine_data_type(column):
	if pd.api.types.is_numeric_dtype(column):
		return 'Numerical Data'
	elif pd.api.types.is_bool_dtype(column) or len(column.unique()) < 20:
		return 'Categorical Data'
	else:
		return 'Text Data'
	
def classify_columns(file_path):
	# Read the CSV file
	df = pd.read_csv("feeding.csv")
	
	# Create a dictionary to hold column data types
	column_types = {}
	
	# Iterate over each column and determine its type
	for column in df.columns:
		column_types[column] = determine_data_type(df[column])
		
	return column_types

# Example usage
file_path = 'your_file.csv'  # Replace with your actual file path
column_types = classify_columns(file_path)

# Print the result
for column, dtype in column_types.items():
	print(f"Column '{column}' is of type: {dtype}")