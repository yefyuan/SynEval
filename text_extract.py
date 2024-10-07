import pandas as pd

# Load data from the CSV file
data = pd.read_csv('feeding.csv')

# Select only the 'text' column
reviews = data['text']


# Define the path for the new text file
output_file_path = 'reviews_example.txt'

# Write each review to a new line in the text file
with open(output_file_path, 'w', encoding='utf-8') as file:
    for review in reviews:
        file.write(review + '\n')
