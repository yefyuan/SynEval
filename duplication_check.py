import csv

# Function to check for duplicates and remove them from the CSV file
def remove_duplicates(file_path):
    # Read the CSV file
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        data = list(reader)

    # Check for duplicates and remove them
    duplicates = []
    unique_data = []

    for row in data:
        if row in unique_data:
            duplicates.append(row)
        else:
            unique_data.append(row)

    # Calculate the total number of duplicates
    total_duplicates = len(duplicates)

    # Print the total number of duplicates
    print(f"Total duplicates found: {total_duplicates}")

    # Write the unique data back to the CSV file
    with open(file_path, 'w', newline='') as file:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unique_data)

    print("Duplicates removed from the CSV file.")

# Specify the path to your CSV file
csv_file_path = 'feeding.csv'

# Call the function to remove duplicates
remove_duplicates(csv_file_path)