import argparse
import os
import csv
import re


def parse_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    # Extract the relevant information
    header = lines[0].strip().split(',')
    data = lines[1].strip().split(',')
    
    # Create a dictionary to easily extract fields
    record = dict(zip(header, data))
    
    # Extract Benchmark, Program, and Optimizer from the filename
    filename = os.path.basename(file_path)
    parts = filename.split('_')
    benchmark = parts[0]
    program = '_'.join(parts[1:-1])  # Everything except the optimizer part
    optimizer = parts[-1].split('.')[0]  # Last part before .txt
    
    # Determine # train from benchmark
    train_match = re.search(r'\d+$', benchmark)
    num_train = train_match.group() if train_match else 15

    # Determine demonstrations
    demonstrations = 8 if optimizer.endswith('D8') else 2

    # Check if 'Typed' is in the program name
    contains_typed = 'Typed' in program

    # Get the score
    score = record.get('score', 'N/A')
    
    return {
        'Benchmark': benchmark,
        'Program': program,
        'Optimizer': optimizer,
        'score': score,
        'demonstrations': demonstrations,
        '# train': num_train,
        'contains_typed': contains_typed
    }


def main(directory):
    # Prepare output CSV
    output_file = os.path.join(directory, 'output.csv')
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['Benchmark', 'Program', 'Optimizer', 'score', 'demonstrations', '# train', 'contains_typed']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Loop through txt files
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory, filename)
                record = parse_file(file_path)
                writer.writerow(record)

    print(f'CSV file created at {output_file}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process text files and extract benchmark data.")
    parser.add_argument('directory', type=str, help='Path to the directory containing .txt files')
    args = parser.parse_args()

    main(args.directory)

# Example usage:
# python utils/analyze.py ./evaluation_llama3170b/HeartDisease