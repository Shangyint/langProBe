import math
from matplotlib.patches import Patch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import pathlib


def extract_information_from_files(directory_path):
    # Define the path to the directory
    file_path = pathlib.Path(directory_path)
    
    # List all .txt files in the directory
    all_result_files = list(file_path.rglob("*.txt"))
    
    # Initialize a list to store the extracted data
    extracted_data = []
    
    # Process each file
    for file in all_result_files:
        # Split the filename to get benchmark, program, and optimizer
        file_name_parts = file.stem.split("_")
        if len(file_name_parts) >= 3:
            benchmark = file_name_parts[0]
            program = file_name_parts[1]
            optimizer = file_name_parts[2]
        else:
            raise ValueError(f"Invalid file name: {file.name}")
        
        with open(file, 'r') as f:
            lines = f.readlines()
            
            # Extract information from the lines
            if len(lines) == 2:  # Checking if we have 2 lines
                header = lines[0].strip()
                values = lines[1].strip().split(",")
                
                # Check if optimizer is present in the file content
                if "optimizer" in header:
                    # Extract values for file with optimizer
                    data = {
                        "file_name": file.name,
                        "benchmark": benchmark,
                        "program": program,
                        "optimizer": optimizer,
                        "score": float(values[0]),
                        "cost": float(values[1]),
                        "input_tokens": int(values[2]),
                        "output_tokens": int(values[3]),
                        "optimizer_cost": float(values[5]),
                        "optimizer_input_tokens": int(values[6]),
                        "optimizer_output_tokens": int(values[7]),
                    }
                else:
                    # Extract values for file without optimizer
                    data = {
                        "file_name": file.name,
                        "benchmark": benchmark,
                        "program": program,
                        "optimizer": optimizer,
                        "score": float(values[0]),
                        "cost": float(values[1]),
                        "input_tokens": int(values[2]),
                        "output_tokens": int(values[3]),
                        "optimizer_cost": 0.0,
                        "optimizer_input_tokens": 0,
                        "optimizer_output_tokens": 0,
                    }
                
                # Append the extracted data to the list
                extracted_data.append(data)
    
    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(extracted_data)
    df['optimizer'].replace("None", "Baseline")
    return df

def plot_scores_by_benchmark(directory_path: str, data_df):
    # Get a list of unique benchmarks
    benchmarks = data_df['benchmark'].unique()
    num_benchmarks = len(benchmarks)
    
    # Determine the number of rows and columns based on the square root of the number of benchmarks
    cols = math.ceil(math.sqrt(num_benchmarks))
    rows = math.ceil(num_benchmarks / cols)
    
    # Set up the figure with subplots arranged in a grid
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), sharey=True)
    fig.suptitle("Scores by Program and Optimizer for Each Benchmark", fontsize=16)
    
    # Flatten axes array for easy iteration
    axes = axes.flatten()
    
    # Create subplots for each benchmark
    for i, benchmark in enumerate(benchmarks):
        ax = axes[i]
        benchmark_data = data_df[data_df['benchmark'] == benchmark]
        
        # Create a bar plot for each benchmark
        sns.barplot(
            data=benchmark_data,
            x='program', y='score', hue='optimizer', ax=ax,
            palette="viridis"
        )
        
        # Set labels and title for each subplot
        ax.set_title(f"Benchmark: {benchmark}")
        ax.set_xlabel("Program")
        ax.set_ylabel("Score")

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.get_legend().remove()  # Remove individual legends
    
    # Hide any unused subplots
    for j in range(i + 1, rows * cols):
        fig.delaxes(axes[j])

    # create a handle for unique optimizers
    unique_optimizers = data_df['optimizer'].unique()
    legend_handles = [Patch(color=sns.color_palette("viridis", len(unique_optimizers))[i], label=optimizer) for i, optimizer in enumerate(unique_optimizers)]
    fig.legend(title="Optimizer", handles=legend_handles, loc='upper left')
    # Adjust layout and add legend
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # To leave space for the suptitle
    
    plt.savefig(f"{directory_path}_scores_by_benchmark.png", dpi=300)

def plot_percentage_gain_by_benchmark(directory_path, data_df):
    # Replace "None" with "Baseline" in the optimizer column
    data_df['optimizer'] = data_df['optimizer'].replace("None", "Baseline")
    
    # Calculate the baseline score for each benchmark and program
    baseline_df = data_df[data_df['optimizer'] == "Baseline"].rename(columns={"score": "baseline_score"})
    baseline_df = baseline_df[['benchmark', 'program', 'baseline_score']]
    
    # Merge the baseline score back into the original dataframe
    data_with_baseline = pd.merge(data_df, baseline_df, on=['benchmark', 'program'], how='left')
    
    # Calculate the percentage gain for each optimizer compared to the baseline
    data_with_baseline['percentage_gain'] = ((data_with_baseline['score'] - data_with_baseline['baseline_score']) /
                                             data_with_baseline['baseline_score']) * 100
    
    # Filter out baseline entries from the plotting data
    data_to_plot = data_with_baseline[data_with_baseline['optimizer'] != "Baseline"]
    
    # Get a list of unique benchmarks
    benchmarks = data_to_plot['benchmark'].unique()
    num_benchmarks = len(benchmarks)
    
    # Determine the number of rows and columns based on the square root of the number of benchmarks
    cols = math.ceil(math.sqrt(num_benchmarks))
    rows = math.ceil(num_benchmarks / cols)
    
    # Set up the figure with subplots arranged in a grid
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), sharey=False)
    fig.suptitle("Percentage Gain by Program and Optimizer for Each Benchmark", fontsize=16)
    
    # Define a color palette, skipping the first color for "Baseline"
    unique_optimizers = data_to_plot['optimizer'].unique()
    palette = sns.color_palette("viridis", len(unique_optimizers) + 1)[1:]
    
    # Flatten axes array for easy iteration
    axes = axes.flatten()
    
    # Create subplots for each benchmark
    for i, benchmark in enumerate(benchmarks):
        ax = axes[i]
        benchmark_data = data_to_plot[data_to_plot['benchmark'] == benchmark]
        
        # Calculate individual y-axis limits to fit the data within each subplot
        y_min, y_max = benchmark_data['percentage_gain'].min(), benchmark_data['percentage_gain'].max()
        ax.set_ylim(y_min - abs(y_min) * 0.1, y_max + abs(y_max) * 0.1)  # Add some padding for visibility
        
        # Create a bar plot for each benchmark
        sns.barplot(
            data=benchmark_data,
            x='program', y='percentage_gain', hue='optimizer', ax=ax,
            palette=palette
        )
        
        # Set title and labels
        ax.set_title(f"Benchmark: {benchmark}")
        ax.set_xlabel("Program")
        ax.set_ylabel("Percentage Gain (%)")
        
        # Rotate x-axis labels to prevent overlapping
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        
        # Remove individual legends
        ax.get_legend().remove()
    
    # Hide any unused subplots
    for j in range(i + 1, rows * cols):
        fig.delaxes(axes[j])

    # Create custom legend handles and labels
    legend_handles = [Patch(color=palette[i], label=optimizer) for i, optimizer in enumerate(unique_optimizers)]
    fig.legend(handles=legend_handles, title="Optimizer", loc='upper left')
    
    # Adjust layout and save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # To leave space for the suptitle and legend
    plt.savefig(f"{directory_path}_percentage_gain_by_benchmark.png", dpi=300)
    plt.close()

def analyze_experiments(data_df):
    # Total cost, total input tokens, and total output tokens
    total_cost = data_df['cost'].sum() + data_df['optimizer_cost'].sum()
    total_input_tokens = data_df['input_tokens'].sum() + data_df['optimizer_input_tokens'].sum()
    # total_input_tokens in terms of millions string (xx M)
    total_input_tokens = f"{total_input_tokens / 1_000_000} M"

    total_output_tokens = data_df['output_tokens'].sum() + data_df['optimizer_output_tokens'].sum()
    # total_output_tokens in terms of millions string (xx M)
    total_output_tokens = f"{total_output_tokens / 1_000_000} M"
    
    # Calculate average performance increase for each optimizer, excluding outliers
    performance_increase = {}
    optimizers = data_df['optimizer'].unique()
    
    for optimizer in optimizers:
        # Filter data for the current optimizer and baseline
        optimizer_data = data_df[data_df['optimizer'] == optimizer]
        baseline_data = data_df[(data_df['optimizer'] == "Baseline") & 
                                (data_df['benchmark'].isin(optimizer_data['benchmark'])) &
                                (data_df['program'].isin(optimizer_data['program']))]
        
        # Merge to calculate performance increase over baseline for matching benchmark/program combinations
        merged_data = pd.merge(
            optimizer_data[['benchmark', 'program', 'score']],
            baseline_data[['benchmark', 'program', 'score']],
            on=['benchmark', 'program'],
            suffixes=('', '_baseline')
        )
        
        # Calculate performance increase as a percentage
        merged_data['performance_increase'] = ((merged_data['score'] - merged_data['score_baseline']) /
                                               merged_data['score_baseline']) * 100
        
        # Exclude outliers using the IQR method
        Q1 = merged_data['performance_increase'].quantile(0.25)
        Q3 = merged_data['performance_increase'].quantile(0.75)
        IQR = Q3 - Q1
        non_outliers = merged_data[(merged_data['performance_increase'] >= Q1 - 1.5 * IQR) &
                                   (merged_data['performance_increase'] <= Q3 + 1.5 * IQR)]
        
        # Calculate the mean performance increase for this optimizer
        avg_performance_increase = non_outliers['performance_increase'].mean()
        performance_increase[optimizer] = avg_performance_increase

    # average_optimizer_cost = data_df.groupby('optimizer')['optimizer_cost'].mean().to_dict()

    # Compile results
    results = {
        "total_cost": total_cost,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "average_performance_increase (excluding outliers)": performance_increase,
        # "average_optimizer_cost": average_optimizer_cost
    }

    return results

if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser()

    args.add_argument(
        "--file_path",
        type=str,
        required=True,
        help="Path to the text files containing benchmark results",
    )

    args = args.parse_args()

    file_path = args.file_path
    data_df = extract_information_from_files(file_path)
    plot_scores_by_benchmark(file_path, data_df)
    plot_percentage_gain_by_benchmark(file_path, data_df)
    results = analyze_experiments(data_df)
    import rich
    rich.print(results)