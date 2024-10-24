import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Update the function to properly use optimizer names in the legend
def plot_benchmark_results(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path, header=None)

    # Extract number of optimizers dynamically
    n_optimizers = (df.shape[1] - 2) // 2
    optimizer_names = df.iloc[0, 2:2 + n_optimizers].values.tolist()
    
    # Prepare data for plotting
    benchmarks = df[0].unique()  # List of all benchmarks
    n_benchmarks = len(benchmarks)

    # Adjust the number of rows and columns to make the figure more square-shaped
    n_cols = 2  # Number of columns
    n_rows = (n_benchmarks + 1) // n_cols  # Calculate rows needed

    # Set up a figure for multiple subplots in a grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8), sharex=False)
    axes = axes.flatten()  # Flatten axes for easier iteration

    # Iterate through each benchmark and plot on its respective axis
    for i, benchmark in enumerate(benchmarks):
        benchmark_df = df[df[0] == benchmark]
        
        # Create a list of optimizer columns (including baseline)
        optimizers = ['Baseline'] + optimizer_names
        score_columns = df.columns[-len(optimizers):]

        # Create a long format DataFrame for easier plotting
        long_format_df = pd.melt(benchmark_df, id_vars=[0, 1], 
                                 value_vars=score_columns, 
                                 var_name='Optimizer', 
                                 value_name='Performance')
        
        # Map optimizer columns to real optimizer names for better legend readability
        long_format_df['Optimizer'] = long_format_df['Optimizer'].replace(
            {df.columns[-len(optimizers)]: 'Baseline', **dict(zip(df.columns[-len(optimizers)+1:], optimizer_names))}
        )
        
        # Rename columns for clarity
        long_format_df.columns = ['Benchmark', 'Program', 'Optimizer', 'Performance']

        # Plot for this specific benchmark on its respective axis
        sns.barplot(data=long_format_df, x='Program', y='Performance', hue='Optimizer', ax=axes[i])

        # Titles and formatting for each subplot
        axes[i].set_title(f'Performance for {benchmark.split("Bench")[0]}')
        axes[i].set_ylabel('Performance (%)')
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)

        # Remove legend from individual subplots
        axes[i].get_legend().remove()

    # Remove unused subplots
    for ax in axes[len(benchmarks):]:
        ax.remove()

    # Create one shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Optimizer', loc='lower right')

    # Adjust layout to prevent overlapping
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    plt.savefig(f'{file_path.split(".")[0]}.png', dpi=300)


if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser()

    args.add_argument(
        "--file_path",
        type=str,
        required=True,
        help="Path to the CSV file containing benchmark results",
    )

    args = args.parse_args()

    file_path = args.file_path
    plot_benchmark_results(file_path)
