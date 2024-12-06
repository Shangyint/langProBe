import math
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import pathlib

color_schemes = ["Reds", "Greens", "Blues", "Purples", "Oranges", "YlGnBu", "coolwarm"]


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

        with open(file, "r") as f:
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
    df["optimizer"] = df["optimizer"].replace("None", "Baseline")
    return df


def plot_scores_by_benchmark(directory_path: str, data_dfs, model_names):
    # Get a list of unique benchmarks

    combined_data = []
    for df, model in zip(data_dfs, model_names):
        df = df.copy()
        df['model'] = model
        combined_data.append(df)
    combined_data = pd.concat(combined_data)

    unique_optimizers = combined_data["optimizer"].unique()


    benchmarks = combined_data["benchmark"].unique()
    num_benchmarks = len(benchmarks)

    # Determine the number of rows and columns based on the square root of the number of benchmarks
    cols = math.ceil(math.sqrt(num_benchmarks))
    rows = math.ceil(num_benchmarks / cols)

    if cols <= 1:
        cols = 3
    if rows <= 1:
        rows = 3

    # Set up the figure with subplots arranged in a grid
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), sharey=False)
    fig.suptitle("Scores by Program and Optimizer for Each Benchmark", fontsize=16)


    if rows == 1 and cols == 1:
        axes = [axes]

    # Flatten axes array for easy iteration
    axes = axes.flatten() if rows > 1 or cols > 1 else axes

    hue_palette = {
        model: sns.color_palette(color_schemes[i % len(color_schemes)], len(unique_optimizers))
        for i, model in enumerate(model_names)
    }

    combined_data['program_model'] = combined_data['program'] + "_" + combined_data['model']

    bar_width = 0.35  # Adjust bar width for each model

    for i, benchmark in enumerate(benchmarks):
        ax = axes[i]
        benchmark_data = combined_data[combined_data['benchmark'] == benchmark].copy()
        
        # Get unique programs and assign numeric positions
        programs = benchmark_data['program'].unique()
        # sort programs based on the order of the program_order
        programs = sorted(programs, key=lambda x: program_order.index(x) if x in program_order else len(program_order))
        program_indices = np.arange(len(programs))  # Base positions for programs
        
        # Dictionary to store offsets for calculating tick positions
        program_offsets = {program: [] for program in programs}
        
        # Loop through programs to group bars by program
        for j, program in enumerate(programs):
            program_data = benchmark_data[benchmark_data['program'] == program]
            models_in_program = program_data['model'].unique()  # Models for this program
            
            # Loop through models to calculate offsets
            for k, model in enumerate(models_in_program):
                model_data = program_data[program_data['model'] == model].copy()
                
                # Calculate bar offsets for the current model
                model_data['x_offset'] = program_indices[j] + (k - len(models_in_program) / 2) * bar_width
                
                # Append the offsets for this program
                program_offsets[program].extend(model_data['x_offset'])
                
                # Plot bars for this model
                sns.barplot(
                    data=model_data,
                    x='x_offset', y='score', hue='optimizer', dodge=True, ax=ax,
                    palette=dict(zip(unique_optimizers, hue_palette[model]))
                )
                
                # Collect actual x-offsets for each program from the plotted bars
        for patch in ax.patches:
            # Extract x-position of each bar
            bar_center = patch.get_x() + patch.get_width() / 2
            # Extract program name based on the base position
            program_idx = int(round(bar_center))  # Round to find the closest program index
            if 0 <= program_idx < len(programs):  # Ignore bars outside valid program range
                program_offsets[programs[program_idx]].append(bar_center)
        
        # Calculate the center of all offsets for each program
        tick_positions = [np.mean(offsets) for offsets in program_offsets.values()]

        
        # Calculate the center of all offsets for each program
        print(tick_positions)

        # Set x-axis ticks and labels
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(programs, rotation=45, ha='right')
        
        # Set title and labels
        ax.set_title(f"Benchmark: {benchmark}")
        ax.set_xlabel("Program")
        ax.set_ylabel("Score")
        
        # Remove individual legends from subplots
        ax.get_legend().remove()
    for j in range(i + 1, rows * cols):
        fig.delaxes(axes[j])

    # create a handle for unique optimizers
    legend_handles = [
        Patch(
            color=sns.color_palette("Reds", len(unique_optimizers))[i],
            label=optimizer,
        )
        for i, optimizer in enumerate(unique_optimizers)
    ]
    fig.legend(title="Optimizer", handles=legend_handles, loc="upper left")

    legend_handles = [Patch(color=hue_palette[model][0], label=model) for model in model_names]
    fig.legend(handles=legend_handles, title="Language Models", loc='upper right')
    
    # Adjust layout and add legend
    if rows > 1 or cols > 1:
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # To leave space for the suptitle and legend
    else:
        plt.subplots_adjust(top=0.85)

    plt.savefig(f"{directory_path}_{'_'.join(model_names)}_scores_by_benchmark.png", dpi=300)


def plot_scores_by_benchmark_model_only(directory_path: str, data_dfs, model_names):
    # Combine all dataframes and assign model names
    combined_data = []
    for df, model in zip(data_dfs, model_names):
        df = df.copy()
        df['model'] = model
        combined_data.append(df)
    combined_data = pd.concat(combined_data)

    # Filter the data to include only "Baseline" optimizer
    combined_data = combined_data[combined_data["optimizer"] == "Baseline"]

    benchmarks = combined_data["benchmark"].unique()
    num_benchmarks = len(benchmarks)

    # Determine rows and columns for subplots
    cols = math.ceil(math.sqrt(num_benchmarks))
    rows = math.ceil(num_benchmarks / cols)

    # Set up the figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), sharey=False)
    fig.suptitle("Scores by Program for Each Benchmark (Baseline Only)", fontsize=16)

    # Flatten axes array for easy iteration
    axes = axes.flatten() if rows > 1 or cols > 1 else [axes]

    # Generate color palette for models
    color_schemes = ["Reds", "Greens", "Blues", "Purples"]
    hue_palette = {
        model: sns.color_palette(color_schemes[i % len(color_schemes)], 1)[0]
        for i, model in enumerate(model_names)
    }

    for i, benchmark in enumerate(benchmarks):
        ax = axes[i]
        benchmark_data = combined_data[combined_data['benchmark'] == benchmark].copy()
        
        # Get unique programs and sort them
        programs = benchmark_data['program'].unique()
        benchmark_data = sort_dataframe(benchmark_data, program_order=program_order)
        
        # Create a bar plot comparing models across programs
        sns.barplot(
            data=benchmark_data,
            x="program",
            y="score",
            hue="model",
            ax=ax,
            palette=hue_palette
        )
        
        # Set title and labels
        ax.set_title(f"Benchmark: {benchmark}")
        ax.set_xlabel("Program")
        ax.set_ylabel("Score")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
        # Remove individual legends from subplots
        ax.get_legend().remove()

    # Remove unused axes
    for j in range(len(benchmarks), len(axes)):
        fig.delaxes(axes[j])

    # Add a global legend for models
    fig.legend(
        title="Language Models",
        handles=[Patch(color=color, label=model) for model, color in hue_palette.items()],
        loc="upper right"
    )

    # Adjust layout and save the figure
    if rows > 1 or cols > 1:
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the suptitle and legend
    else:
        plt.subplots_adjust(top=0.85)

    plt.savefig(f"{directory_path}_{'_'.join(model_names)}_baseline_scores_by_benchmark.png", dpi=300)

def plot_percentage_gain_by_benchmark(directory_path, data_dfs, model_names):

    return
    # Calculate the baseline score for each benchmark and program
    baseline_df = data_df[data_df["optimizer"] == "Baseline"].rename(
        columns={"score": "baseline_score"}
    )
    baseline_df = baseline_df[["benchmark", "program", "baseline_score"]]

    # Merge the baseline score back into the original dataframe
    data_with_baseline = pd.merge(
        data_df, baseline_df, on=["benchmark", "program"], how="left"
    )

    # Calculate the percentage gain for each optimizer compared to the baseline
    data_with_baseline["percentage_gain"] = (
        (data_with_baseline["score"] - data_with_baseline["baseline_score"])
        / data_with_baseline["baseline_score"]
    ) * 100

    # Filter out baseline entries from the plotting data
    data_to_plot = data_with_baseline[data_with_baseline["optimizer"] != "Baseline"]

    # Get a list of unique benchmarks
    benchmarks = data_to_plot["benchmark"].unique()
    num_benchmarks = len(benchmarks)

    # Determine the number of rows and columns based on the square root of the number of benchmarks
    cols = math.ceil(math.sqrt(num_benchmarks))
    rows = math.ceil(num_benchmarks / cols)

    if cols <= 1:
        cols = 3
    if rows <= 1:
        rows = 3

    # Set up the figure with subplots arranged in a grid
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), sharey=False)
    fig.suptitle(
        "Percentage Gain by Program and Optimizer for Each Benchmark", fontsize=16
    )

    # Define a color palette, skipping the first color for "Baseline"
    unique_optimizers = data_to_plot["optimizer"].unique()
    palette = sns.color_palette("viridis", len(unique_optimizers) + 1)[1:]

    if rows == 1 and cols == 1:
        axes = [axes]
    # Flatten axes array for easy iteration
    axes = axes.flatten() if rows > 1 or cols > 1 else axes

    # Create subplots for each benchmark
    for i, benchmark in enumerate(benchmarks):
        ax = axes[i]
        benchmark_data = data_to_plot[data_to_plot["benchmark"] == benchmark]

        # Calculate individual y-axis limits to fit the data within each subplot
        y_min, y_max = (
            benchmark_data["percentage_gain"].min(),
            benchmark_data["percentage_gain"].max(),
        )
        ax.set_ylim(
            y_min - abs(y_min) * 0.1, y_max + abs(y_max) * 0.1
        )  # Add some padding for visibility

        # Create a bar plot for each benchmark
        sns.barplot(
            data=benchmark_data,
            x="program",
            y="percentage_gain",
            hue="optimizer",
            ax=ax,
            palette=palette,
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
    legend_handles = [
        Patch(color=palette[i], label=optimizer)
        for i, optimizer in enumerate(unique_optimizers)
    ]
    fig.legend(handles=legend_handles, title="Optimizer", loc="upper left")

    # Adjust layout and save the figure
    if rows > 1 or cols > 1:
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # To leave space for the suptitle and legend
    else:
        plt.subplots_adjust(top=1.2)

    plt.savefig(f"{directory_path}_{'_'.join(model_names)}_percentage_gain_by_benchmark.png", dpi=300)
    plt.close()


def analyze_experiments(data_df):
    # Total cost, total input tokens, and total output tokens
    total_cost = data_df["cost"].sum() + data_df["optimizer_cost"].sum()
    total_input_tokens = (
        data_df["input_tokens"].sum() + data_df["optimizer_input_tokens"].sum()
    )
    # total_input_tokens in terms of millions string (xx M)
    total_input_tokens = f"{total_input_tokens / 1_000_000} M"

    total_output_tokens = (
        data_df["output_tokens"].sum() + data_df["optimizer_output_tokens"].sum()
    )
    # total_output_tokens in terms of millions string (xx M)
    total_output_tokens = f"{total_output_tokens / 1_000_000} M"

    # Calculate average performance increase for each optimizer, excluding outliers
    performance_increase = {}
    optimizers = data_df["optimizer"].unique()

    for optimizer in optimizers:
        # Filter data for the current optimizer and baseline
        optimizer_data = data_df[data_df["optimizer"] == optimizer]
        baseline_data = data_df[
            (data_df["optimizer"] == "Baseline")
            & (data_df["benchmark"].isin(optimizer_data["benchmark"]))
            & (data_df["program"].isin(optimizer_data["program"]))
        ]

        # Merge to calculate performance increase over baseline for matching benchmark/program combinations
        merged_data = pd.merge(
            optimizer_data[["benchmark", "program", "score"]],
            baseline_data[["benchmark", "program", "score"]],
            on=["benchmark", "program"],
            suffixes=("", "_baseline"),
        )

        # Calculate performance increase as a percentage
        merged_data["performance_increase"] = (
            (merged_data["score"] - merged_data["score_baseline"])
            / merged_data["score_baseline"]
        ) * 100

        # Exclude outliers using the IQR method
        Q1 = merged_data["performance_increase"].quantile(0.25)
        Q3 = merged_data["performance_increase"].quantile(0.75)
        IQR = Q3 - Q1
        non_outliers = merged_data[
            (merged_data["performance_increase"] >= Q1 - 1.5 * IQR)
            & (merged_data["performance_increase"] <= Q3 + 1.5 * IQR)
        ]

        # Calculate the mean performance increase for this optimizer
        avg_performance_increase = non_outliers["performance_increase"].mean()
        performance_increase[optimizer] = avg_performance_increase

    average_optimizer_cost = (
        data_df.groupby("optimizer")["optimizer_cost"].mean().to_dict()
    )

    # Compile results
    results = {
        "total_cost": total_cost,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "average_performance_increase (excluding outliers)": performance_increase,
        "average_optimizer_cost": average_optimizer_cost,
    }

    return results


def get_programs(data_df):
    # Get the unique programs from the data
    programs = data_df["program"].unique()

    # Convert the programs list to a text format, each program on a new line
    programs_text = "\n".join(programs)

    return programs_text


def display_benchmark_performance(data_df, selected_benchmarks):
    # Filter data for the selected benchmarks
    filtered_data = data_df[data_df['benchmark'].isin(selected_benchmarks)]
    
    # Sort the data for better readability
    filtered_data = filtered_data.sort_values(by=['benchmark', 'program', 'optimizer'])

    # for the same benchmark and program, None should always come first

    
    # Create a human-readable table
    performance_table = filtered_data.pivot_table(
        index=['benchmark', 'program'],
        columns='optimizer',
        values='score',
        aggfunc='mean'
    )
    
    # Reorder columns to ensure "None (Baseline)" is always first
    if "Baseline" in performance_table.columns:
        cols = ["Baseline"] + [col for col in performance_table.columns if col != "Baseline"]
        performance_table = performance_table[cols]

    # Return the performance table
    return performance_table


def compare_all_programs_with_predict_baseline(data_df):
    # Filter data to only include rows where optimizer is "Baseline"
    baseline_data = data_df[data_df["optimizer"] == "Baseline"]

    # Initialize a dictionary to store comparisons for each benchmark
    comparisons = {}

    # Group data by benchmark
    benchmark_groups = baseline_data.groupby("benchmark")

    for benchmark, group in benchmark_groups:
        # Get the score for program "CoT" with optimizer "Baseline"
        cot_score = group[group["program"].str.contains("predict", case=False)]["score"].values

        # Ensure "CoT" exists within the benchmark
        if len(cot_score) == 0:
            continue  # Skip if "CoT" is missing for this benchmark

        cot_score = cot_score[0]  # Get the CoT score value

        # Calculate the score difference for each program with respect to CoT
        group["score_difference"] = group["score"] - cot_score

        # Store the comparison results for this benchmark
        comparisons[benchmark] = group[
            ["program", "score", "score_difference"]
        ].set_index("program")

    return comparisons


def compare_programs_with_reference_across_benchmarks(data_df):
    # Filter data to only include rows where optimizer is "Baseline"
    baseline_data = data_df[data_df["optimizer"] == "Baseline"]

    # Initialize a list to store score differences for each program relative to the reference
    score_diffs = []

    # Group data by benchmark
    benchmark_groups = baseline_data.groupby("benchmark")

    for benchmark, group in benchmark_groups:
        # Get the score for the reference program ("CoT" or "ChainOfThought") with optimizer "Baseline"
        reference_score = group[group["program"].str.contains("predict", case=False)]["score"].values

        # Ensure that the reference program exists within the benchmark
        if len(reference_score) == 0:
            continue  # Skip if neither "CoT" nor "ChainOfThought" is present for this benchmark

        reference_score = reference_score[0]  # Get the reference score value

        # Calculate the score difference for each program with respect to the reference program
        group["score_difference"] = group["score"] - reference_score

        # Append the results for non-reference programs to the list
        score_diffs.extend(
            group[group["program"] != "CoT"][["program", "score_difference"]].values
        )

    # Create a DataFrame from the collected score differences
    score_diffs_df = pd.DataFrame(score_diffs, columns=["program", "score_difference"])

    # Calculate the average score difference for each program across benchmarks
    avg_score_diffs = (
        score_diffs_df.groupby("program")["score_difference"].mean().reset_index()
    )
    avg_score_diffs.columns = ["program", "average_score_difference"]

    return avg_score_diffs

def plot_df_with_programs(df, programs, file_name=None):
    filtered_df = df[
        (df["program"].isin(programs)) & (df["optimizer"] == "Baseline")
    ]
    
    # Identify benchmarks containing all specified programs
    valid_benchmarks = filtered_df.groupby("benchmark")["program"].nunique()
    valid_benchmarks = valid_benchmarks[valid_benchmarks == len(programs)].index
    
    # Filter the DataFrame to include only these benchmarks
    plot_df = filtered_df[filtered_df["benchmark"].isin(valid_benchmarks)]
    
    if plot_df.empty:
        print("No valid data to plot.")
        return

    # Set up the plot
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    # Get unique colors for each program
    palette = sns.color_palette("husl", len(programs))
    program_colors = dict(zip(programs, palette))

    # Plot each program
    for program in programs:
        program_data = plot_df[plot_df["program"] == program]
        sns.lineplot(
            data=program_data,
            x="benchmark",
            y="score",
            label=program,
            color=program_colors[program],
            marker="o",
        )

    # Plot averages for each program
    averages = (
        plot_df.groupby("program")["score"]
        .mean()
        .reindex(programs)
    )
    for program, avg in averages.items():
        plt.axhline(
            avg,
            linestyle="--",
            color=program_colors[program],
            alpha=0.7,
        )
        # Add the average value as text above the line
        plt.text(
            x=0.02,  # Position slightly inside the plot
            y=avg,
            s=f"{avg:.2f}",
            color=program_colors[program],
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="left",
        )

    # Customize plot
    plt.title("Benchmark Scores by Program (Baseline Only)", fontsize=16)
    plt.xlabel("Benchmark", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    
    # Add a single dotted line in the legend labeled "Average"
    plt.plot([], [], linestyle="--", color="black", label="Average")


    # Update legend
    plt.legend(title="Programs", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # Save the plot
    plt.savefig(file_name, dpi=300)
    plt.close()



def find_benchmarks_where_miprov2_performs_worse(data_df):
    # Initialize a list to store benchmarks where MIPROv2 performs worse
    underperforming_benchmarks = []

    # Group data by benchmark
    benchmark_groups = data_df.groupby("benchmark")

    for benchmark, group in benchmark_groups:
        # Get MIPROv2 score for this benchmark
        miprov2_score = group[group["optimizer"] == "MIPROv2"]["score"].values

        # Ensure MIPROv2 exists within the benchmark
        if len(miprov2_score) == 0:
            continue  # Skip if MIPROv2 is not present in this benchmark

        miprov2_score = miprov2_score[0]

        # Get scores of all other optimizers in the same benchmark
        other_optimizers = group[group["optimizer"] != "MIPROv2"]

        # Check if MIPROv2 performs worse than any other optimizer
        if miprov2_score < other_optimizers["score"].max():
            # Calculate the score difference between MIPROv2 and other optimizers
            other_optimizers["score_difference"] = (
                miprov2_score - other_optimizers["score"]
            )

            # Store the benchmark information and the score differences
            benchmark_info = {
                "benchmark": benchmark,
                "miprov2_score": miprov2_score,
                "comparison": other_optimizers[
                    ["optimizer", "score", "score_difference"]
                ].set_index("optimizer"),
            }
            underperforming_benchmarks.append(benchmark_info)

    return underperforming_benchmarks

def sort_dataframe(df, program_order=None, optimizer_order=None):
    """
    Sort a DataFrame by custom order for 'program' and 'optimizer' columns.
    
    Args:
        df (pd.DataFrame): The DataFrame to be sorted.
        program_order (list): Custom order for the 'program' column. Programs not in the list retain their original order.
        optimizer_order (list): Custom order for the 'optimizer' column. Optimizers not in the list retain their original order.
    
    Returns:
        pd.DataFrame: The sorted DataFrame.
    """
    # If program_order is provided, create a mapping for custom sorting
    if program_order:
        program_mapping = {program: i for i, program in enumerate(program_order)}
        df['program_priority'] = df['program'].map(program_mapping).fillna(len(program_order))
    else:
        df['program_priority'] = 0  # Default priority for all programs
    
    # If optimizer_order is provided, create a mapping for custom sorting
    if optimizer_order:
        optimizer_mapping = {optimizer: i for i, optimizer in enumerate(optimizer_order)}
        df['optimizer_priority'] = df['optimizer'].map(optimizer_mapping).fillna(len(optimizer_order))
    else:
        df['optimizer_priority'] = 0  # Default priority for all optimizers
    
    # Sort by the calculated priority columns
    sorted_df = df.sort_values(by=['program_priority', 'optimizer_priority', 'program', 'optimizer']).drop(
        ['program_priority', 'optimizer_priority'], axis=1
    )
    
    return sorted_df

program_map = {
    'Predict' : 'Predict',
    'ChainOfThought' : 'CoT', 
    'GeneratorCriticRanker': 'GeneratorCriticRanker',
    'GeneratorCriticFuser': 'GeneratorCriticFuser',  
    'RAG': 'RAG',
    'EvaluationValidityPredict': 'Predict', 
    'EvaluationValidityModule': 'CoT', 
    'CoT': 'CoT',
    'Classify': 'CoTBasedVote', 
    'HeartDiseaseClassify': 'CoTBasedVote',  
    'RetrieveMultiHop': "RetrieveMultiHop",
    'SimplifiedBaleen': 'SimplifiedBaleen',
    'SimplifiedBaleenWithHandwrittenInstructions': "SimplifiedBaleenWithHandwrittenInstructions",
    'UnderspecifiedAnnotationGenerator': "CoT", 
    'UnderspecifiedAnnotationPredict': "Predict",

    # Relook at the following programs
    'IReRaCOT' : 'CoT',
    'IReRaPredict' : 'Predict', 
    'Infer': "CoT", 
    'InferRetrieve': 'IReRaRetrieve', 
    'IReRaRetrieve': 'IReRaRetrieve', 
    'IReRaRetrieveRank' : "IReRaRetrieveRank",
    'InferRetrieveRank': "IReRaRetrieveRank",
    
    # 'IReRaCOT' : 'IReRaCOT',
    # 'IReRaPredict' : 'Predict', 
    # 'Infer': "Infer", 
    # 'InferRetrieve': 'InferRetrieve', 
    # 'IReRaRetrieve': 'IReRaRetrieve', 
    # 'IReRaRetrieveRank' : "IReRaRetrieveRank",
    # 'InferRetrieveRank': "InferRetrieveRank",
}

if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser()

    args.add_argument(
        "--file_path",
        nargs='+',
        required=True,
        default=[],
        help="Path to the text files containing benchmark results",
    )

    args.add_argument(
        "--benchmark",
        type=str,
        required=False,
        default=None,
        help="Name of the benchmark to analyze",
    )

    args = args.parse_args()

    file_paths = args.file_path
    data_dfs = []
    model_names = [file_path.split('_')[1] for file_path in file_paths]


    # Custom order for program and optimizer
    program_order = ['Predict', 'ChainOfThought']  # Followed by original order
    optimizer_order = ['Baseline', 'BootstrapFewshot', 'BootstrapFewshotWithRandomSearch', 'MIPROv2']

    for file_path in file_paths:
        data_df = extract_information_from_files(file_path)
        # exclude hoverBench, JudgeBench, and HeartDiseaseBench
        data_df = data_df[~data_df["benchmark"].isin(["hoverBench", "JudgeBench", "HeartDiseaseBench"])]
        data_df = sort_dataframe(data_df, program_order=program_order, optimizer_order=optimizer_order)
        data_dfs.append(data_df)
    # TODO FIX THIS
    if args.benchmark:
        data_df = data_df[data_df["benchmark"] == f"{args.benchmark}Bench"]
    plot_scores_by_benchmark(file_path, data_dfs, model_names)
    plot_scores_by_benchmark_model_only(file_path, data_dfs, model_names)
    # plot_percentage_gain_by_benchmark(file_path, data_df)
    for data_df, model_name in zip(data_dfs, model_names):
        data_df.to_csv(f"evaluation_{model_name}.csv", index=False, header=True)
        results = analyze_experiments(data_df)
        import rich

        rich.print(f"[bold red]Showing results for {model_name}[/bold red]")
        rich.print(results)
        results = analyze_experiments(data_df)
        import rich

        rich.print(results)

        # Example usage
        programs_list = get_programs(data_df)
        comparisons = compare_all_programs_with_predict_baseline(data_df)
        rich.print(comparisons)

        avg_score_diffs = compare_programs_with_reference_across_benchmarks(data_df)
        rich.print(avg_score_diffs)

        # data_df replace with program_map
        data_df["program"] = data_df["program"].map(program_map)

        plot_df_with_programs(data_df, ["Predict", "CoT"], file_name=f"{model_name}_program_performance_baseline.png")
        plot_df_with_programs(data_df, ["Predict", "CoT", "GeneratorCriticRanker", "GeneratorCriticFuser"], file_name=f"{model_name}_program_performance.png")
        plot_df_with_programs(data_df, ["Predict", "CoT", "RAG", "SimplifiedBaleen"], file_name=f"{model_name}_program_performance_rag.png")