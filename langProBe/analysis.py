import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    print(df["benchmark"].unique())
    return df


program_mapping = {
    "AppWorldReact": "ReActBaseline",
    "AppWorldReactAugumented": "ReActAugumented",
    "Predict": "Predict",
    "ChainOfThought": "CoT",
    "GeneratorCriticRanker": "GeneratorCriticRanker",
    "GeneratorCriticFuser": "GeneratorCriticFuser",
    "RAG": "RAG",
    "EvaluationValidityPredict": "Predict",
    "EvaluationValidityModule": "CoT",
    "CoT": "CoT",
    "Classify": "CoTBasedVote",
    "HeartDiseaseClassify": "CoTBasedVote",
    "RetrieveMultiHop": "RetrieveMultiHop",
    "SimplifiedBaleen": "SimplifiedBaleen",
    "SimplifiedBaleenWithHandwrittenInstructions": "SimplifiedBaleenWithInst",
    "UnderspecifiedAnnotationCoT": "CoT",
    "UnderspecifiedAnnotationPredict": "Predict",
    "EvaluationValidityCoT": "CoT",
    "EvaluationValidityPredict": "Predict",
    # Relook at the following programs
    "IReRaCOT": "CoT",
    "IReRaPredict": "Predict",
    "Infer": "CoT",
    "InferRetrieve": "RAG",
    "IReRaRetrieve": "RAG",
    "IReRaRetrieveRank": "RAGBasedRank",
    "InferRetrieveRank": "RAGBasedRank",
    "HoverMultiHopPredict": "Predict",
    "HoverMultiHop": "MultiHopSummarize",
}


def canonicalize_program(data_df):
    # Update the benchmark names based on the program
    data_df.loc[
        data_df["program"].isin(["UnderspecifiedAnnotationCoT", "UnderspecifiedAnnotationPredict"]),
        "benchmark"
    ] = "SWEBenchUnderspecified"

    data_df.loc[
        data_df["program"].isin(["EvaluationValidityCoT", "EvaluationValidityPredict"]),
        "benchmark"
    ] = "SWEBenchValidity"
    data_df["program"] = data_df["program"].replace(program_mapping)
    data_df["benchmark"] = data_df["benchmark"].apply(lambda x: x.replace("Bench", ""))
    print("HEREHERE")
    print(data_df["benchmark"].unique())
    return data_df


## Plotting functions
# Global variable to store consistent program colors
PROGRAM_COLORS = {}

CUD_COLORS = [
    "#56B4E9",  # Sky Blue
    "#E69F00",  # Orange
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#CC79A7",  # Reddish Purple
    "#999999",  # Gray
    "#882255",  # Dark Red (new)
    "#44AA99",  # Teal (new)
    "#332288",  # Dark Blue (new)
    "#AA4499",  # Purple (new)
    "#117733",  # Dark Green (new)
    "#DDCC77",  # Sand Yellow (new)
]


def plot_program_specific(data_df, programs, model, benchmark_to_categories=None):
    """
    Plot program-specific benchmark scores for Baseline optimizer.

    Args:
        data_df (pd.DataFrame): The input DataFrame containing benchmark data.
        programs (list): List of programs to include in the plot.
        model (str): Name of the model used in the experiment.
        benchmark_to_categories (dict, optional): A mapping from benchmarks to categories for highlighting.
    """
    # Filter benchmarks that have all specified programs
    benchmarks_with_all_programs = data_df[data_df["optimizer"] == "Baseline"]
    valid_benchmarks = (
        benchmarks_with_all_programs.groupby("benchmark")
        .filter(
            lambda x: set(programs).issubset(
                set(x["program"])
            )  # Ensure all programs exist for the benchmark
        )["benchmark"]
        .unique()
    )

    # Filter the DataFrame to include only valid benchmarks and specified programs
    filtered_df = data_df[
        (data_df["benchmark"].isin(valid_benchmarks))
        & (data_df["program"].isin(programs))
        & (data_df["optimizer"] == "Baseline")
    ]

    # Sort programs to ensure Predict comes first and CoT second
    sorted_programs = sorted(programs, key=lambda x: (x != "Predict", x != "CoT", x))

    # Group by benchmark and program to calculate mean scores
    grouped = filtered_df.groupby(["benchmark", "program"])["score"].mean().unstack()

    # Ensure all programs are represented in the DataFrame
    for program in sorted_programs:
        if program not in grouped.columns:
            grouped[program] = float("nan")  # Add missing programs as NaN

    # Reorder columns
    grouped = grouped[sorted_programs]

    # Sort benchmarks by category if benchmark_to_categories is provided
    if benchmark_to_categories:
        grouped = grouped.reindex(
            sorted(grouped.index, key=lambda x: benchmark_to_categories.get(x, "zzz"))
        )

    # Assign consistent colors to programs
    global PROGRAM_COLORS
    cmap = plt.get_cmap("tab10")  # Default color palette
    new_colors = {}
    for idx, program in enumerate(sorted_programs):
        if program not in PROGRAM_COLORS:
            new_colors[program] = cmap(
                len(PROGRAM_COLORS) + len(new_colors)
            )  # Assign unique color
        else:
            new_colors[program] = PROGRAM_COLORS[program]  # Preserve existing color
    PROGRAM_COLORS.update(new_colors)  # Update global program colors

    # Define category colors if provided
    category_colors = {}
    if benchmark_to_categories:
        unique_categories = set(benchmark_to_categories.values())
        cmap_category = plt.get_cmap("Set2")  # Use Set2 colormap for categories
        for idx, category in enumerate(unique_categories):
            category_colors[category] = cmap_category(idx)

    # Plot bar chart
    fig, ax = plt.subplots(figsize=(12, 9))
    grouped.plot(
        kind="bar",
        ax=ax,
        alpha=0.8,
        edgecolor="black",
        color=[PROGRAM_COLORS[program] for program in sorted_programs],
    )

    # Add dotted average line for each program with matching colors
    avg_scores = grouped.mean()
    for program, avg in avg_scores.items():
        ax.axhline(
            y=avg,
            color=PROGRAM_COLORS[program],
            linestyle="dotted",
            linewidth=1.5,
            label=f"{program} Avg",
        )

    # Highlight benchmarks according to categories if mapping is provided
    if benchmark_to_categories:
        from matplotlib.patches import Patch

        category_patches = []
        for idx, benchmark in enumerate(grouped.index):
            if benchmark in benchmark_to_categories:
                category = benchmark_to_categories[benchmark]
                ax.get_xticklabels()[idx].set_backgroundcolor(category_colors[category])

        # Add category legend at the bottom
        category_patches = [
            Patch(facecolor=color, label=category)
            for category, color in category_colors.items()
        ]
        fig.legend(
            handles=category_patches,
            title="Benchmark Categories",
            loc="lower left",
            bbox_to_anchor=(0, -0.05),
            ncol=len(category_patches),
            fontsize=10,
            title_fontsize=12,
        )

    # Set plot title, labels, and legend
    ax.set_title(f"Program-Specific Benchmark Scores ({model})", fontsize=14)
    ax.set_xlabel("Benchmark", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.legend(title="Programs", fontsize=10, title_fontsize=12, loc="upper left")

    # Adjust layout to accommodate legend
    # plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()

    # Save the figure
    programs_str = "_".join(sorted_programs)
    filename = f"{model}_program_{programs_str}.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.show()

    print(f"Plot saved as {filename}")



def plot_best_program(data_df, model, optimizers=False):
    """
    Plot program-specific benchmark scores for Baseline optimizer.

    Args:
        data_df (pd.DataFrame): The input DataFrame containing benchmark data.
        programs (list): List of programs to include in the plot.
        model (str): Name of the model used in the experiment.
        benchmark_to_categories (dict, optional): A mapping from benchmarks to categories for highlighting.
    """
    # Filter benchmarks that have all specified programs
    benchmarks_with_all_programs = data_df[data_df["optimizer"] == "Baseline"] if not optimizers else data_df

    ## Group by benchmarks and select the best-performing program other than "Predict"
    best_programs = (
        benchmarks_with_all_programs[benchmarks_with_all_programs["program"] != "Predict"]
        .groupby("benchmark", as_index=False)
        .apply(lambda group: group.loc[group["score"].idxmax()])
    )

    # Extract scores for "Predict" program and merge with the best programs
    predict_scores = benchmarks_with_all_programs[
        (("ReActBaseline" == benchmarks_with_all_programs["program"]) | (benchmarks_with_all_programs["program"] == "Predict")) & (benchmarks_with_all_programs["optimizer"] == "Baseline")
    ][["benchmark", "score"]].rename(columns={"score": "predict_score"})
    
    best_programs = best_programs.rename(columns={"score": "best_score", "program": "best_program"})
    merged_data = pd.merge(best_programs, predict_scores, on="benchmark")

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    x_positions = np.arange(len(merged_data))

    # Bar width and offsets
    bar_width = 0.4

    # Plot bars for Predict and Best programs
    merged_data = merged_data.sort_values(by="best_program", ascending=True).reset_index(drop=True)

    ax.bar(x_positions - bar_width / 2, merged_data["predict_score"], width=bar_width, color="#56B4E9", label="Baseline")
    ax.bar(x_positions + bar_width / 2, merged_data["best_score"], width=bar_width, color="red", label="Best Program")


    for i, row in merged_data.iterrows():
        ax.text(
            x_positions[i],
            -0.04,  # Adjusted position closer to the axis
            row["benchmark"],
            fontsize=10,
            ha="right",
            va="top",
            rotation=45,
            transform=ax.get_xaxis_transform()
        )
        ax.text(
            x_positions[i],
            -0.10,  # Further below the benchmark name
            f"({row['best_program']})",
            fontsize=8,
            ha="right",
            va="top",
            rotation=45,
            transform=ax.get_xaxis_transform()
        )

    # Customize the plot
    ax.set_xlim(-0.5, len(merged_data) - 0.5)
    ax.set_ylabel("Score")
    optimized = "optimized" if optimizers else "unoptimized"
    ax.set_title(f"Baseline vs. Best Performing Programs ({optimized}) for {model}")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)  # Hide default x-axis labels

    plt.tight_layout()
    filename = f"{model}_best_program_{optimizers}.png"
    plt.savefig(filename, bbox_inches="tight", dpi=400)

    print(f"Plot saved as {filename}")

OPTIMIZER_COLORS = {}  # Initialize global optimizer colors


def plot_optimizer_specific(
    data_df,
    optimizers,
    model,
    benchmark_to_categories=None,
    benchmark_categories=None,
    programs=[],
):
    """
    Plot optimizer-specific benchmark scores for specified optimizers.

    Args:
        data_df (pd.DataFrame): The input DataFrame containing benchmark data.
        optimizers (list): List of optimizers to include in the plot.
        model (str): Name of the model used in the experiment.
        benchmark_to_categories (dict, optional): A mapping from benchmarks to categories for highlighting.
        benchmark_categories (list, optional): List of benchmark categories to include.
        programs (list, optional): List of programs to filter the data.
    """
    # Filter benchmarks based on categories
    if benchmark_categories and benchmark_to_categories:
        selected_benchmarks = [
            b for b, c in benchmark_to_categories.items() if c in benchmark_categories
        ]
        data_df = data_df[data_df["benchmark"].isin(selected_benchmarks)]

    # Filter programs if provided
    if programs:
        data_df = data_df[data_df["program"].isin(programs)]

    # Filter optimizers based on the provided list
    data_df = data_df[data_df["optimizer"].isin(optimizers)]

    # Sort optimizers based on the predefined order
    sorted_optimizers = [
        opt
        for opt in [
            "Baseline",
            "BootstrapFewShot",
            "BootstrapFewShotWithRandomSearch",
            "MIPROv2",
        ]
        if opt in optimizers
    ]

    # Group by benchmark and optimizer to calculate mean scores
    grouped = data_df.groupby(["benchmark", "optimizer"])["score"].mean().unstack()

    # Ensure all optimizers are represented in the DataFrame
    for optimizer in sorted_optimizers:
        if optimizer not in grouped.columns:
            grouped[optimizer] = float("nan")

    # Reorder optimizers and sort benchmarks
    grouped = grouped[sorted_optimizers]

    # Sort benchmarks by categories if mapping is provided
    if benchmark_to_categories:
        grouped = grouped.reindex(
            sorted(grouped.index, key=lambda x: benchmark_to_categories.get(x, "zzz"))
        )

    # Assign consistent colors to optimizers
    cmap = plt.get_cmap("tab10")  # Default color palette
    new_colors = {}
    for idx, optimizer in enumerate(sorted_optimizers):
        if optimizer not in OPTIMIZER_COLORS:
            new_colors[optimizer] = cmap(
                len(OPTIMIZER_COLORS) + len(new_colors)
            )  # Assign unique color
        else:
            new_colors[optimizer] = OPTIMIZER_COLORS[optimizer]
    OPTIMIZER_COLORS.update(new_colors)

    fig, ax = plt.subplots(figsize=(12, 9))
    # Plot bar chart

    grouped.plot(
        kind="bar",
        ax=ax,
        alpha=0.8,
        edgecolor="black",
        color=[OPTIMIZER_COLORS[optimizer] for optimizer in sorted_optimizers],
    )

    # Add dotted average line for each optimizer
    avg_scores = grouped.mean()
    for optimizer, avg in avg_scores.items():
        ax.axhline(
            y=avg,
            color=OPTIMIZER_COLORS[optimizer],
            linestyle="dotted",
            linewidth=1.5,
            label=f"{optimizer} Avg",
        )

    # Highlight benchmarks according to categories if mapping is provided
    if benchmark_to_categories:
        from matplotlib.patches import Patch

        category_colors = {}
        unique_categories = set(benchmark_to_categories.values())
        cmap_category = plt.get_cmap("Set2")
        for idx, category in enumerate(unique_categories):
            category_colors[category] = cmap_category(idx)

        for idx, benchmark in enumerate(grouped.index):
            if benchmark in benchmark_to_categories:
                category = benchmark_to_categories[benchmark]
                ax.get_xticklabels()[idx].set_backgroundcolor(category_colors[category])

        # Add category legend
        category_patches = [
            Patch(facecolor=color, label=category)
            for category, color in category_colors.items()
        ]
        fig.legend(
            handles=category_patches,
            title="Benchmark Categories",
            loc="lower left",
            bbox_to_anchor=(0, -0.05),
            ncol=len(category_patches),
            fontsize=10,
            title_fontsize=12,
        )

    # Set plot title, labels, and legend
    ax.set_title(
        f"Optimizer-Specific Benchmark Scores ({model}, {'all programs' if not programs else ', '.join(programs)})",
        fontsize=14,
    )
    ax.set_xlabel("Benchmark", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.legend(title="Optimizers", fontsize=10, title_fontsize=12, loc="upper left")

    # Adjust layout and save the plot
    plt.tight_layout()
    filename = f"{model}_optimizer_{'_'.join(optimizers)}_{'_'.join(programs)}.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.show()

    print(f"Plot saved as {filename}")


def plot_optimizer_specific_with_budget(
    data_df,
    optimizers,
    model,
    benchmark_to_categories=None,
    benchmark_categories=None,
    programs=[],
):
    """
    Plot optimizer-specific benchmark scores for specified optimizers.

    Args:
        data_df (pd.DataFrame): The input DataFrame containing benchmark data.
        optimizers (list): List of optimizers to include in the plot.
        model (str): Name of the model used in the experiment.
        benchmark_to_categories (dict, optional): A mapping from benchmarks to categories for highlighting.
        benchmark_categories (list, optional): List of benchmark categories to include.
        programs (list, optional): List of programs to filter the data.
    """
    # Filter benchmarks based on categories
    if benchmark_categories and benchmark_to_categories:
        selected_benchmarks = [
            b for b, c in benchmark_to_categories.items() if c in benchmark_categories
        ]
        data_df = data_df[data_df["benchmark"].isin(selected_benchmarks)]

    # Filter programs if provided
    if programs:
        data_df = data_df[data_df["program"].isin(programs)]

    # Filter optimizers based on the provided list
    data_df = data_df[data_df["optimizer"].isin(optimizers)]

    data_df["optimizer_cost"] = (
        data_df["optimizer_input_tokens"]
        + data_df["optimizer_output_tokens"]
        + data_df["input_tokens"]
        + data_df["output_tokens"]
    )
    # Sort optimizers based on the predefined order
    sorted_optimizers = [
        opt
        for opt in [
            "Baseline",
            "BootstrapFewShot",
            "MIPROv2",
            "MIPROv2+",
            "BootstrapFewShotWithRandomSearch",
        ]
        if opt in optimizers
    ]

    print(data_df[data_df["benchmark"] == "GSM8K"])
    # Group by benchmark and optimizer to calculate mean scores
    grouped = data_df.groupby(["benchmark", "optimizer"])["score"].mean().unstack()

    # Ensure all optimizers are represented in the DataFrame
    for optimizer in sorted_optimizers:
        if optimizer not in grouped.columns:
            grouped[optimizer] = float("nan")

    # Reorder optimizers and sort benchmarks
    grouped = grouped[sorted_optimizers]

    # Sort benchmarks by categories if mapping is provided
    if benchmark_to_categories:
        grouped = grouped.reindex(
            sorted(grouped.index, key=lambda x: benchmark_to_categories.get(x, "zzz"))
        )

    # Assign consistent colors to optimizers
    cmap = plt.get_cmap("tab10")  # Default color palette
    new_colors = {}
    for idx, optimizer in enumerate(sorted_optimizers):
        if optimizer not in OPTIMIZER_COLORS:
            new_colors[optimizer] = cmap(
                len(OPTIMIZER_COLORS) + len(new_colors)
            )  # Assign unique color
        else:
            new_colors[optimizer] = OPTIMIZER_COLORS[optimizer]
    OPTIMIZER_COLORS.update(new_colors)

    fig, ax = plt.subplots(figsize=(12, 9))
    # Plot bar chart

    grouped.plot(
        kind="bar",
        ax=ax,
        alpha=0.8,
        edgecolor="black",
        color=[OPTIMIZER_COLORS[optimizer] for optimizer in sorted_optimizers],
    )

    # Add dotted average line for each optimizer
    # avg_scores = grouped.mean()
    # for optimizer, avg in avg_scores.items():
    #     ax.axhline(y=avg, color=OPTIMIZER_COLORS[optimizer], linestyle='dotted', linewidth=1.5, label=f'{optimizer} Avg')

    # Highlight benchmarks according to categories if mapping is provided
    if benchmark_to_categories:
        from matplotlib.patches import Patch

        category_colors = {}
        unique_categories = set(benchmark_to_categories.values())
        cmap_category = plt.get_cmap("Set2")
        for idx, category in enumerate(unique_categories):
            category_colors[category] = cmap_category(idx)

        for idx, benchmark in enumerate(grouped.index):
            if benchmark in benchmark_to_categories:
                category = benchmark_to_categories[benchmark]
                ax.get_xticklabels()[idx].set_backgroundcolor(category_colors[category])

        # Add category legend
        category_patches = [
            Patch(facecolor=color, label=category)
            for category, color in category_colors.items()
        ]
        fig.legend(
            handles=category_patches,
            title="Benchmark Categories",
            loc="lower left",
            bbox_to_anchor=(0, -0.05),
            ncol=len(category_patches),
            fontsize=10,
            title_fontsize=12,
        )

    # ax2 = ax.twinx()
    # bar_width = 0.8 / len(sorted_optimizers)
    # for optimizer_idx, optimizer in enumerate(sorted_optimizers):
    #     cost_data = data_df[data_df['optimizer'] == optimizer].groupby('benchmark')['optimizer_cost'].mean()
    #     print(optimizer, cost_data)
    #     x_positions = np.arange(len(cost_data)) + optimizer_idx * (bar_width - 0.05) + (bar_width / 2) - 0.30
    #     ax2.scatter(x_positions, cost_data, label=f'{optimizer} Cost', marker='o', color="black")

    ax2 = ax.twinx()
    for benchmark in grouped.index:
        cost_data = (
            data_df[data_df["benchmark"] == benchmark]
            .groupby("optimizer")["optimizer_cost"]
            .mean()
        )
        cost_data = cost_data.reindex(sorted_optimizers)
        x_positions = [
            list(grouped.index).index(benchmark)
            + i * (0.8 / (len(sorted_optimizers)) - 0.06)
            - 0.20
            for i in range(len(sorted_optimizers))
        ]
        # ax2.plot(x_positions, cost_data, label=f'{benchmark} Cost', linestyle='-', marker='x', linewidth=1.5, color="black")
        ax2.scatter(x_positions, cost_data, color="black", zorder=5, marker="x", s=60)
        for x, cost in zip(x_positions, cost_data):
            ax2.plot([x, x], [0, cost], color="black", linestyle="-", linewidth=2)

    ax2.spines["bottom"].set_position(
        ("outward", 0)
    )  # Align the bottom spine of ax2 with ax1
    ax2.set_ylim(bottom=ax.get_ylim()[0])

    ax2.set_ylabel(
        "Optimization Cost (Total number of tokens, denoted by x)", color="black"
    )
    ax2.tick_params(axis="y", labelcolor="black")

    # Set plot title, labels, and legend
    ax.set_title(
        f"Optimizer-Specific Benchmark Scores ({model}, {'all programs' if not programs else ', '.join(programs)})",
        fontsize=14,
    )
    ax.set_xlabel("Benchmark", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.legend(title="Optimizers", fontsize=10, title_fontsize=12, loc="upper left")

    # Adjust layout and save the plot
    plt.tight_layout()
    filename = (
        f"{model}_optimizer_{'_'.join(optimizers)}_{'_'.join(programs)}_with_budget.png"
    )
    plt.savefig(filename, bbox_inches="tight")
    plt.show()

    print(f"Plot saved as {filename}")

def compare_programs(data_df, model, optimized=False):
    """
    Plot the performance comparison of each program against Predict and CoT.

    Args:
        data_df (pd.DataFrame): The input DataFrame containing benchmark data.
    """
    # Ensure the necessary columns exist
    required_columns = {"benchmark", "program", "score"}
    if not required_columns.issubset(data_df.columns):
        raise ValueError(f"The DataFrame must contain the following columns: {required_columns}")

    # filter out all baseline scores 
    data_df = data_df[data_df["optimizer"] == "Baseline"] if not optimized else data_df

    # Prepare results storage
    program_comparison = []

    # Iterate over unique programs
    for program in data_df["program"].unique():
        if program == "CoT" or program == "Predict" or program=="CoTBasedVote":
            continue
        # Filter data for the current program
        program_data = data_df[data_df["program"] == program]

        # Initialize counters
        better_than_predict = 0
        better_than_cot = 0
        total_benchmarks = 0


        # Compare with Predict and CoT for each benchmark
        valid_bench = 0
        predict_cost = 0
        cot_cost = 0
        program_cost = 0

        for benchmark in program_data["benchmark"].unique():
            
            # Get scores for the current benchmark
            scores = data_df[data_df["benchmark"] == benchmark]

            if "Predict" in scores["program"].values and "CoT" in scores["program"].values:
                valid_bench += 1

            if optimized:
                for optimizer in scores["optimizer"].unique():
                    total_benchmarks += 1
                    optimizer_scores = scores[scores["optimizer"] == optimizer]

                    # Program score for this optimizer
                    program_scores = optimizer_scores[optimizer_scores["program"] == program]["score"].values

                    # Predict comparison for this optimizer
                    if "Predict" in optimizer_scores["program"].values:
                        predict_scores = optimizer_scores[optimizer_scores["program"] == "Predict"]["score"].values
                        if any(program_score >= predict_score * 0.95 for program_score in program_scores for predict_score in predict_scores):
                            better_than_predict += 1

                    # CoT comparison for this optimizer
                    if "CoT" in optimizer_scores["program"].values:
                        cot_scores = optimizer_scores[optimizer_scores["program"] == "CoT"]["score"].values
                        if any(program_score >= cot_score * 0.95 for program_score in program_scores for cot_score in cot_scores):
                            better_than_cot += 1
            else:
                # Non-optimized: Use all scores
                program_scores = scores[scores["program"] == program]["score"].values

                # Predict comparison
                if "Predict" in scores["program"].values:
                    predict_scores = scores[scores["program"] == "Predict"]["score"].values
                    if any(program_score >= predict_score * 0.95 for program_score in program_scores for predict_score in predict_scores):
                        better_than_predict += 1

                # CoT comparison
                if "CoT" in scores["program"].values:
                    cot_scores = scores[scores["program"] == "CoT"]["score"].values
                    if any(program_score >= cot_score * 0.95 for program_score in program_scores for cot_score in cot_scores):
                        better_than_cot += 1
                total_benchmarks += 1

        if valid_bench == 0:
            continue

        print(optimized, program, better_than_predict, better_than_cot, total_benchmarks)

        # Calculate percentages
        program_comparison.append({
            "program": program,
            "better_than_predict": (better_than_predict / total_benchmarks) * 100 if total_benchmarks > 0 else 0,
            "better_than_cot": (better_than_cot / total_benchmarks) * 100 if total_benchmarks > 0 else 0,
        })

    # Convert results to a DataFrame
    comparison_df = pd.DataFrame(program_comparison)
    comparison_df = comparison_df.sort_values(by="program").reset_index(drop=True)


    # Plot the results
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(comparison_df))
    bar_width = 0.35

    ax.bar(
        [pos - bar_width / 2 for pos in x],
        comparison_df["better_than_predict"],
        width=bar_width,
        color="#56B4E9",
        label="Better/Within 5% (relatively) of Predict",
    )
    ax.bar(
        [pos + bar_width / 2 for pos in x],
        comparison_df["better_than_cot"],
        width=bar_width,
        color="#117733",
        label="Better/Within 5% (relatively) of CoT",
    )

    # Customize x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["program"], rotation=45, ha="right")
    ax.set_ylabel("Percentage")
    optimized = "optimized" if optimized else "unoptimized"
    ax.set_title(f"Program Performance Comparison Against Predict and CoT ({model}, {optimized})")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    filename = f"{model}_program_comparison_{optimized}.png"
    plt.savefig(filename, dpi=1000)
    print(f"saved plot {filename}")

# def compare_programs_merged(data_df, model):
#     """
#     Plot the performance comparison of each program against Predict and CoT
#     for both optimized and unoptimized settings.

#     Args:
#         data_df (pd.DataFrame): The input DataFrame containing benchmark data.
#         model (str): The name of the model used in the experiment.
#     """
#     # Ensure the necessary columns exist
#     required_columns = {"benchmark", "program", "score", "optimizer"}
#     if not required_columns.issubset(data_df.columns):
#         raise ValueError(f"The DataFrame must contain the following columns: {required_columns}")

#     # Helper function to calculate comparison data
#     def calculate_comparison(data_df, optimized):
#         filtered_df = data_df if optimized else data_df[data_df["optimizer"] == "Baseline"]
#         program_comparison = []

#         for program in filtered_df["program"].unique():
#             if program in {"CoT", "Predict", "CoTBasedVote"}:
#                 continue

#             program_data = filtered_df[filtered_df["program"] == program]
#             better_than_predict = 0
#             better_than_cot = 0
#             total_predict = 0
#             total_cot = 0

#             valid_bench = 0

#             predict_cost = 0
#             cot_cost = 0
#             program_cost = 0

#             for benchmark in program_data["benchmark"].unique():
#                 scores = filtered_df[filtered_df["benchmark"] == benchmark]

#                 if "Predict" in scores["program"].values and "CoT" in scores["program"].values:
#                     valid_bench += 1

#                 if optimized:
#                     for optimizer in scores["optimizer"].unique():
#                         optimizer_scores = scores[scores["optimizer"] == optimizer]
#                         program_score = optimizer_scores[optimizer_scores["program"] == program]["score"].values[0]
#                         # print(program_scores, optimizer, program)

#                         if "Predict" in optimizer_scores["program"].values:
#                             predict_data = optimizer_scores[optimizer_scores["program"] == "Predict"]
#                             predict_score = predict_data["score"].values[0]
#                             predict_cost = predict_data["input_tokens"] + predict_data["output_tokens"]
#                             if program_score >= predict_score * 0.95:
#                                 better_than_predict += 1
#                             total_predict += 1


#                         if "CoT" in optimizer_scores["program"].values:
#                             cot_score = optimizer_scores[optimizer_scores["program"] == "CoT"]["score"].values[0]
#                             if program_score >= cot_score * 0.95:
#                                 better_than_cot += 1
#                             total_cot += 1
#                 else:
#                     program_score = scores[(scores["program"] == program) & (scores["optimizer"]=="Baseline")]["score"].values[0]

#                     if "Predict" in scores["program"].values:
#                         predict_score = scores[(scores["program"] == "Predict") & (scores["optimizer"]=="Baseline")]["score"].values[0]
#                         if program_score >= predict_score * 0.95:
#                             better_than_predict += 1
#                         total_predict += 1

#                     if "CoT" in scores["program"].values:
#                         cot_score = scores[(scores["program"] == "CoT") & (scores["optimizer"]=="Baseline")]["score"].values[0]
#                         print(cot_score)
#                         if program_score >= cot_score * 0.95:
#                             better_than_cot += 1
#                         total_cot += 1

#             print(program, valid_bench)
            
#             if valid_bench == 0:
#                 continue

#             program_comparison.append({
#                 "program": program,
#                 f"better_than_predict_{'optimized' if optimized else 'unoptimized'}": (better_than_predict / total_predict) * 100 if total_predict > 0 else 0,
#                 f"better_than_cot_{'optimized' if optimized else 'unoptimized'}": (better_than_cot / total_cot) * 100 if total_cot > 0 else 0,
#             })

#         return pd.DataFrame(program_comparison)

#     # Calculate comparison data for both modes
#     unoptimized_data = calculate_comparison(data_df, optimized=False)
#     optimized_data = calculate_comparison(data_df, optimized=True)

#     # Merge the data on the "program" column
#     comparison_df = pd.merge(unoptimized_data, optimized_data, on="program", how="outer")

#     comparison_df = comparison_df.sort_values(by="program").reset_index(drop=True)


#     # Prepare data for plotting
#     programs = comparison_df["program"]
#     x_positions = range(len(programs))
#     bar_width = 0.2

#     # Define heights for bars
#     heights = [
#         comparison_df["better_than_predict_unoptimized"],
#         comparison_df["better_than_predict_optimized"],
#         comparison_df["better_than_cot_unoptimized"],
#         comparison_df["better_than_cot_optimized"],
#     ]

#     # Define offsets for grouped bars
#     offsets = [-1.5 * bar_width, -0.5 * bar_width, 0.5 * bar_width, 1.5 * bar_width]

#     # Define colors and labels
#     colors = ["#ADD8E6", "#00509E", "#FFDAB9", "#FF7F00"]
#     labels = [
#         "Better/Within 5% (relatively, same below) of Predict (unoptimized)",
#         "Better/Within 5% of Predict (optimized)",
#         "Better/Within 5% of CoT (unoptimized)",
#         "Better/Within 5% of CoT (optimized)",
#     ]
#     fig, ax = plt.subplots(figsize=(18, 10))

#     # Plot all bars in a single call
#     for height, offset, color, label in zip(heights, offsets, colors, labels):
#         ax.bar(
#             [pos + offset for pos in x_positions],
#             height,
#             width=bar_width,
#             color=color,
#             label=label,
#         )

#     # Customize the plot
#     ax.set_xticks(x_positions)
#     ax.set_xticklabels(programs, rotation=45, ha="right", fontsize=20)
#     ax.set_ylabel("Percentage", fontsize=20)
#     ax.set_title(f"Program Performance Comparison Against Predict and CoT ({model})", fontsize=26)
#     ax.legend(fontsize=14)
#     ax.grid(axis="y", linestyle="--", alpha=0.7)

#     plt.tight_layout()
#     filename = f"{model}_program_comparison_combined.png"
#     plt.savefig(filename, dpi=400)
#     print(f"Saved plot {filename}")



def compare_programs_merged(data_df, model, with_cost=False):
    """
    Plot the performance comparison of each program against Predict and CoT
    for both optimized and unoptimized settings, including relative cost gains.

    Args:
        data_df (pd.DataFrame): The input DataFrame containing benchmark data.
        model (str): The name of the model used in the experiment.
    """
    # Ensure the necessary columns exist
    required_columns = {"benchmark", "program", "score", "optimizer", "input_tokens", "output_tokens"}
    if not required_columns.issubset(data_df.columns):
        raise ValueError(f"The DataFrame must contain the following columns: {required_columns}")

    # Helper function to calculate comparison data
    def calculate_comparison(data_df, optimized):
        filtered_df = data_df if optimized else data_df[data_df["optimizer"] == "Baseline"]
        program_comparison = []

        for program in filtered_df["program"].unique():
            if program in {"CoT", "Predict", "CoTBasedVote"}:
                continue

            program_data = filtered_df[filtered_df["program"] == program]
            better_than_predict = 0
            better_than_cot = 0

            total_predict = 0
            total_cot = 0

            total_predict_cost = 0
            total_cot_cost = 0
            total_predict_program_cost = 0
            total_cot_program_cost = 0

            valid_bench = 0

            for benchmark in program_data["benchmark"].unique():
                scores = filtered_df[filtered_df["benchmark"] == benchmark]

                if not ("Predict" in scores["program"].values and "CoT" in scores["program"].values):
                    continue
                else:
                    valid_bench += 1

                optimizers = scores["optimizer"].unique()
                optimizers = [v for v in optimizers if v != "Baseline"] if optimized else ["Baseline"]
                    
                for optimizer in scores["optimizer"].unique():
                    optimizer_scores = scores[scores["optimizer"] == optimizer]

                    # Program cost for this optimizer under Predict branch
                    if "Predict" in optimizer_scores["program"].values:
                        predict_data = optimizer_scores[optimizer_scores["program"] == "Predict"]
                        predict_score = predict_data["score"].values[0]
                        predict_cost = predict_data["input_tokens"].values[0] + predict_data["output_tokens"].values[0]
                        total_predict_cost += predict_cost

                        program_cost = optimizer_scores[optimizer_scores["program"] == program]["input_tokens"].values[0] + \
                                        optimizer_scores[optimizer_scores["program"] == program]["output_tokens"].values[0]
                        total_predict_program_cost += program_cost

                        if optimizer_scores[optimizer_scores["program"] == program]["score"].values[0] >= predict_score * 0.95:
                            better_than_predict += 1
                        total_predict += 1

                    # Program cost for this optimizer under CoT branch
                    if "CoT" in optimizer_scores["program"].values:
                        cot_data = optimizer_scores[optimizer_scores["program"] == "CoT"]
                        cot_score = cot_data["score"].values[0]
                        cot_cost = cot_data["input_tokens"].values[0] + cot_data["output_tokens"].values[0]
                        total_cot_cost += cot_cost

                        program_cost = optimizer_scores[optimizer_scores["program"] == program]["input_tokens"].values[0] + \
                                        optimizer_scores[optimizer_scores["program"] == program]["output_tokens"].values[0]
                        total_cot_program_cost += program_cost

                        if optimizer_scores[optimizer_scores["program"] == program]["score"].values[0] >= cot_score * 0.95:
                            better_than_cot += 1
                        total_cot += 1

            if valid_bench == 0:
                continue

            program_comparison.append({
                "program": program,
                f"better_than_predict_{'optimized' if optimized else 'unoptimized'}": (better_than_predict / total_predict) * 100 if total_predict > 0 else 0,
                f"better_than_cot_{'optimized' if optimized else 'unoptimized'}": (better_than_cot / total_cot) * 100 if total_cot > 0 else 0,
                f"cost_gain_predict_{'optimized' if optimized else 'unoptimized'}": ((total_predict_program_cost) / total_predict_cost) * 100 if total_predict_cost > 0 else 0,
                f"cost_gain_cot_{'optimized' if optimized else 'unoptimized'}": ((total_cot_program_cost) / total_cot_cost) * 100 if total_cot_cost > 0 else 0,
            })


        return pd.DataFrame(program_comparison)

    # Calculate comparison data for both modes
    unoptimized_data = calculate_comparison(data_df, optimized=False)
    optimized_data = calculate_comparison(data_df, optimized=True)

    # Merge the data on the "program" column
    comparison_df = pd.merge(unoptimized_data, optimized_data, on="program", how="outer")

    # Sort programs alphabetically
    comparison_df = comparison_df.sort_values(by="program").reset_index(drop=True)

    # Prepare data for plotting
    programs = comparison_df["program"]
    x_positions = range(len(programs))
    bar_width = 0.2

    # Define heights for bars
    heights = [
        comparison_df["better_than_predict_unoptimized"],
        comparison_df["better_than_predict_optimized"],
        comparison_df["better_than_cot_unoptimized"],
        comparison_df["better_than_cot_optimized"],
    ]

    # Define offsets for grouped bars
    offsets = [-1.5 * bar_width, -0.5 * bar_width, 0.5 * bar_width, 1.5 * bar_width]

    # Define colors and labels
    colors = ["#ADD8E6", "#00509E", "#FFDAB9", "#FF7F00"]
    labels = [
        "Better/Within 5% (relatively, same below) of Predict (unoptimized)",
        "Better/Within 5% of Predict (optimized)",
        "Better/Within 5% of CoT (unoptimized)",
        "Better/Within 5% of CoT (optimized)",
    ]
    fig, ax = plt.subplots(figsize=(22, 12))

    # Plot all bars
    for height, offset, color, label in zip(heights, offsets, colors, labels):
        ax.bar(
            [pos + offset for pos in x_positions],
            height,
            width=bar_width,
            color=color,
            label=label,
        )

    # Plot relative cost gains as a line plot
    if with_cost:
        # Plot relative cost gains as points and connect them to zero
        ax2 = ax.twinx()
        cost_gain_colors = ["blue", "darkblue", "orange", "darkorange"]
        cost_gain_data = [
            comparison_df["cost_gain_predict_unoptimized"],  # Matches first bar group
            comparison_df["cost_gain_predict_optimized"],   # Matches second bar group
            comparison_df["cost_gain_cot_unoptimized"],     # Matches third bar group
            comparison_df["cost_gain_cot_optimized"],       # Matches fourth bar group
        ]
        cost_gain_offsets = [-1.5 * bar_width, -0.5 * bar_width, 0.5 * bar_width, 1.5 * bar_width]
        markers = ["o", "s", "o", "s"]  # Matches corresponding cost gain types
        labels = ["Cost relative to Predict (unoptimized)", "Cost relative to Predict (optimized)", "Cost relative to CoT (unoptimized)", "Cost relative to CoT (optimized)"]

        for data, offset, color, marker, label in zip(cost_gain_data, cost_gain_offsets, cost_gain_colors, markers, labels):
            positions = [pos + offset for pos in x_positions]

            # Draw thin lines from zero to the points
            for x, y in zip(positions, data):
                ax2.plot([x, x], [0, y], color=color, linewidth=0.8, alpha=0.7)  # Thin line from zero

            # Plot the points
            ax2.scatter(positions, data, color=color, label=label, marker=marker, s=100, edgecolors="black", alpha=0.8)
        ax2.set_ylabel("Relative cost (%)", fontsize=20)
        max_cost_gain = max(max(data) for data in cost_gain_data)
        ax2.set_ylim(0, max_cost_gain * 1.2) 

        ax2.legend(loc="upper right", fontsize=14)

    # Customize the plot
    ax.set_xticks(x_positions)
    ax.set_xticklabels(programs, rotation=45, ha="right", fontsize=20)
    ax.set_ylabel("Percentage of better performance experiment (%)", fontsize=20)
    cost_title = "and Cost Gains "if with_cost else ""
    ax.set_title(f"Program Performance {cost_title}Comparison ({model})", fontsize=26, pad=30)
    ax.legend(loc="upper left", fontsize=14)
    ax.set_ylim(0, 120)
    ax.set_yticks(range(0, 101, 20)) 

    
    
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    with_cost = "with_cost" if with_cost else "without_cost"
    filename = f"{model}_program_comparison_{with_cost}.png"
    plt.savefig(filename, dpi=400)
    print(f"Saved plot {filename}")
    plt.show()

# Benchmarks ['hover' 'IReRa' 'MMLU' 'GSM8K' 'HotpotQAConditional' 'HotpotQA'
#  'HeartDisease' 'MATH' 'Iris' 'RAGQAArena' 'SWEVerifiedAnnotationTask'
#  'Judge' 'HumanEval' 'Scone']

benchmark_to_categories = {
    "AppWorld": "Agent",
    "MATH": "MATH",
    "GSM8K": "MATH",
    "hover": "Summarization",
    "IReRa": "Knowledge",
    "HotpotQA": "Knowledge",
    "HotpotQAConditional": "Knowledge",
    "RAGQAArena": "Knowledge",
    "SWEVerifiedAnnotationTask": "Code",
    "Judge": "Reasoning",
    "HumanEval": "Code",
    "Scone": "Reasoning",
    "HeartDisease": "Classification",
    "Iris": "Classification",
    "MMLU": "Knowledge",
}

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
    if file_path.endswith("csv"):
        data_df = pd.read_csv(file_path)
        data_df = canonicalize_program(data_df)

    else:
        data_df = extract_information_from_files(file_path)
        data_df = canonicalize_program(data_df)
        data_df.to_csv(f"{file_path.split('/')[-1]}_data.csv", index=False)
        print(f"saved as {file_path.split('/')[-1]}_data.csv")

   
   
    print((data_df["benchmark"].unique()))

    import rich
    plot_best_program(data_df, "gpt-4o-mini")
    plot_best_program(data_df, "gpt-4o-mini", True)
    # compare_programs(data_df, "gpt-4o-mini")
    # compare_programs(data_df, "gpt-4o-mini", optimized=True)
    compare_programs_merged(data_df, "gpt-4o-mini", False)
    compare_programs_merged(data_df, "gpt-4o-mini", True)



    # Example usage
    # plot_program_specific(data_df, ['Predict', 'CoT'], "gpt-4o-mini", benchmark_to_categories)
    # plot_program_specific(data_df, ['CoT', 'RAG', 'SimplifiedBaleen'], "gpt-4o-mini",  benchmark_to_categories)
    # plot_program_specific(data_df, ['CoT', 'GeneratorCriticRanker', 'GeneratorCriticFuser'], "gpt-4o-mini", benchmark_to_categories)

    # plot_optimizer_specific(data_df, ["Baseline", "BootstrapFewShot", "BootstrapFewShotWithRandomSearch", "MIPROv2"], "gpt-4o-mini", benchmark_to_categories, ["Knowledge", "Summarization", "Classification", "Reasoning", "Code", "MATH"])
    # plot_optimizer_specific(data_df, ["Baseline", "BootstrapFewShot", "BootstrapFewShotWithRandomSearch", "MIPROv2"], "gpt-4o-mini", benchmark_to_categories, ["Knowledge"], ["Predict"])
    # plot_optimizer_specific(data_df, ["Baseline", "BootstrapFewShot", "BootstrapFewShotWithRandomSearch", "MIPROv2"], "gpt-4o-mini", benchmark_to_categories, ["Knowledge"], ["CoT"])
    # plot_optimizer_specific(data_df, ["Baseline", "BootstrapFewShot", "BootstrapFewShotWithRandomSearch", "MIPROv2"], "gpt-4o-mini", benchmark_to_categories, ["Knowledge"], ["RAG"])
    # plot_optimizer_specific(data_df, ["Baseline", "BootstrapFewShot", "BootstrapFewShotWithRandomSearch", "MIPROv2"], "gpt-4o-mini", benchmark_to_categories, ["Knowledge"], ["SimplifiedBaleen"])
    # plot_optimizer_specific(data_df, ["Baseline", "BootstrapFewShot", "BootstrapFewShotWithRandomSearch", "MIPROv2"], "gpt-4o-mini", benchmark_to_categories, ["Knowledge", "MATH", "Reasoning"], ["CoT"])
    # plot_optimizer_specific(data_df, ["Baseline", "BootstrapFewShot", "BootstrapFewShotWithRandomSearch", "MIPROv2"], "gpt-4o-mini", benchmark_to_categories, ["Knowledge", "MATH", "Reasoning"], ["Predict"])
    # plot_optimizer_specific(data_df, ["Baseline", "BootstrapFewShot", "BootstrapFewShotWithRandomSearch", "MIPROv2"], "gpt-4o-mini", benchmark_to_categories, ["Knowledge", "MATH", "Reasoning"], ["GeneratorCriticRanker"])
    # plot_optimizer_specific(data_df, ["Baseline", "BootstrapFewShot", "BootstrapFewShotWithRandomSearch", "MIPROv2"], "gpt-4o-mini", benchmark_to_categories, ["Knowledge", "MATH", "Reasoning"], ["GeneratorCriticFuser"])

    # plot_optimizer_specific_with_budget(
    #     data_df,
    #     [
    #         "Baseline",
    #         "BootstrapFewShot",
    #         "BootstrapFewShotWithRandomSearch",
    #         "MIPROv2",
    #         "MIPROv2+",
    #     ],
    #     "gpt-4o-mini",
    #     benchmark_to_categories,
    #     ["Knowledge", "Summarization", "Classification", "Reasoning"],
    #     ["CoT"],
    # )
