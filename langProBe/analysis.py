from collections import Counter, defaultdict
import numpy as np
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

optimizer_mapping = {
    "BootstrapFewShotInfer_20Rules_10Candidates": "RIO*",
    "BootstrapFewShotInfer_10Rules_10Candidates": "RIO",
    "MIPROv2": "MIPROv2",
    "MIPROv2+": "MIPROv2*",
    "Baseline": "Baseline",
    "BootstrapFewShot": "BootstrapFewShot",
    "BootstrapFewShotWithRandomSearch": "BootstrapFewShotWithRandomSearch",
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
    data_df["optimizer"] = data_df["optimizer"].replace(optimizer_mapping)
    data_df["benchmark"] = data_df["benchmark"].apply(lambda x: x.replace("Bench", ""))
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

def plot_best_program_combined(data_df, model, benchmark_to_categories):
    """
    Plot program-specific benchmark scores for Baseline, Best Unoptimized, and Optimized programs.
    Show relative improvement percentages on top of the non-baseline bars.

    Args:
        data_df (pd.DataFrame): The input DataFrame containing benchmark data.
        model (str): Name of the model used in the experiment.
    """
    # Filter data for baseline optimizer
    baseline_data = data_df[data_df["optimizer"] == "Baseline"]

    # Group by benchmarks to find the best unoptimized program (excluding Predict)
    best_unoptimized = (
        baseline_data[baseline_data["program"] != "Predict"]
        .groupby("benchmark", as_index=False)
        .apply(lambda group: group.loc[group["score"].idxmax()])
    ).rename(columns={"score": "unoptimized_score", "program": "unoptimized_program"})

    # Filter data for optimized programs
    optimized_data = data_df[data_df["optimizer"] != "Baseline"]
    best_optimized = (
        optimized_data[optimized_data["program"] != "Predict"]
        .groupby("benchmark", as_index=False)
        .apply(lambda group: group.loc[group["score"].idxmax()])
    ).rename(columns={"score": "optimized_score", "program": "optimized_program"})

    # Extract Predict scores
    predict_scores = baseline_data[
        (baseline_data["program"] == "Predict") | (baseline_data["program"] == "ReActBaseline")
    ][["benchmark", "score"]].rename(columns={"score": "baseline_score"})

    # Merge all data
    merged_data = (
        best_unoptimized.merge(best_optimized, on="benchmark", how="outer")
        .merge(predict_scores, on="benchmark", how="outer")
        .sort_values(by="benchmark")
        .reset_index(drop=True)
    )

    if benchmark_to_categories:
        merged_data["category"] = merged_data["benchmark"].map(benchmark_to_categories)
    else:
        merged_data["category"] = "Uncategorized"

    merged_data = merged_data.sort_values(by=["category", "benchmark"]).reset_index(drop=True)

    # Plotting
    fig, ax = plt.subplots(figsize=(13, 8))
    x_positions = np.arange(len(merged_data))

    # Bar width
    bar_width = 0.25

    # Plot bars
    ax.bar(
        x_positions - bar_width, 
        merged_data["baseline_score"], 
        width=bar_width, 
        color="#82B6A5", 
        label="Baseline"
    )
    ax.bar(
        x_positions, 
        merged_data["unoptimized_score"], 
        width=bar_width, 
        color="#FB9A99", 
        label="Best Unoptimized"
    )
    ax.bar(
        x_positions + bar_width, 
        merged_data["optimized_score"], 
        width=bar_width, 
        color="#E31A1C", 
        label="Best Optimized"
    )

    unique_categories = merged_data["category"].unique()
    category_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_categories)))  # Use a colormap for category colors
    category_to_color = dict(zip(unique_categories, category_colors))

    for i, category in enumerate(merged_data["category"]):
        ax.hlines(
            y=-0.15, 
            xmin=x_positions[i] - 0.5, 
            xmax=x_positions[i] + 0.5, 
            color=category_to_color[category], 
            linewidth=10
        )

    # Annotate percentage changes
    for i, row in merged_data.iterrows():
        baseline = row["baseline_score"]
        
        if pd.notna(row["unoptimized_score"]):
            if baseline == 0:
                # infinite percentage change
                unoptimized_change = float("inf")
            else:
                unoptimized_change = ((row["unoptimized_score"] - baseline) / baseline) * 100
            ax.text(
                x_positions[i],
                row["unoptimized_score"] + 0.02,  # Position above the bar
                f"{unoptimized_change:+.1f}%",
                fontsize=9,
                ha="center",
                color="black",
            )
        if pd.notna(row["optimized_score"]):
            if baseline == 0:
                # infinite percentage change
                optimized_change = float("inf")
            else:
                optimized_change = ((row["optimized_score"] - baseline) / baseline) * 100
            if optimized_change != float("inf") and optimized_change == unoptimized_change:
                continue
            if row["unoptimized_score"] + 0.4 >= row["optimized_score"] >= row["unoptimized_score"]:
                y_position = row["optimized_score"] + 0.06
            else:
                y_position = row["optimized_score"] + 0.04

            ax.text(
                x_positions[i] + bar_width,
                y_position,  # Position above the bar
                f"{optimized_change:+.1f}%",
                fontsize=9,
                ha="center",
                color="black",
            )

    # Set x-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(merged_data["benchmark"], rotation=45, ha="right", fontsize=10)

    ax.set_xlim(-0.5, len(merged_data) - 0.5)
    ax.set_ylim(-0.3, ax.get_ylim()[1])  # Extend y-axis to make space for the color bar
    # for category, color in category_to_color.items():
    #     ax.plot([], [], color=color, label=f"Category: {category}")

    fig.subplots_adjust(bottom=0.5)  # Adjust bottom margin to make space for additional axes

    # Add a lower axis for the category legend
    main_plot_bbox = ax.get_position()
    print(main_plot_bbox)
    category_ax = fig.add_axes([0.02, -0.02, main_plot_bbox.width+0.16, 0.04])  # Position at the bottom
    category_ax.axis("off")  # Turn off axis lines and labels

    # Calculate relative x-positions for each category based on benchmarks
    x_min, x_max = ax.get_xlim()
    category_ax.set_xlim(x_min, x_max)


    # Calculate normalized x-positions for each category
    category_positions = []
    for category in unique_categories:
        benchmarks_in_category = merged_data[merged_data["category"] == category]
        if not benchmarks_in_category.empty:
            start_index = benchmarks_in_category.index[0]
            x_pos = x_positions[start_index]  # Absolute x-position in the main plot
            category_positions.append((category, x_pos))
            print(category, x_pos)
    # category_positions[-1] = ("Reasoning", 1)
    
    category_ax.set_ylim(0, 1)

    # Custom placement for category legend
    for i, (category, relative_x) in enumerate(category_positions):
        category_ax.text(
            x=relative_x,  # Position based on the category's start
            y=0.5,  # Vertical alignment within the category_ax
            s=category,
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",  # Make the text bold
            color=category_to_color[category]
        )
        # # Add the underline
        # text_length = len(category) * 0.1  # Estimate length based on number of characters
        # category_ax.plot(
        #     [relative_x, relative_x + text_length],  # Center the underline
        #     [0.35, 0.35],  # Position below the text
        #     color=category_to_color[category],
        #     lw=1.5
        # )
    

    # Customize the plot
    ax.set_xlim(-0.5, len(merged_data) - 0.5)
    ax.set_ylim(-1, ax.get_ylim()[1])
    ax.set_ylabel("Score")
    ax.set_title(f"Baseline vs. Best Programs (Unoptimized and Optimized) for {model}")
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()

    filename = f"{model}_best_program_combined.png"
    plt.savefig(filename, bbox_inches="tight", dpi=400)

    print(f"Plot saved as {filename}")




def plot_best_program_combined_multi_lms(data_dfs, models, benchmark_to_categories):
    """
    Plot program-specific benchmark scores for Baseline, Best Unoptimized, and Optimized programs.
    Show relative improvement percentages on top of the non-baseline bars.

    Args:
        data_df (pd.DataFrame): The input DataFrame containing benchmark data.
        model (str): Name of the model used in the experiment.
    """
    def preprocess_data(data_df, model):
        # Filter data for baseline optimizer
        baseline_data = data_df[data_df["optimizer"] == "Baseline"]

        # Group by benchmarks to find the best unoptimized program (excluding Predict)
        best_unoptimized = (
            baseline_data[baseline_data["program"] != "Predict"]
            .groupby("benchmark", as_index=False)
            .apply(lambda group: group.loc[group["score"].idxmax()])
        ).rename(columns={"score": "unoptimized_score", "program": "unoptimized_program"})

        # Filter data for optimized programs
        optimized_data = data_df[data_df["optimizer"] != "Baseline"]
        best_optimized = (
            optimized_data[optimized_data["program"] != "Predict"]
            .groupby("benchmark", as_index=False)
            .apply(lambda group: group.loc[group["score"].idxmax()])
        ).rename(columns={"score": "optimized_score", "program": "optimized_program"})

        # Extract Predict scores
        predict_scores = baseline_data[
            (baseline_data["program"] == "Predict") | (baseline_data["program"] == "ReActBaseline")
        ][["benchmark", "score"]].rename(columns={"score": "baseline_score"})

        # Merge all data
        merged_data = (
            best_unoptimized.merge(best_optimized, on="benchmark", how="outer")
            .merge(predict_scores, on="benchmark", how="outer")
            .sort_values(by="benchmark")
            .reset_index(drop=True)
        )

        if benchmark_to_categories:
            merged_data["category"] = merged_data["benchmark"].map(benchmark_to_categories)
        else:
            merged_data["category"] = "Uncategorized"

        merged_data = merged_data.sort_values(by=["category", "benchmark"]).reset_index(drop=True)
        merged_data["model"] = model
        return merged_data

    # Identify benchmarks that exist in all models
    processed_dfs = [preprocess_data(data_df, model) for data_df, model in zip(data_dfs, models)]

    # common_benchmarks = set.intersection(*(set(df["benchmark"]) for df in processed_dfs))
    
    # # Keep only benchmarks present in all models
    # processed_dfs = [df[df["benchmark"].isin(common_benchmarks)] for df in processed_dfs]
    
    # Merge data across models
    merged_data = pd.concat(processed_dfs, ignore_index=True)

    # Compute average scores for baseline, unoptimized, and optimized
    avg_scores = merged_data.groupby("benchmark", as_index=False).agg(
        category=("category", "first"),
        avg_baseline_score=("baseline_score", "mean"),
        avg_unoptimized_score=("unoptimized_score", "mean"),
        avg_optimized_score=("optimized_score", "mean")
    )

    avg_scores = avg_scores.sort_values(by=["category", "benchmark"]).reset_index(drop=True)


    fig, ax = plt.subplots(figsize=(13, 8))
    x_positions = np.arange(len(avg_scores))

    # Bar width
    bar_width = 0.25

    # Plot bars
    ax.bar(
        x_positions - bar_width, 
        avg_scores["avg_baseline_score"], 
        width=bar_width, 
        color="#82B6A5", 
        label="Baseline"
    )
    ax.bar(
        x_positions, 
        avg_scores["avg_unoptimized_score"], 
        width=bar_width, 
        color="#FB9A99", 
        label="Best Unoptimized"
    )
    ax.bar(
        x_positions + bar_width, 
        avg_scores["avg_optimized_score"], 
        width=bar_width, 
        color="#E31A1C", 
        label="Best Optimized"
    )

    unique_categories = avg_scores["category"].unique()
    category_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_categories)))  # Use a colormap for category colors
    category_to_color = dict(zip(unique_categories, category_colors))

    for i, category in enumerate(avg_scores["category"]):
        ax.hlines(
            y=-0.15, 
            xmin=x_positions[i] - 0.5, 
            xmax=x_positions[i] + 0.5, 
            color=category_to_color[category], 
            linewidth=10
        )

    # Annotate percentage changes
    for i, row in avg_scores.iterrows():
        baseline = row["avg_baseline_score"]
        
        if pd.notna(row["avg_unoptimized_score"]):
            if baseline == 0:
                # infinite percentage change
                unoptimized_change = float("inf")
            else:
                unoptimized_change = ((row["avg_unoptimized_score"] - baseline) / baseline) * 100
            ax.text(
                x_positions[i],
                row["avg_unoptimized_score"] + 0.02,  # Position above the bar
                f"{unoptimized_change:+.1f}%",
                fontsize=9,
                ha="center",
                color="black",
            )
        if pd.notna(row["avg_optimized_score"]):
            if baseline == 0:
                # infinite percentage change
                optimized_change = float("inf")
            else:
                optimized_change = ((row["avg_optimized_score"] - baseline) / baseline) * 100
            if optimized_change != float("inf") and optimized_change == unoptimized_change:
                continue
            if row["avg_unoptimized_score"] + 0.4 >= row["avg_optimized_score"] >= row["avg_unoptimized_score"]:
                y_position = row["avg_optimized_score"] + 0.06
            else:
                y_position = row["avg_optimized_score"] + 0.04

            ax.text(
                x_positions[i] + bar_width,
                y_position,  # Position above the bar
                f"{optimized_change:+.1f}%",
                fontsize=9,
                ha="center",
                color="black",
            )

    # Set x-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(avg_scores["benchmark"], rotation=45, ha="right", fontsize=10)

    ax.set_xlim(-0.5, len(avg_scores) - 0.5)
    ax.set_ylim(-0.3, ax.get_ylim()[1])  # Extend y-axis to make space for the color bar
    # for category, color in category_to_color.items():
    #     ax.plot([], [], color=color, label=f"Category: {category}")

    fig.subplots_adjust(bottom=0.5)  # Adjust bottom margin to make space for additional axes

    # Add a lower axis for the category legend
    main_plot_bbox = ax.get_position()
    print(main_plot_bbox)
    category_ax = fig.add_axes([0.02, -0.02, main_plot_bbox.width+0.16, 0.04])  # Position at the bottom
    category_ax.axis("off")  # Turn off axis lines and labels

    # Calculate relative x-positions for each category based on benchmarks
    x_min, x_max = ax.get_xlim()
    category_ax.set_xlim(x_min, x_max)


    # Calculate normalized x-positions for each category
    category_positions = []
    for category in unique_categories:
        benchmarks_in_category = avg_scores[avg_scores["category"] == category]
        if not benchmarks_in_category.empty:
            start_index = benchmarks_in_category.index[0]
            x_pos = x_positions[start_index]  # Absolute x-position in the main plot
            category_positions.append((category, x_pos))
            print(category, x_pos)
    # category_positions[-1] = ("Reasoning", 1)
    
    category_ax.set_ylim(0, 1)

    # Custom placement for category legend
    for i, (category, relative_x) in enumerate(category_positions):
        category_ax.text(
            x=relative_x,  # Position based on the category's start
            y=0.5,  # Vertical alignment within the category_ax
            s=category,
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",  # Make the text bold
            color=category_to_color[category]
        )

    model_names = ", ".join(models)
    ax.set_xlim(-0.5, len(avg_scores) - 0.5)
    ax.set_ylim(-1, ax.get_ylim()[1])
    ax.set_ylabel("Score")
    ax.set_title(f"Baseline vs. Best Programs (Unoptimized and Optimized) for llama and gpt-4o-mini models")
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    model_names = "_".join(models)
    filename = f"{model_names}_best_program_combined.png"
    plt.savefig(filename, bbox_inches="tight", dpi=400)

    print(f"Plot saved as {filename}")



def plot_program_gains_category(data_df, model, benchmark_to_categories=None, category=["Knowledge"]):
    """
    Plot program-specific benchmark gains compared to the Predict program for a given category.

    Args:
        data_df (pd.DataFrame): The input DataFrame containing benchmark data.
        model (str): Name of the model used in the experiment.
        benchmark_to_categories (dict, optional): A mapping from benchmarks to categories for filtering.
        category (list): The categories to plot.
    """
    # Filter data for the specified category if benchmark_to_categories is provided
    if benchmark_to_categories:
        # category is a list
        category_benchmarks = [
            benchmark for benchmark, cat in benchmark_to_categories.items() if cat in category
        ]
        data_df = data_df[data_df["benchmark"].isin(category_benchmarks)]

    # Filter for unoptimized programs
    unoptimized_data = data_df[data_df["optimizer"] == "Baseline"]

    # Extract Predict scores
    predict_scores = unoptimized_data[unoptimized_data["program"] == "Predict"]
    predict_scores = predict_scores.set_index("benchmark")["score"]

    # Find max and min performing programs per benchmark (excluding Predict)
    performance_data = unoptimized_data[unoptimized_data["program"] != "Predict"].copy()
    max_performers = performance_data.loc[performance_data.groupby("benchmark")["score"].idxmax()]
    min_performers = performance_data.loc[performance_data.groupby("benchmark")["score"].idxmin()]

    # Compute gains over Predict

    # Compute gains over Predict with safety checks
    max_performers["gain"] = ((max_performers["score"] - predict_scores.loc[max_performers["benchmark"]].values) / 
                            predict_scores.loc[max_performers["benchmark"]].values * 100)

    min_performers["gain"] = ((min_performers["score"] - predict_scores.loc[min_performers["benchmark"]].values) / 
                            predict_scores.loc[min_performers["benchmark"]].values * 100)
 
    # Format labels as "benchmark\n(program)"
    max_performers["label"] = max_performers["benchmark"] + "\n(" + max_performers["program"] + ")"
    min_performers["label"] = min_performers["benchmark"] + "\n(" + min_performers["program"] + ")"

    # Merge data into a single DataFrame
        # Merge data into a single DataFrame
    merged_data = pd.concat([
        max_performers[["benchmark", "label", "gain", "program"]],
        min_performers[["benchmark", "label", "gain", "program"]]
    ]).drop_duplicates(subset=["benchmark", "program"], keep="first")  # Keep only the first occurrence if duplicate

    # Sorting so that all positive gains appear first, followed by negative gains
    merged_data = merged_data.sort_values(by="gain", ascending=False).reset_index(drop=True)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    x_positions = np.arange(len(merged_data))

    bar_width = 0.6

    # Bars with proper coloring
    bars = ax.bar(
        x_positions, 
        merged_data["gain"], 
        width=bar_width, 
        color=["blue" if g >= 0 else "red" for g in merged_data["gain"]]
    )

    # Annotate bars exceeding 200%
    for bar, gain, label in zip(bars, merged_data["gain"], merged_data["label"]):
        if gain > 100:
            if gain == float("inf"):
                # infinity sign
                ax.text(bar.get_x() + bar.get_width() / 2, 105, "∞", 
                    ha="center", va="top", fontsize=9, color="black")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, 105, f"{gain:.4g}", 
                    ha="center", va="top", fontsize=9, color="black")
            bar.set_height(100)  # Cap bar height at 200%
        elif gain < 0:
            ax.text(bar.get_x() + bar.get_width() / 2, 5, f"{gain:.4g}", 
                    ha="center", va="top", fontsize=9, color="black")
            if gain < -20:
                bar.set_height(-20)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, gain + 5, f"{gain:.4g}",
                    ha="center", va="top", fontsize=9, color="black")

    # Set x-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(merged_data["label"], rotation=45, ha="right", fontsize=10)

    # Customize the plot
    ax.set_xlim(-0.5, len(merged_data) - 0.5)
    ax.set_ylim(-20, 110)  # y-axis range fixed to -20 to 200 (annotations above 200)
    ax.set_ylabel("Gain (%) over Baseline")
    category_name= ", ".join(category) if len(category) > 1 else category[0]
    ax.set_title(f"Relative Gains Compared to Baseline ({category_name}, Unoptimized, gpt-4o-mini)")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    category_name = "_".join(category)
    filename = f"{model}_unoptimized_{category_name}_gains.png"
    plt.savefig(filename, bbox_inches="tight", dpi=400)

    print(f"Plot saved as {filename}")
  


def plot_cost_gains(data_df, model, benchmark_to_categories=None, category=["Knowledge"]):
    """
    Plot the relative cost gains (input_tokens + output_tokens) compared to the Baseline (Predict program),
    ensuring comparison only over common benchmarks.

    Args:
        data_df (pd.DataFrame): The input DataFrame containing benchmark data.
        model (str): Name of the model used in the experiment.
        benchmark_to_categories (dict, optional): A mapping from benchmarks to categories for filtering.
        category (list): List of categories to include in the analysis.
    """

    # Filter data based on category if specified
    if benchmark_to_categories:
        category_benchmarks = {bench for bench, cat in benchmark_to_categories.items() if cat in category}
        data_df = data_df[data_df["benchmark"].isin(category_benchmarks)]

    # Compute total cost (input_tokens + output_tokens)
    # only for the unoptimized programs
    data_df = data_df[data_df["optimizer"] == "Baseline"]
    data_df["cost"] = data_df["input_tokens"] + data_df["output_tokens"]

    # Filter out only programs that have Predict as a baseline
    predict_data = data_df[data_df["program"] == "Predict"]

    # Prepare a list to store valid cost comparisons
    cost_comparisons = []

    # Iterate over all programs (excluding Predict)
    for program in data_df["program"].unique():
        if program == "Predict":
            continue

        # Get benchmarks where this program has data
        program_benchmarks = set(data_df[data_df["program"] == program]["benchmark"])

        # Get Predict's cost **only** for those same benchmarks
        predict_subset = predict_data[predict_data["benchmark"].isin(program_benchmarks)]
        
        # Get program's total cost for those benchmarks
        program_subset = data_df[(data_df["program"] == program) & (data_df["benchmark"].isin(program_benchmarks))]

        # Ensure Predict has matching benchmarks; otherwise, skip
        if predict_subset.empty or program_subset.empty:
            print(f"Skipping {program} due to missing benchmarks in Predict data.")
            continue

        merged_costs = pd.merge(
            program_subset[["benchmark", "cost"]],
            predict_subset[["benchmark", "cost"]],
            on="benchmark",
            suffixes=("_program", "_predict")
        )

        # Compute relative gain for each benchmark
        merged_costs["relative_gain"] = ((merged_costs["cost_program"] - merged_costs["cost_predict"]) / merged_costs["cost_predict"]) * 100

        # Average gains across benchmarks
        avg_relative_gain = merged_costs["relative_gain"].mean()

        cost_comparisons.append({"program": program, "relative_gain": avg_relative_gain})

    # Convert to DataFrame
    total_costs = pd.DataFrame(cost_comparisons)
    print(total_costs)

    # Filter out negative gains (we only show programs that increase cost)
    total_costs = total_costs[total_costs["relative_gain"] > 0]

    # Sorting programs by cost gains (descending)
    total_costs = total_costs.sort_values(by="relative_gain", ascending=False).reset_index(drop=True)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    x_positions = np.arange(len(total_costs))

    # Bar width and color logic (all bars are blue since negatives are removed)
    bar_width = 0.6

    # Plot bars
    bars = ax.bar(
        x_positions, 
        total_costs["relative_gain"], 
        width=bar_width, 
        color="blue"
    )

    # Annotate bars with values
    for bar, gain, program in zip(bars, total_costs["relative_gain"], total_costs["program"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, 
            bar.get_height() + 2,  # Slightly above the bar
            f"{gain:.4g}%",  # Rounded to 3 significant figures
            ha="center", va="bottom", fontsize=10, color="black"
        )

    # Set x-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(total_costs["program"], rotation=45, ha="right", fontsize=10)

    # Customize the plot
    ax.set_xlim(-0.5, len(total_costs) - 0.5)
    ax.set_ylabel("Relative Cost Increase (%) over Baseline")
    ax.set_title(f"Cost Gains Compared to Baseline ({', '.join(category)}, {model})")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    filename = f"{model}_cost_gains_{'_'.join(category)}.png"
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

def compare_programs_merged(data_df, model, with_cost=False, excluding_programs=[]):
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
    
    if excluding_programs:
        data_df = data_df[~data_df["program"].isin(excluding_programs)]

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




def compare_programs_merged_performance_increase(data_df, model, with_cost=False, excluding_programs=[]):
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
    
    if excluding_programs:
        data_df = data_df[~data_df["program"].isin(excluding_programs)]

    # Helper function to calculate comparison data
    def calculate_comparison(data_df, optimized):
        filtered_df = data_df if optimized else data_df[data_df["optimizer"] == "Baseline"]
        program_comparison = []

        for program in filtered_df["program"].unique():
            if program in {"CoT", "Predict", "CoTBasedVote"}:
                continue

            program_data = filtered_df[filtered_df["program"] == program]
            predict_performance_increase = 0
            cot_performance_increase = 0

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

                        predict_performance_increase = optimizer_scores[optimizer_scores["program"] == program]["score"].values[0] - predict_score
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

                        cot_performance_increase += optimizer_scores[optimizer_scores["program"] == program]["score"].values[0] - cot_score
                        total_cot += 1

            if valid_bench == 0:
                continue

            program_comparison.append({
                "program": program,
                f"predict_performance_increase_{'optimized' if optimized else 'unoptimized'}": (predict_performance_increase / total_predict) if total_predict > 0 else 0,
                f"cot_performance_increase_{'optimized' if optimized else 'unoptimized'}": (cot_performance_increase / total_cot) if total_cot > 0 else 0,
                f"cost_gain_predict_{'optimized' if optimized else 'unoptimized'}": ((total_predict_program_cost) / total_predict_cost) if total_predict_cost > 0 else 0,
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
        comparison_df["predict_performance_increase_unoptimized"],
        comparison_df["predict_performance_increase_optimized"],
        comparison_df["cot_performance_increase_unoptimized"],
        comparison_df["cot_performance_increase_optimized"],
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
    ax.set_title(f"Program Performance Increase {cost_title} ({model})", fontsize=26, pad=30)
    ax.legend(loc="upper left", fontsize=14)
    # ax.set_ylim(0, 120)
    # ax.set_yticks(range(0, 101, 20)) 


    
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    with_cost = "with_cost" if with_cost else "without_cost"
    filename = f"{model}_program_comparison_performance_increase_{with_cost}.png"
    plt.savefig(filename, dpi=400)
    print(f"Saved plot {filename}")


optimizer_frequency = Counter()
top3_optimizer_frequency = Counter()
within_3_percent_frequency = Counter()
raw_relative_improvements = defaultdict(list)



def process_comparison_dfs(all_comparison_dfs, model_names):
    for name, comparison_dfs in all_comparison_dfs.items():
        print(f"Processing model: {name}")
        optimizer_frequency.clear()
        top3_optimizer_frequency.clear()
        within_3_percent_frequency.clear()
        raw_relative_improvements.clear()
        for key, df in comparison_dfs.items():
            process_single_comparison_df(df, name)
        # generate_and_plot_results(name)

def process_single_comparison_df(df, name):
    df['score'] = df['score'].astype(float).round(2)
    df['total_input_tokens'] = df['input_tokens'].astype(float) + df['optimizer_input_tokens'].astype(float)
    df['total_output_tokens'] = df['output_tokens'].astype(float) + df['optimizer_output_tokens'].astype(float)
    df[['total_input_tokens', 'total_output_tokens']] = df[['total_input_tokens', 'total_output_tokens']].round(2)

    df['total_tokens_sum'] = (df['total_input_tokens'] + df['total_output_tokens']).round(2)
    # df = df[['optimizer', 'score', 'total_input_tokens', 'total_output_tokens', 'total_tokens_sum']]
    df = df.sort_values(by=['score', 'optimizer', 'total_tokens_sum'], ascending=[False, True, True]).drop(columns=['total_tokens_sum'])
    for benchmark in df['benchmark'].unique():
        temp_df = df[df['benchmark'] == benchmark]
        for program in temp_df['program'].unique():
            temp_df_program = temp_df[temp_df['program'] == program]
            if len(temp_df) > 0:
                print(benchmark, program)
                highest_score = temp_df_program['score'].max()
                optimizer_frequency[temp_df_program.loc[temp_df_program['score'].idxmax(), 'optimizer']] += 1
                top3_optimizer_frequency.update(temp_df_program.head(3)['optimizer'])
                within_3_percent_frequency.update(temp_df_program[temp_df_program['score'] >= 0.97 * highest_score]['optimizer'])
    print(optimizer_frequency)
    plot_frequency_distribution(within_3_percent_frequency, "Optimizer within 3% of Highest Score on Task", "Frequency", name)
    plot_frequency_distribution(top3_optimizer_frequency, "Optimizer Ranks Top 3 Score on Task", "Frequency", name)
    plot_frequency_distribution(optimizer_frequency, "Optimizer Represents Top 1 Highest Score on Task", "Frequency", name)

    merge_and_plot_frequency_distributions(
        within_3_percent_frequency, 
        # top3_optimizer_frequency, 
        optimizer_frequency, 
        name
    )
    # if 'Baseline' in df['optimizer'].values:
    #     none_score = df.loc[df['optimizer'] == 'Baseline', 'score'].values[0]
    #     df['percentage_improvement'] = ((df['score'] - none_score) / none_score * 100).fillna(0) if none_score != 0 else 0
    #     for _, row in df.iterrows():
    #         if row['optimizer'] != 'Baseline':
    #             raw_relative_improvements[row['optimizer']].append(row['percentage_improvement'])

    

def generate_and_plot_results(converted_name):
    improvements_data = [{"Optimizer": opt, "Percentage Improvement": imp} for opt, imps in raw_relative_improvements.items() for imp in imps]
    improvements_df = pd.DataFrame(improvements_data)
    improvements_df = improvements_df[improvements_df["Optimizer"] != "BootstrapFewShot3_Teacher_gpt4o-mini"]

    stats_df = (
        improvements_df.groupby("Optimizer", as_index=False)
        .agg(p10=('Percentage Improvement', lambda x: x.quantile(0.1)),
             median=('Percentage Improvement', 'median'),
             p90=('Percentage Improvement', lambda x: x.quantile(0.9)))
        .round(2)
    )
    plot_frequency_distribution(within_3_percent_frequency, "Optimizer within 3% of Highest Score on Task", "Frequency", converted_name)
    percentiles = ['p10', 'median', 'p90']
    for percentile in reversed(percentiles):
        sorted_stats_df = stats_df.sort_values(by=percentile, ascending=False)
        sorted_stats_df['Optimizer'] = sorted_stats_df['Optimizer'].str.replace(r'gpt4o', 'gpt4o-mini', regex=True)
        filtered_stats_df = stats_df[stats_df['Optimizer'] != "BootstrapFewShot3_Teacher_gpt4o-mini"].sort_values(by=percentile, ascending=False)
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=filtered_stats_df,
            x='Optimizer',
            y=percentile,
            order=filtered_stats_df['Optimizer'].unique(),
            alpha=0.7,
            color="blue"
        )
        plt.title(f"{percentile.capitalize()} Relative Percentile Gains by DSPy Optimizer")
        plt.xlabel('Optimizer')
        plt.ylabel(f"Relative Gain%")
        plt.xticks(rotation=45, ha='right')
        plt.text(0.95, 0.95, f"Model = {converted_name}", transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
        plt.tight_layout()
        plt.savefig(f"{converted_name}_{percentile}_percentile_relative_gains.png", dpi=400)

def plot_frequency_distribution(freq_counter, title, ylabel, converted_name):
    df = pd.DataFrame.from_dict(freq_counter, orient='index', columns=['Frequency']).sort_values(by='Frequency', ascending=False)

    plt.figure(figsize=(10, 6))
    ax = df.plot(kind='bar', legend=False, color='green')
    ax.set_title(title)
    ax.set_xlabel("Optimizer")
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(df.index, rotation=45, ha='right')
    plt.text(0.95, 0.95, f"Model = {converted_name}", transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
    plt.tight_layout()
    plt.savefig(f"{converted_name}_{title.replace(' ', '_').replace('%', 'percent')}.png", dpi=400)
    

def merge_and_plot_frequency_distributions(
    within_3_percent_frequency, 
    # top3_optimizer_frequency, 
    optimizer_frequency, 
    converted_name
):
    # Consolidate frequency data into a single DataFrame
    merged_data = pd.DataFrame({
        "Within 3% of Highest Score": pd.Series(within_3_percent_frequency),
        # "Top 3 Score": pd.Series(top3_optimizer_frequency),
        "Highest Score": pd.Series(optimizer_frequency)
    }).fillna(0).astype(int)  # Fill missing values with 0 and convert to integer

    # Sort by Highest Score (optimizer_frequency)
    merged_data = merged_data.sort_values(by=["Highest Score", "Within 3% of Highest Score"], ascending=False)

    # Plot the merged bar chart
    plt.figure(figsize=(12, 8))
    colors = np.where(np.arange(len(merged_data)) % 2 == 0, "blue", "green")

    # Create bar plot with alternating colors
    ax = merged_data.plot(kind='bar', figsize=(12, 8), width=0.8, color=colors)

    # Customizing the plot
    ax.set_title("Frequency Distribution of Optimizers", fontsize=14)
    ax.set_xlabel("Optimizer", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_xticklabels(merged_data.index, rotation=45, ha='right', fontsize=10)
    ax.legend(title="Frequency Type", fontsize=10, loc="upper right")
    plt.text(
        0.95, 0.80, f"Model = {converted_name}", transform=plt.gca().transAxes,
        fontsize=10, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
    )
    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig(f"{converted_name}_merged_frequency_distribution.png", dpi=400)
    print(f"Saved plot {converted_name}_merged_frequency_distribution.png")

# Benchmarks ['hover' 'IReRa' 'MMLU' 'GSM8K' 'HotpotQAConditional' 'HotpotQA'
#  'HeartDisease' 'MATH' 'Iris' 'RAGQAArena' 'SWEVerifiedAnnotationTask'
#  'Judge' 'HumanEval' 'Scone']

def ensure_data_df(data_df):
    # for all data_df["optimizer"] that is not a string, name it "Baseline"
    data_df["optimizer"] = data_df["optimizer"].apply(lambda x: "Baseline" if not isinstance(x, str) else x)
    # if data_df do not have benchmark and program columns, create them
    if "benchmark" not in data_df.columns or "program" not in data_df.columns:
        if "file_name" in data_df.columns:
            data_df["benchmark"] = data_df["file_name"].apply(lambda x: x.split("_")[0])
            data_df["program"] = data_df["file_name"].apply(lambda x: x.split("_")[1])
        else:
            data_df["benchmark"] = data_df["filename"].apply(lambda x: x.split("_")[0])
            data_df["program"] = data_df["filename"].apply(lambda x: x.split("_")[1])
    return data_df
        
    

benchmark_to_categories = {
    "AppWorld": "Agent",
    "MATH": "Math",
    "GSM8K": "Math",
    "hover": "Knowledge",
    "IReRa": "Knowledge",
    "HotpotQA": "Knowledge",
    "HotpotQAConditional": "Knowledge",
    "RAGQAArena": "Knowledge",
    "SWEUnderspecified": "Code",
    "SWEValidity": "Code",
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
        help="Path to the text files containing benchmark results",
    )

    args.add_argument(
        "--file_paths",
        nargs="+",
    )

    args.add_argument(
        "--plot_type",
        type=str,
        default="program",
        help="Type of plot to generate (program, optimizer)",
    )

    args.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name used in the experiment",
    )

    args.add_argument(
        "--models",
        nargs="+",
    )

    args = args.parse_args()

    file_path = args.file_path
    if file_path:
        if file_path.endswith("csv"):
            data_df = pd.read_csv(file_path)
            ensure_data_df(data_df)
            data_df = canonicalize_program(data_df)

        else:
            data_df = extract_information_from_files(file_path)
            data_df = canonicalize_program(data_df)
            data_df.to_csv(f"{file_path.split('/')[-1]}_data.csv", index=False)
            print(f"saved as {file_path.split('/')[-1]}_data.csv")

    
    
        data_df = data_df[~((data_df["program"] == "GeneratorCriticFuser20") | (data_df["program"] == "GeneratorCriticRanker20"))]

        if args.plot_type == "program": 
            plot_cost_gains(data_df, args.model, benchmark_to_categories, ["Knowledge", "Reasoning", "Math"])
            plot_cost_gains(data_df, args.model, benchmark_to_categories)


            plot_program_gains_category(data_df, args.model, benchmark_to_categories)
            
            plot_program_gains_category(data_df, args.model, benchmark_to_categories, ["Code", "Math"])
            plot_program_gains_category(data_df, args.model, benchmark_to_categories, ["Knowledge", "Reasoning", "Math"])
            plot_program_gains_category(data_df, args.model, benchmark_to_categories, ["Classification"])


            plot_best_program_combined(data_df, args.model, benchmark_to_categories)

            compare_programs_merged(data_df, args.model, False)
            compare_programs_merged(data_df, args.model, True, ["GeneratorCriticRanker20", "GeneratorCriticFuser20"])

            compare_programs_merged_performance_increase(data_df, args.model, False)
            compare_programs_merged_performance_increase(data_df, args.model, True, ["GeneratorCriticRanker20", "GeneratorCriticFuser20"])

        if args.plot_type == "optimizer":
            all_comparison_dfs = {
                args.model: {
                    args.model: data_df,
                }
            }
            process_comparison_dfs(all_comparison_dfs, args.model)


    file_paths = args.file_paths
    if file_paths:
        models = args.models
        if not models:
            raise ValueError("Please provide model names for the given file paths.")
        
        if len(file_paths) != len(models):
            raise ValueError("The number of file paths and model names must match.")
    
        all_comparison_dfs = []
        for file_path, model in zip(file_paths, models):
            if file_path.endswith("csv"):
                data_df = pd.read_csv(file_path)
                ensure_data_df(data_df)
                data_df = canonicalize_program(data_df)

            else:
                data_df = extract_information_from_files(file_path)
                data_df = canonicalize_program(data_df)
                data_df.to_csv(f"{file_path.split('/')[-1]}_data.csv", index=False)
                print(f"saved as {file_path.split('/')[-1]}_data.csv")

            all_comparison_dfs.append(data_df)
        
        plot_best_program_combined_multi_lms(all_comparison_dfs, models, benchmark_to_categories)
        