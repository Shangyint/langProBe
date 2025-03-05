import pathlib

import pandas as pd


def read_evaluation_results(dir: str):
    # Define the path to the directory
    file_path = pathlib.Path(dir)

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
    df = canonicalize_program(df)
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
    "UnderspecifiedAnnotationGeneratorCriticFuser": "GeneratorCriticFuser",
    "UnderspecifiedAnnotationGeneratorCriticRanker": "GeneratorCriticRanker",
    "EvaluationValidityGeneratorCriticRanker": "GeneratorCriticRanker",
    "EvaluationValidityGeneratorCriticFuser": "GeneratorCriticFuser",
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
        data_df["program"].isin(
            [
                "UnderspecifiedAnnotationCoT",
                "UnderspecifiedAnnotationPredict",
                "UnderspecifiedAnnotationGeneratorCriticFuser",
                "UnderspecifiedAnnotationGeneratorCriticRanker",
            ]
        ),
        "benchmark",
    ] = "SWEBenchUnderspecified"

    data_df.loc[
        data_df["program"].isin(
            [
                "EvaluationValidityCoT",
                "EvaluationValidityPredict",
                "EvaluationValidityGeneratorCriticFuser",
                "EvaluationValidityGeneratorCriticRanker",
            ]
        ),
        "benchmark",
    ] = "SWEBenchValidity"
    data_df["program"] = data_df["program"].replace(program_mapping)
    data_df["benchmark"] = data_df["benchmark"].apply(lambda x: x.replace("Bench", ""))
    return data_df
