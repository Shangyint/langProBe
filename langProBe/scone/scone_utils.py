import glob
import pandas as pd
import dspy
import os

def load_scone(dirname):
    dfs = []
    for filename in glob.glob(dirname + "/*.csv"):
        df = pd.read_csv(filename, index_col=0)
        df["category"] = os.path.basename(filename).replace(".csv", "")
        dfs.append(df)
    data_df = pd.concat(dfs)

    def as_example(row):
        # The 'one_scoped' file is from an earlier dataset, MoNLI, and
        # so is formatted a bit differently:
        suffix = "" if row["category"] == "one_scoped" else "_edited"
        # Reformat the hypothesis to be an embedded clause in a question:
        hkey = "sentence2" + suffix
        question = row[hkey][0].lower() + row[hkey][1:].strip(".")
        question = f"Can we logically conclude for sure that {question}?"
        # Binary task formulation:
        label = "Yes" if row["gold_label" + suffix] == "entailment" else "No"
        return dspy.Example(
            {
                "context": row["sentence1" + suffix],
                "question": question,
                "answer": label,
                "category": row["category"],
            }
        ).with_inputs("context", "question")

    return list(data_df.apply(as_example, axis=1).values)