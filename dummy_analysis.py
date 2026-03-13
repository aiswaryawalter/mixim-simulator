import pandas as pd
import ast

df = pd.read_csv("Logs/DummyMessages.csv", index_col=0)

# Parse the DummyPr column from string representations of lists
pr_lists = df["DummyPr"].apply(ast.literal_eval)

# Convert to a DataFrame where each list element becomes a column
pr_df = pd.DataFrame(pr_lists.tolist())

# Sum each column
columnwise_sums = pr_df.sum()

print("Columnwise sum of pr_target for dummy messages:")
for i, s in enumerate(columnwise_sums):
    print(f"  target[{i}]: {s}")