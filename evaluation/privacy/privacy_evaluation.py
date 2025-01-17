import pandas as pd
from syntheval import SynthEval
from sklearn.model_selection import train_test_split

# Access datasets
df_train = pd.read_csv('')
df_train, df_test = train_test_split(df_train, test_size=0.3, random_state=42)

# The datasets supplied as a filepath
SYN_PATH = r''   
# Dictionary of metric configuration
metrics = {
    "mia_risk": {"num_eval_iter": 3},
    "att_discl": {}
}
class_cat_col = []
predict_class = 'label'

# Initialize SynthEval
SE = SynthEval(df_train, holdout_dataframe=df_test, cat_cols=class_cat_col)

# Run the benchmarking process
df_vals, df_rank = SE.benchmark(SYN_PATH, predict_class, rank_strategy='linear', **metrics)
