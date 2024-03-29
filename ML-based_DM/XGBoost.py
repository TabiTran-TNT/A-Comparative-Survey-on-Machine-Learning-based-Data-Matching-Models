# %%
import py_entitymatching as em
import pandas as pd
from sklearn.metrics import f1_score
import os

# %%
fe_train_path = os.path.join(".", "sample_data", "abt-buy-textual", "fe_train.csv")
load_fe_train = em.read_csv_metadata(
    fe_train_path,
    key="_id",
)
# attrs_from_table = list(load_fe_train.columns[3:19])
# attrs_from_table = list(load_fe_train.columns[3:11])
attrs_from_table = list(load_fe_train.columns[3:9])

attrs_to_be_excluded = []
attrs_to_be_excluded.extend(["_id", "ltable_id", "rtable_id", "label"])
attrs_to_be_excluded.extend(attrs_from_table)

fe_valid_path = os.path.join(".", "sample_data", "abt-buy-textual", "fe_valid.csv")
load_fe_valid = em.read_csv_metadata(
    fe_valid_path,
    key="_id",
)

fe_test_path = os.path.join(".", "sample_data", "abt-buy-textual", "fe_test.csv")
load_fe_test = em.read_csv_metadata(
    fe_test_path,
    key="_id",
)

# Get the attributes to be excluded while predicting
attrs_to_be_excluded_eval = []
attrs_to_be_excluded_eval.extend(["_id", "ltable_id", "rtable_id"])
attrs_to_be_excluded_eval.extend(attrs_from_table)

## load grouth truths
true_valid_path = os.path.join(
    ".", "sample_data", "abt-buy-textual", "learn_data", "valid.csv"
)
true_valid_df = pd.read_csv(true_valid_path)
true_test_path = os.path.join(
    ".", "sample_data", "abt-buy-textual", "learn_data", "test.csv"
)
true_test_df = pd.read_csv(true_test_path)
# %%
params = {
    "learning_rate": [0.1, 0.01, 0.001],
    "max_depth": [3, 5, 7],
    "n_estimators": [100, 500, 1000],
}
best_score = 0
best_params = {}
for learning_rate in params["learning_rate"]:
    for max_depth in params["max_depth"]:
        for n_estimators in params["n_estimators"]:
            rf = em.XGBoostMatcher(
                learning_rate=learning_rate,
                max_depth=max_depth,
                n_estimators=n_estimators,
            )
            rf.fit(
                table=load_fe_train,
                exclude_attrs=attrs_to_be_excluded,
                target_attr="label",
            )
            # evaluation on validation set
            predictions = rf.predict(
                table=load_fe_valid,
                exclude_attrs=attrs_to_be_excluded_eval,
                append=True,
                target_attr="predicted",
                inplace=False,
            )
            true_label = list(true_valid_df["label"])
            predicted_label = list(predictions["predicted"])
            f1 = f1_score(true_label, predicted_label)
            if f1 > best_score:
                best_score = f1
                best_params = {
                    "learning_rate": learning_rate,
                    "max_depth": max_depth,
                    "n_estimators": n_estimators,
                }
print("Best Score:", best_score)
print("Best Parameters:", best_params)
# %%
###########################################
# Predict using trained matcher on test set
###########################################
rf_last = em.XGBoostMatcher(
    learning_rate=best_params["learning_rate"],
    max_depth=best_params["max_depth"],
    n_estimators=best_params["n_estimators"],
)
rf_last.fit(
    table=load_fe_train,
    exclude_attrs=attrs_to_be_excluded,
    target_attr="label",
)
# evaluation on validation set
predictions = rf_last.predict(
    table=load_fe_test,
    exclude_attrs=attrs_to_be_excluded_eval,
    append=True,
    target_attr="predicted",
    inplace=False,
)
true_label = list(true_test_df["label"])
predicted_label = list(predictions["predicted"])
f1 = f1_score(true_label, predicted_label)
print("F1 Score: " + str(f1))
# %%
attrs_proj = []
attrs_proj.extend(["_id", "ltable_id", "rtable_id"])
attrs_proj.extend(attrs_from_table)
attrs_proj.append("predicted")
# Project the attributes
predictions = predictions[attrs_proj]
predictions.head()
# %%
##F1 Score: 0.912280701754386