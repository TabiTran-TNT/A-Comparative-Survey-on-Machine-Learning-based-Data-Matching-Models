# %%
import py_entitymatching as em
import os

######### LOAD TRAIN-TEST-VALIDATION #########
path_A = os.path.join(".", "sample_data", "abt-buy-textual", "learn_data", "tableA.csv")
path_B = os.path.join(".", "sample_data", "abt-buy-textual", "learn_data", "tableB.csv")
# Load the two tables.
print(path_A)
# output_attrs = ["Song_Name", "Artist_Name","Album_Name","Genre","Price","CopyRight","Time","Released"]
# output_attrs = ["title", "authors", "venue", "year"]
output_attrs = ["name", "description", "price"]
A = em.read_csv_metadata(path_A, key="id")
B = em.read_csv_metadata(path_B, key="id")
ob = em.OverlapBlocker()
C = ob.block_tables(
    A,
    B,
    # "Album_Name",
    # "Album_Name",
    # "venue",
    # "venue",
    "price",
    "price",
    l_output_attrs=output_attrs,
    r_output_attrs=output_attrs,
    overlap_size=2,
)
# %%
combined_columns = list(C.columns)[3:]
# %%
path_train = os.path.join(
    ".", "sample_data", "abt-buy-textual", "learn_data", "train.csv"
)
path_validation = os.path.join(
    ".", "sample_data", "abt-buy-textual", "learn_data", "valid.csv"
)
path_test = os.path.join(".", "sample_data", "abt-buy-textual", "learn_data", "test.csv")
# %%
import pandas as pd

train_df = pd.read_csv(path_train)
validation_df = pd.read_csv(path_validation)
test_df = pd.read_csv(path_test)
# %%
train_df.head(5)
# %%
merged_left_train = train_df.merge(A, left_on="ltable_id", right_on="id", how="left")
merged_left_train.rename(
    columns={
        # "Song_Name": "ltable_Song_Name",
        # "Artist_Name": "ltable_Artist_Name",
        # "Album_Name": "ltable_Album_Name",
        # "Genre": "ltable_Genre",
        # "Price": "ltable_Price",
        # "CopyRight": "ltable_CopyRight",
        # "Time": "ltable_Time",
        # "Released": "ltable_Released",
        # "title": "ltable_title",
        # "authors": "ltable_authors",
        # "venue": "ltable_venue",
        # "year": "ltable_year",
        "name": "ltable_name",
        "description": "ltable_description",
        "price": "ltable_price",
    },
    inplace=True,
)
# %%
merged_left_train.drop("id", axis=1, inplace=True)
merged_left_train.head(5)
# %%
merged_train_df = merged_left_train.merge(
    B, left_on="rtable_id", right_on="id", how="left"
)
merged_train_df.rename(
    columns={
        # "Song_Name": "rtable_Song_Name",
        # "Artist_Name": "rtable_Artist_Name",
        # "Album_Name": "rtable_Album_Name",
        # "Genre": "rtable_Genre",
        # "Price": "rtable_Price",
        # "CopyRight": "rtable_CopyRight",
        # "Time": "rtable_Time",
        # "Released": "rtable_Released",
        # "title": "rtable_title",
        # "authors": "rtable_authors",
        # "venue": "rtable_venue",
        # "year": "rtable_year",
        "name": "rtable_name",
        "description": "rtable_description",
        "price": "rtable_price",
    },
    inplace=True,
)
# %%
merged_train_df.drop("id", axis=1, inplace=True)
print(merged_train_df.columns)
# %%
merged_train_df.reset_index(inplace=True)
merged_train_df.rename(columns={"index": "_id"}, inplace=True)
merged_train_df.head(5)
# %%
save_path = os.path.join(".", "sample_data", "abt-buy-textual", "merged_train.csv")
merged_train_df.to_csv(save_path, index=False)
# %%
### TEST ###
merged_left_test = test_df.merge(A, left_on="ltable_id", right_on="id", how="left")
merged_left_test.rename(
    columns={
        # "Song_Name": "ltable_Song_Name",
        # "Artist_Name": "ltable_Artist_Name",
        # "Album_Name": "ltable_Album_Name",
        # "Genre": "ltable_Genre",
        # "Price": "ltable_Price",
        # "CopyRight": "ltable_CopyRight",
        # "Time": "ltable_Time",
        # "Released": "ltable_Released",
        # "title": "ltable_title",
        # "authors": "ltable_authors",
        # "venue": "ltable_venue",
        # "year": "ltable_year",
        "name": "ltable_name",
        "description": "ltable_description",
        "price": "ltable_price",
    },
    inplace=True,
)
merged_left_test.drop("id", axis=1, inplace=True)
merged_test_df = merged_left_test.merge(
    B, left_on="rtable_id", right_on="id", how="left"
)
merged_test_df.rename(
    columns={
        # "Song_Name": "rtable_Song_Name",
        # "Artist_Name": "rtable_Artist_Name",
        # "Album_Name": "rtable_Album_Name",
        # "Genre": "rtable_Genre",
        # "Price": "rtable_Price",
        # "CopyRight": "rtable_CopyRight",
        # "Time": "rtable_Time",
        # "Released": "rtable_Released",
        # "title": "rtable_title",
        # "authors": "rtable_authors",
        # "venue": "rtable_venue",
        # "year": "rtable_year",
        "name": "rtable_name",
        "description": "rtable_description",
        "price": "rtable_price",
    },
    inplace=True,
)
merged_test_df.drop("id", axis=1, inplace=True)
merged_test_df.reset_index(inplace=True)
merged_test_df.rename(columns={"index": "_id"}, inplace=True)
save_path = os.path.join(".", "sample_data", "abt-buy-textual", "merged_test.csv")
merged_test_df.to_csv(save_path, index=False)
# %%
### VALIDATION ###
merged_left_validation = validation_df.merge(
    A, left_on="ltable_id", right_on="id", how="left"
)
merged_left_validation.rename(
    columns={
        # "Song_Name": "ltable_Song_Name",
        # "Artist_Name": "ltable_Artist_Name",
        # "Album_Name": "ltable_Album_Name",
        # "Genre": "ltable_Genre",
        # "Price": "ltable_Price",
        # "CopyRight": "ltable_CopyRight",
        # "Time": "ltable_Time",
        # "Released": "ltable_Released",
        # "title": "ltable_title",
        # "authors": "ltable_authors",
        # "venue": "ltable_venue",
        # "year": "ltable_year",
        "name": "ltable_name",
        "description": "ltable_description",
        "price": "ltable_price",
    },
    inplace=True,
)
merged_left_validation.drop("id", axis=1, inplace=True)
merged_validation_df = merged_left_validation.merge(
    B, left_on="rtable_id", right_on="id", how="left"
)
merged_validation_df.rename(
    columns={
        # "Song_Name": "rtable_Song_Name",
        # "Artist_Name": "rtable_Artist_Name",
        # "Album_Name": "rtable_Album_Name",
        # "Genre": "rtable_Genre",
        # "Price": "rtable_Price",
        # "CopyRight": "rtable_CopyRight",
        # "Time": "rtable_Time",
        # "Released": "rtable_Released",
        # "title": "rtable_title",
        # "authors": "rtable_authors",
        # "venue": "rtable_venue",
        # "year": "rtable_year",
        "name": "rtable_name",
        "description": "rtable_description",
        "price": "rtable_price",
    },
    inplace=True,
)
merged_validation_df.drop("id", axis=1, inplace=True)
merged_validation_df.reset_index(inplace=True)
merged_validation_df.rename(columns={"index": "_id"}, inplace=True)
save_path = os.path.join(".", "sample_data", "abt-buy-textual", "merged_validation.csv")
merged_validation_df.to_csv(save_path, index=False)
###################################################
###################################################
# %%
feature_table = em.get_features_for_matching(A, B)
load_path = os.path.join(".", "sample_data", "abt-buy-textual", "merged_train.csv")
load_train = em.read_csv_metadata(
    load_path,
    key="_id",
    ltable=A,
    rtable=B,
    fk_ltable="ltable_id",
    fk_rtable="rtable_id",
)
# %%
import pandas as pd

load_path = os.path.join(".", "sample_data", "abt-buy-textual", "merged_train.csv")
# Select the attrs. to be included in the feature vector table
load_train_df = pd.read_csv(load_path)
attrs_from_table = list(load_train_df.columns[4:])
# %%
# Convert the labeled data to feature vectors using the feature table
train_H = em.extract_feature_vecs(
    load_train,
    feature_table=feature_table,
    attrs_before=attrs_from_table,
    attrs_after="label",
    show_progress=False,
)
# %%
train_H.head(5)
# %%
save_path = os.path.join(".", "sample_data", "abt-buy-textual", "fe_train.csv")
train_H.to_csv(save_path, index=False)
# %%
feature_table = em.get_features_for_matching(A, B)
##################################################
##################################################
valid_load_path = os.path.join(
    ".", "sample_data", "abt-buy-textual", "merged_validation.csv"
)
load_validation = em.read_csv_metadata(
    valid_load_path,
    key="_id",
    ltable=A,
    rtable=B,
    fk_ltable="ltable_id",
    fk_rtable="rtable_id",
)
validation_H = em.extract_feature_vecs(
    load_validation,
    feature_table=feature_table,
    attrs_before=attrs_from_table,
    show_progress=False,
)
# %%
save_path = os.path.join(".", "sample_data", "abt-buy-textual", "fe_valid.csv")
validation_H.to_csv(save_path, index=False)
# %%
##################################################
##################################################
test_load_path = os.path.join(".", "sample_data", "abt-buy-textual", "merged_test.csv")
load_test = em.read_csv_metadata(
    test_load_path,
    key="_id",
    ltable=A,
    rtable=B,
    fk_ltable="ltable_id",
    fk_rtable="rtable_id",
)
test_H = em.extract_feature_vecs(
    load_test,
    feature_table=feature_table,
    attrs_before=attrs_from_table,
    show_progress=False,
)
save_path = os.path.join(".", "sample_data", "abt-buy-textual", "fe_test.csv")
test_H.to_csv(save_path, index=False)
# %%


