# %%
# 0. SET UP
import py_entitymatching as em

# %%
import os
# output_attrs = ["Song_Name", "Artist_Name","Album_Name","Genre","Price","CopyRight","Time","Released"]
# output_attrs = ["title", "authors", "venue", "year"]
output_attrs = ["name", "description", "price"]

path_A = os.path.join(".", "sample_data", "abt-buy-textual", "learn_data", "tableA.csv")
path_B = os.path.join(".", "sample_data", "abt-buy-textual", "learn_data", "tableB.csv")
# Load the two tables.
A = em.read_csv_metadata(path_A, key="id")
B = em.read_csv_metadata(path_B, key="id")
# %%
# Basic information about the tables.
print("Number of tuples in A: " + str(len(A)))
print("Number of tuples in B: " + str(len(B)))
print("Number of tuples in A X B (i.e the cartesian product): " + str(len(A) * len(B)))
# %%
# The first few tuples in table A.
A.head()
# %%
# The first few tuples in table B.
B.head()
# %%
# 2. Block tables to get the candidate set
# Create an overlap blocker in Magellan and apply it to A and B to get the candidate set K1 which is in the format of
# a dataframe. The "l_out_attrs" and "r_out_attrs" parameters indicate the columns that will be included in K1 from A
# and B respectively.
ob = em.OverlapBlocker()
K1 = ob.block_tables(
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
# The number of tuple pairs in K1.
len(K1)
# %%
# The first few tuple pairs in K1
K1.head()
# %%
# Create a new overlap blocker to remove pairs from K1 that have no common word in "Artist_Name".

# K2 = ob.block_candset(K1, "Artist_Name", "Artist_Name", overlap_size=1)
# K2 = ob.block_candset(K1, "authors", "authors", overlap_size=1)
K2 = ob.block_candset(K1, "description", "description", overlap_size=1)


# %%
# The number of tuple pairs in K2.
len(K2)
# %%
# Apply the third overlap blocker.

# K3 = ob.block_candset(K2, "Song_Name", "Song_Name", overlap_size=1)
# K3 = ob.block_candset(K2, "title", "title", overlap_size=1)
K3 = ob.block_candset(K2, "name", "name", overlap_size=1)


# %%
# The number of tuples pairs in K3.
len(K3)
# %%
path_K = os.path.join(".", "sample_data", "abt-buy-textual", "candidate.csv")
K3.to_csv(path_K, index=False)
# %%
# 3. Match tuple pairs in the candidate set
## Sample and label the candidate set
# Take a sample of 500 pairs from the candidate set.
S = em.sample_table(K3, 500)
# %%
# Label the sample S in a GUI. Enter 1 for match and 0 for non-match.
G = em.label_table(S, "gold")
# %%
# The path to the labeled data file.
path_G = os.path.join(".", "sample_data", "abt-buy-textual", "gold.csv")
# %%
G.to_csv(path_G, index=False)
# %%
# Load the labeled data into a dataframe.
G = em.read_csv_metadata(
    path_G, key="_id", ltable=A, rtable=B, fk_ltable="ltable_id", fk_rtable="rtable_id"
)
print("Number of labeled pairs:", len(G))
# %%
feature_table = em.get_features_for_matching(A, B, validate_inferred_attr_types=False)
# %%
feature_table
# %%
G.columns[3:-1]