import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer


def preprocess(path1, path2):
    
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    
    text_cols1 = ["subject", "sender", "body"]
    df1_normalized = pd.DataFrame()
    df1_normalized['body'] = df1[text_cols1].fillna('').agg(' '.join, axis=1)
    df1_normalized['urls'] = df1['urls']
    df1_normalized['label'] = df1['label']
    
    text_cols2 = ["Message"]
    df2_normalized = pd.DataFrame()
    df2_normalized['body'] = df2[text_cols2].fillna('').agg(' '.join, axis=1)
    df2_normalized['urls'] = 0 
    df2_normalized['label'] = (df2['Category'] == 'spam').astype(int)

    df_merged = pd.concat([df1_normalized, df2_normalized], axis=0, ignore_index=True)

    print("--- MERGED INFO ---")
    print(df_merged.info())
    
    return df_merged