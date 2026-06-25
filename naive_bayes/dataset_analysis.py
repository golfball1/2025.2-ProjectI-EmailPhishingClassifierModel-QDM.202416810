import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Read Data
df1 = pd.read_csv("CEAS_08.csv") 
df2 = pd.read_csv("mail_data.csv") 
dfs = [df1, df2]

# Data Analysis
for df in dfs:
    print("****Shape****")
    print(df.shape)
    print("\n****Info****")
    print(df.info())
    print("\n****Head****")
    print(df.head())
    print("\n****Null Counts****")
    print(df.isnull().sum())
    print("\n****Duplicate Rows****")
    print(df.duplicated().sum())
    if df is df1:
        print("\n****Label Distribution****")
        print(df['label'].value_counts())
        print("\n****URL Flag Distribution****")
        print(df['urls'].value_counts())
    else:
        print("\n****Label Distribution****")
        print(df['Category'].value_counts())

# Label Distribution Visualization
for df in dfs:
    if df is df1:
        ham_spam_counts = df['label'].value_counts()
    else:
        ham_spam_counts = df['Category'].value_counts()
    plt.figure(figsize=(12, 4))
    sns.barplot(x=ham_spam_counts.index, y=ham_spam_counts.values, palette='viridis')
    plt.title('Distribution of Ham and Spam Emails')
    plt.xticks(ticks=[0, 1], labels=['Ham', 'Spam'])
    plt.ylabel('Count')
    plt.xlabel('Email Type')
    plt.show()

    # Email Body Length Analysis
    if df is df1:
        mail_length = df['body'].str.len()
        plt.figure(figsize=(12, 6))
        sns.boxenplot(x=df['label'], y=mail_length, palette='Set2')
        plt.title('Email Body Length by Label')
        plt.xticks(ticks=[0, 1], labels=['Ham', 'Spam'])
        plt.ylabel('Email Body Length (characters)')
        plt.xlabel('Email Type')
        plt.show()
    else:
        mail_length = df['Message'].str.len()
        plt.figure(figsize=(12, 6))
        sns.boxenplot(x=df['Category'], y=mail_length, palette='Set2')
        plt.title('Email Body Length by Label')
        plt.xticks(ticks=[0, 1], labels=['Ham', 'Spam'])
        plt.ylabel('Email Body Length (characters)')
        plt.xlabel('Email Type')
        plt.show()

    # URL Presence Analysis
    url_counts = df1.groupby('label')['urls'].value_counts().unstack().fillna(0)

    ax = url_counts.plot(
        kind='bar',
        stacked=True,
        figsize=(12, 6),
        colormap='Set2'
    )

    ax.set_xticklabels(['Ham', 'Spam'], rotation=0)
    ax.legend(['No URL', 'Contains URL'])
    plt.title('URL Presence in Legitimate vs Phishing Emails')
    plt.xlabel('Email Type')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
    