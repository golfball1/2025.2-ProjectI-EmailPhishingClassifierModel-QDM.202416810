from sklearn.metrics import fbeta_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import joblib
from sklearn.model_selection import train_test_split 
from preprocess import preprocess



# Load model đã train sẵn
best_model = joblib.load("best_model.pkl")
processor = joblib.load("processor.pkl")

# skip training
df = preprocess("CEAS_08.csv", "mail_data.csv")

_, X_testdf, _, y_test = train_test_split(
    df[['body', 'urls']],
    df['label'],
    test_size=0.2,
    stratify=df['label'],
    random_state=42
)

X_test = processor.transform(X_testdf)

#------------------------------------------------------------------
for custom_threshold in (0.5, 0.75, 0.99):
    print(f"For threshold of {custom_threshold}:")
    y_prob = best_model.predict_proba(X_test)[:, 1] # predict all emails in test set, put results in Nx1 matrix y_prob.

    y_pred = (y_prob >= custom_threshold).astype(int) # only make it spam if 50, 75, 85, 95 then 99% sure.

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    print("F0.5 Score:", fbeta_score(y_test, y_pred, beta=0.5))
    print("F1 Score:", fbeta_score(y_test, y_pred, beta=1))
    print("F2 Score:", fbeta_score(y_test, y_pred, beta=2))
    print()
    
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.show()
    
vectorizer = processor.named_transformers_['text']

print(len(vectorizer.vocabulary_))