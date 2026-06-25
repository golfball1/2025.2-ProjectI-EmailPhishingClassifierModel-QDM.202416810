import time
import joblib

from preprocess import ColumnTransformer, TfidfVectorizer, preprocess

from sklearn.metrics import make_scorer, fbeta_score

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split
)

df = preprocess("CEAS_08.csv", "mail_data.csv")

X_train, X_test, y_train, y_test = train_test_split(
    df[['body', 'urls']],
    df['label'],
    test_size=0.2,
    stratify=df['label'],
    random_state=42
)

processor = ColumnTransformer([
    ('text', TfidfVectorizer(stop_words='english'), 'body'),
    ('urls', 'passthrough', ['urls'])
])

X_train = processor.fit_transform(X_train)
X_test = processor.transform(X_test)

scoring = {
    "f1": "f1",
    "f2": make_scorer(fbeta_score, beta=2),
    "f0_5": make_scorer(fbeta_score, beta=0.5)
}

param_grid = {
    "alpha": [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0],
    "fit_prior": [True, False]
}

def print_summary(name, model, execution_time):
    print(f"\n[{name} on k= {k}]")
    print(f"Best Params: {model.best_params_}")
    print(f"Best F1 Score: {model.best_score_:.4f}")

    best_idx = model.best_index_
    f05_val = model.cv_results_['mean_test_f0_5'][best_idx]
    f2_val = model.cv_results_['mean_test_f2'][best_idx]

    print(f"F0.5: {f05_val:.4f}")
    print(f"F2: {f2_val:.4f}")

    print(f"Execution Time: {execution_time:.2f}s")
        
for i in [5]:
    
    k = StratifiedKFold(i, random_state=42, shuffle=True)
    
    # 1. Grid Search
    print("--- Đang chạy Grid Search ---")
    grid_search = GridSearchCV(
        estimator=MultinomialNB(),
        param_grid=param_grid,
        cv=k,
        scoring=scoring,
        refit="f1",
        n_jobs=-1,
        verbose=2
    )
    start = time.time()
    grid_search.fit(X_train, y_train)
    grid_time = time.time() - start

    # In ra kết quả so sánh
    def print_summary(name, model, execution_time):
        print(f"\n{name}")
        print(f"Best Params: {model.best_params_}")
        print(f"Best F1 Score: {model.best_score_:.4f}")

        # GridSearchCV và RandomizedSearchCV
        # có multi-scoring
        if 'mean_test_f0_5' in model.cv_results_:
            best_idx = model.best_index_
            f05_val = model.cv_results_['mean_test_f0_5'][best_idx]
            f2_val = model.cv_results_['mean_test_f2'][best_idx]
            print(f"F0.5: {f05_val:.4f}")
            print(f"F2  : {f2_val:.4f}")

        print(f"Time: {execution_time:.2f}s")

    print_summary("Grid Search", grid_search, grid_time)
    # print_summary("Random Search", random_search, random_time)
    # print_summary("Halving Random Search", halving_search, halving_time)

joblib.dump(grid_search.best_estimator_, "best_model.pkl")
joblib.dump(processor, "processor.pkl")