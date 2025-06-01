from train_dataset import build_training_dataset
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import numpy as np

if __name__ == "__main__":
    # 데이터 불러오기
    X_train, X_test, Y_train, Y_test = build_training_dataset()

    # 모델 정의
    xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0,
        tree_method='hist',         # CPU 빠른 학습
        learning_rate=0.1,
        max_depth=6,
        n_estimators=100,
        random_state=42
    )

    model = MultiOutputClassifier(xgb)
    model.fit(X_train, Y_train)

    # 예측
    Y_pred = model.predict(X_test)

    # 평가 출력
    print("\n[Classification Report]")
    print(classification_report(Y_test, Y_pred))

    print("\n[Multilabel Confusion Matrix]")
    print(multilabel_confusion_matrix(Y_test, Y_pred))
