from c_train_dataset_frames import build_frame_dataset
from sklearn.multioutput import MultiOutputClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import numpy as np

if __name__ == "__main__":
    # 데이터 로딩
    X_train, X_test, Y_train, Y_test = build_frame_dataset()

    # 모델 정의
    lgbm = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        verbose=-1
    )

    model = MultiOutputClassifier(lgbm)
    model.fit(X_train, Y_train)

    # 예측
    Y_pred = model.predict(X_test)

    # 평가 출력
    print("\n[Classification Report]")
    print(classification_report(Y_test, Y_pred))

    print("\n[Multilabel Confusion Matrix]")
    print(multilabel_confusion_matrix(Y_test, Y_pred))
