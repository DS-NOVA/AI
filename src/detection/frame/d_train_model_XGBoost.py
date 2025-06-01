from c_train_dataset_frames import build_frame_dataset
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import numpy as np
import pandas as pd
from e_extract_true_clips import extract_true_clips
from e_get_wrong_frames import get_wrong_frames

if __name__ == "__main__":
    # 데이터 불러오기
    X_train, X_test, Y_train, Y_test, test_meta, label_names = build_frame_dataset(return_meta=True)

    # 모델 정의
    xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0,
        tree_method='hist',
        learning_rate=0.1,
        max_depth=6,
        n_estimators=100,
        random_state=42
    )

    model = MultiOutputClassifier(xgb)
    model.fit(X_train, Y_train)

    # 예측 (이진화 필요시)
    Y_pred = model.predict(X_test)
    # Y_pred_bin = (Y_pred > 0.5).astype(int)  # XGB는 이미 0/1

    # 평가 출력
    print("\n[Classification Report]")
    print(classification_report(Y_test, Y_pred))

    print("\n[Multilabel Confusion Matrix]")
    print(multilabel_confusion_matrix(Y_test, Y_pred))

    # === 여기에 삽입 ===
    # test_meta, label_names 준비 필요!
    # 예시: test_meta = pd.DataFrame({'video': video_names, 'frame_id': frame_ids})
    pred_df = test_meta.copy()
    for i, label in enumerate(label_names):
        pred_df[label] = Y_pred[:, i]

    # TRUE 연속 구간 추출
    seg_df = extract_true_clips(pred_df, label_names)
    print(seg_df)
    seg_df.to_csv("data/results/predicted_true_clips.csv", index=False)

    wrong_df = get_wrong_frames(test_meta, Y_test, Y_pred, label_names)
    wrong_df.to_csv("data/results/wrong_predicted_frames.csv", index=False)
    print(wrong_df.head())