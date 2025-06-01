import pandas as pd

def get_wrong_frames(test_meta, Y_test, Y_pred, label_names):
    """
    test_meta: pd.DataFrame(['video', 'frame_id'])
    Y_test: (N, C) numpy array (정답)
    Y_pred: (N, C) numpy array (예측값)
    label_names: 라벨명 리스트
    """
    wrong_records = []

    for i in range(Y_test.shape[0]):
        for j, label in enumerate(label_names):
            if Y_test[i, j] != Y_pred[i, j]:
                wrong_records.append({
                    'video': test_meta.iloc[i]['video'],
                    'frame_id': test_meta.iloc[i]['frame_id'],
                    'label': label,
                    'true': Y_test[i, j],
                    'pred': Y_pred[i, j],
                })
    return pd.DataFrame(wrong_records)
