import pandas as pd

def extract_true_clips(pred_df, label_names):
    """
    pred_df: columns=['video', 'frame_id', label1, label2, ...]
    label_names: list of label column names
    """
    result = []

    for video_name, group in pred_df.groupby('video'):
        # frame_id 기준으로 정렬!
        group_sorted = group.sort_values('frame_id').reset_index(drop=True)
        for label in label_names:
            is_true = group_sorted[label].values
            frame_ids = group_sorted['frame_id'].values

            in_segment = False
            seg_start = None

            for idx, val in enumerate(is_true):
                if val and not in_segment:
                    in_segment = True
                    seg_start = frame_ids[idx]
                elif not val and in_segment:
                    in_segment = False
                    seg_end = frame_ids[idx - 1]
                    result.append({
                        'video': video_name,
                        'label': label,
                        'start_frame_id': seg_start,
                        'end_frame_id': seg_end
                    })
            if in_segment:
                result.append({
                    'video': video_name,
                    'label': label,
                    'start_frame_id': seg_start,
                    'end_frame_id': frame_ids[-1]
                })
    return pd.DataFrame(result)

