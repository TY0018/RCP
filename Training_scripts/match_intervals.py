import pandas as pd
import numpy as np

def parse_timestamp(ts_str):
    ts_str = str(ts_str).strip()
    if not ts_str or ts_str == 'nan':
         return None
    try:
        parts = ts_str.split(':')
        if len(parts) == 2:
             return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
             return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except:
        return None
    return None


def main():
    input_csv = "/home/users/ntu/ytong005/RCP/Values - Sheet1.csv"
    output_csv = "/home/users/ntu/ytong005/RCP/MatchedDetections.csv"
    
    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Extract Predictions (cols 0, 1, 2)
    preds = []
    for idx, row in df.iterrows():
        start_str = str(row.iloc[0]).strip()
        end_str = str(row.iloc[1]).strip()
        label = str(row.iloc[2]).strip()
        
        start_sec = parse_timestamp(start_str)
        end_sec = parse_timestamp(end_str)
        
        if start_sec is not None and end_sec is not None:
             preds.append({
                  'idx': idx,
                  'start': start_sec,
                  'end': end_sec,
                  'start_str': start_str,
                  'end_str': end_str,
                  'label': label if label != 'nan' else ""
             })
             
    # Extract Ground Truths (cols 7, 8, 9)
    gts = []
    for idx, row in df.iterrows():
        # Usually pandas fills unnamed columns or keeps them indexed properly
        # We assume cols 7, 8, 9 are the ground truth columns based on your description
        if len(row) > 9:
            start_str = str(row.iloc[7]).strip()
            end_str = str(row.iloc[8]).strip()
            label = str(row.iloc[9]).strip()
            
            start_sec = parse_timestamp(start_str)
            end_sec = parse_timestamp(end_str)
            
            if start_sec is not None and end_sec is not None:
                 gts.append({
                      'idx': idx,
                      'start': start_sec,
                      'end': end_sec,
                      'start_str': start_str,
                      'end_str': end_str,
                      'label': label if label != 'nan' else "",
                      'matched': False
                 })
                 
    print(f"Parsed {len(preds)} predictions and {len(gts)} ground truths.")
    
    matched_rows = []
    
    # Match Predictions to GTs
    for p in preds:
        best_overlap = 0
        best_gt = None
        
        for gt in gts:
             # Calculate overlap duration
             overlap_start = max(p['start'], gt['start'])
             overlap_end = min(p['end'], gt['end'])
             overlap = overlap_end - overlap_start
             
             # Any overlap counts as a match
             if overlap > 0:
                 if overlap > best_overlap:
                     best_overlap = overlap
                     best_gt = gt
                     
        if best_gt is not None:
             # Matched — use GT label as the pred_label
             best_gt['matched'] = True
             matched_rows.append({
                  'pred_start': p['start_str'],
                  'pred_end': p['end_str'],
                  'pred_label': best_gt['label'],
                  'gt_start': best_gt['start_str'],
                  'gt_end': best_gt['end_str'],
                  'gt_label': best_gt['label'],
                  'result': ''
             })
        else:
             # False Positive
             matched_rows.append({
                  'pred_start': p['start_str'],
                  'pred_end': p['end_str'],
                  'pred_label': "",
                  'gt_start': "",
                  'gt_end': "",
                  'gt_label': "",
                  'result': 'FP'
             })
             
    # Add False Negatives (unmatched GTs)
    for gt in gts:
         if not gt['matched']:
              matched_rows.append({
                   'pred_start': "",
                   'pred_end': "",
                   'pred_label': "",
                   'gt_start': gt['start_str'],
                   'gt_end': gt['end_str'],
                   'gt_label': gt['label'],
                   'result': 'FN'
              })
             
    out_df = pd.DataFrame(matched_rows)
    out_df.to_csv(output_csv, index=False)
    
    print(f"\nSaved {len(out_df)} matched ranges to {output_csv}")
    print("="*40)
    print("MATCH SUMMARY:")
    print(f"  Matched (True Positives): {len([r for r in matched_rows if r['result'] == ''])}")
    print(f"  False Positives (Predicted but no GT): {len([r for r in matched_rows if r['result'] == 'FP'])}")
    print(f"  False Negatives (GT but no Predict): {len([r for r in matched_rows if r['result'] == 'FN'])}")
    print("="*40)
        
if __name__ == '__main__':
    main()