
import pandas as pd
import pickle
import numpy as np

HT_HR_FC_HD = {0: 'take', 
               1: 'release', 
               2: 'first_contact', 
               3: 'decontact'}

HT_HR = {0: 'take',
         1: 'release'}

FC_HD = {0: 'first_contact',
         1: 'decontact'}
    
def interpolated_prec_rec(prec: np.ndarray, rec: np.ndarray) -> float:
    """Interpolated AP - VOCdevkit from VOC 2011.

    Args:
        prec (np.ndarray): Precision array.
        rec (np.ndarray): Recall array.

    Returns:
        float: Interpolated AP.
    """
    
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap

def temporal_offset(target_AS: float, candidate_AS: np.ndarray) -> np.ndarray:    
    """Compute the temporal offset between a target AS and all the test AS.

    Args:
        target_AS (float): Ground truth action start considered as target.
        candidate_AS (np.ndarray): Predicted action starts considered as candidates.

    Returns:
        np.ndarray: Temporal offset between a target AS and all the test AS.
    """   
    
    result = np.absolute(candidate_AS - target_AS)
    return result

def compute_average_precision_detection(ground_truth: pd.DataFrame, prediction: pd.DataFrame, tOffset_thresholds: np.ndarray, fps: float) -> np.ndarray:
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with smallest offset is matches as
    true positive.

    Args:
        ground_truth (pd.DataFrame): Ground truth of action starts.
        prediction (pd.DataFrame): Predictions of action starts.
        tOffset_thresholds (np.ndarray): Temporal offset thresholds in seconds.
        fps (float): Frame rate of the video.

    Returns:
        np.ndarray: Average precision score for each tOffset_threshold.
    """    

    # since we will use indexes, we reset them in order to avoid problems
    # (e.g, if the indexes are not in order)
    
    ground_truth = ground_truth.reset_index(drop=True)
    prediction = prediction.reset_index(drop=True)
    
    # Convert thresholds seconds to frames.
    tOffset_thresholds = tOffset_thresholds * fps
    
    ap = np.zeros(len(tOffset_thresholds))
    if prediction.empty:
        return ap
    
    num_pos = float(len(ground_truth))

    lock_gt = np.ones((len(tOffset_thresholds),len(ground_truth))) * -1

    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction_sorted = prediction.loc[sort_idx].reset_index(drop=True)
    
    size = (len(tOffset_thresholds), len(prediction))
    tp = np.zeros(size)
    fp = np.zeros(size)

    ground_truth_gbvn = ground_truth.groupby('video-id')
    
    for idx, this_pred in prediction_sorted.iterrows():
        try:
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
            # In case there is no ground truth for this video-id
            # everything is considered as false positive
        except Exception as e:
            # print(f"no ground truth for video-id {this_pred['video-id']}")
            fp[:, idx] = 1
            continue
        
        this_gt = ground_truth_videoid.reset_index()

        toff_arr = temporal_offset(this_pred['t-start'],
                               this_gt['t-start'].values)
        
        tOffset_sorted_idx = toff_arr.argsort()
  
        for tidx, toff_thr in enumerate(tOffset_thresholds):
            for jdx in tOffset_sorted_idx:
                if toff_arr[jdx] > toff_thr:
                    fp[tidx, idx] = 1
                    break
                    
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                
                tp[tidx, idx] = 1
                
                # lock this ground truth
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break
            
            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(float)

    recall_cumsum = tp_cumsum / num_pos
    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tOffset_thresholds)):      
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx,:], recall_cumsum[tidx,:])
        
    return ap


def get_gt_action_start(csv_filename: str) -> pd.DataFrame:
    """ Get the ground truth action starts from the csv file as a pandas dataframe.

    Args:
        csv_filename (str): Path to the csv file.

    Returns:
        pd.DataFrame: Ground truth action starts.
        Columns: 'video-id', 't-start', 'action_name'.
    """    
    
    interactions = pd.read_csv(csv_filename)
    interactions_groupby_video = interactions.groupby('video_id')
    gt_action_starts = pd.DataFrame()

    for video_id, video_interactions in interactions_groupby_video:
        
        video_id = str(video_id)
        groupby_category = video_interactions.groupby('interaction_category')
        for action_class, interactions in groupby_category:
                
            data_interactions = interactions.reset_index(drop=True)
            

            data = {'video-id': [video_id] * len(data_interactions),
                    't-start': data_interactions['timestamp'].values,
                    'action_name': [action_class] * len(data_interactions)}

            gt_action_starts = pd.concat([gt_action_starts, pd.DataFrame(data)], ignore_index=True)
    
    return gt_action_starts

def class_p_ap(gt: pd.DataFrame, prediction: pd.DataFrame, tOffset_thresholds: np.ndarray, fps:float, name_class: bool=None, show: bool = False) -> np.ndarray:
    """ Get the p-Average Precision for a single class.

    Args:
        gt (pd.DataFrame): Ground truth action starts.
        prediction (pd.DataFrame): Predictions of action starts.
        tOffset_thresholds (np.ndarray): Temporary offset thresholds in seconds.
        fps (float): Frame rate of the video.
        name_class (bool, optional): Name of the class to show if 'show' is True. Defaults to None.
        show (bool, optional): Show the results. Defaults to False.
    Returns:
        np.ndarray: p-Average Precision for a single class. mAP for each tOffset_threshold.
    """
        
    ap = compute_average_precision_detection(gt,
                                            prediction,
                                            tOffset_thresholds,
                                            fps)
    
    if name_class is None:
        name_class = "class"

    if show:
        print(f"{name_class} p-Average Precision: {ap}")
        print(f"mean {name_class} p-Average Precision: {np.mean(ap)}")
       
    return ap

def p_ap_classes(gt_action_starts: pd.DataFrame, pred_df: pd.DataFrame, tOffset_thresholds: np.ndarray, fps: float, show: bool=False) -> list:
    """ Get the p-Average Precision for each class.

    Args:
        gt_action_starts (pd.DataFrame): Ground truth action starts.
        pred_df (pd.DataFrame): Predictions of action starts.
        tOffset_thresholds (np.ndarray): Temporary offset thresholds in seconds.
        fps (float): Frame rate of the video.
        show (bool, optional): . Defaults to False.

    Returns:
        list: List of p-Average Precision for each class.
    """    
    
    gt_action_starts_gpc = gt_action_starts.groupby('action_name')
    ap_classes = []
    
    for class_name, pred_df_class in pred_df.groupby('action_name'):
        pred_df_class = pred_df_class.reset_index(drop=True)
        
        gt_action_starts_class = gt_action_starts_gpc.get_group(class_name).reset_index(drop=True)
    	
        ap = class_p_ap(gt_action_starts_class, 
                        pred_df_class, 
                        tOffset_thresholds,
                        fps = fps,
                        name_class=class_name,
                        show=show)
        
        ap_classes.append(ap)
    
    return ap_classes

    
def get_metrics(tOffset_thresholds: np.ndarray, gt_action_starts: pd.DataFrame, pred_df: pd.DataFrame, cfg: dict) -> dict:
    """ Get the metrics (results) given the ground truth action starts and the predictions.

    Args:
        tOffset_thresholds (np.ndarray): Temporary offset thresholds in seconds.
        gt_action_starts (pd.DataFrame): Ground truth action starts.
        pred_df (pd.DataFrame): Predictions of action starts.
        cfg (dict): Configuration dict. Keys: 'fps'.

    Returns:
        dict: Dict with the results. Keys: 'p_mAP_tOffset', 'mp_mAP', 'cfg'.
    """    
    
    fps = cfg['fps']
    
    ap_classes = p_ap_classes(gt_action_starts, pred_df, tOffset_thresholds, fps=fps, show=False)
    
    p_mAP = np.mean(ap_classes, axis=0)
    mp_mAP = np.mean(p_mAP)
    
    results = {'p_mAP_tOffset': p_mAP,
            'mp_mAP': mp_mAP,
            'cfg': cfg}

    return results


def get_formatted_df(results: dict, dataset_classes: dict) -> pd.DataFrame:
    """ Format the results dict into a pandas dataframe.

    Args:
        results (dict): Dict with the results. Keys: 'video-id', 't-start', 'label', 'score'.
        dataset_classes (dict): Dict of the dataset classes (e.g. {'0': 'take', '1': 'release'}).

    Returns:
        pd.DataFrame: Formatted dataframe of the results.
    """    
    
    video_ids = results['video-id']
    t_starts = results['t-start']
    labels = results['label']
    scores = results['score']

    labels = [dataset_classes[label] for label in labels]


    return pd.DataFrame({'video-id': video_ids, 't-start': t_starts, 't-pred': t_starts, 'action_name': labels, 'score': scores})


def calculate_mp_map(csv_filename: str, filename_predictions: str, dataset_classes: dict) -> None:
    """ Test a single configuration; If save_test_folder is not None, save the result in a single csv file and in a pickle file.
        Otherwise, print the results.

    Args:
        csv_filename (str): Cannot be None. Path to the csv file where the results should or are already saved.
        filename_predictions (str): Path to the predictions file.
        dataset_classes (dict): Dict of the dataset classes (e.g. {'0': 'take', '1': 'release'}).
    
    Returns:
        None
    """
    
    tOffset_thresholds=np.linspace(1.0, 10.0, 10)
    cfg = {
        'fps': 1
    }
    
    with open(filename_predictions, 'rb') as f:
        enigma_evals = pickle.load(f)

    enigma_evals = get_formatted_df(enigma_evals, dataset_classes)
    gt_action_starts = get_gt_action_start(csv_filename)
    
    results = get_metrics(tOffset_thresholds, gt_action_starts, enigma_evals, cfg)
    print(results)
    
    
if __name__ == "__main__":
    # HT_HR_FC_HD, HT_HR, FC_HD
    dataset_classes = HT_HR
    csv_filename = "test_csv/ht_hr.csv"
    filename_predictions = "../pretrained/enigma/ht_hr/eval_results.pkl"
    filename_predictions = "../outputs/enigma_ht_hr/eval_results.pkl"

    # dataset_classes = HT_HR_FC_HD
    # csv_filename = "test_csv/ht_hr_fc_hd.csv"
    # filename_predictions = "../pretrained/enigma/ht_hr_fc_hd/eval_results.pkl"

    # dataset_classes = FC_HD
    # csv_filename = "test_csv/fc_hd.csv"
    # filename_predictions = "../pretrained/enigma/fc_hd/eval_results.pkl"

    calculate_mp_map(csv_filename, filename_predictions, dataset_classes)