import torch
import scipy.ndimage as ndimage
import numpy as np
import cc3d

def _quick_row_unique(arr):
    # who needs 0s anyway
    arr = arr[np.all(arr != 0, axis=1)]
    # Lexsort rows based on the values in each column
    arr = arr.reshape(-1, 2)
    sorted_indices = np.lexsort(arr.T)
    
    # Create a sorted version of the array
    sorted_arr = arr[sorted_indices]
    
    # Find where rows change
    unique_mask = np.ones(len(arr), dtype=bool)
    unique_mask[1:] = (sorted_arr[1:] != sorted_arr[:-1]).any(axis=1)
    
    # Apply the mask to get the unique rows
    return sorted_arr[unique_mask]

def graphmatch_torch_adapter(x, y, do_bg=False):
    with torch.no_grad():
        if x.ndim != y.ndim:
            y = y.view((y.shape[0], 1, *y.shape[1:]))

        if x.shape == y.shape:
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = y
        else:
            y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
            y_onehot.scatter_(1, y.long(), 1)

        if not do_bg:
            y_onehot = y_onehot[:, 1:]
            x = x[:, 1:]

        x = x.detach().cpu().numpy()
        y_onehot = y_onehot.detach().cpu().numpy()

        # this is not perfect but the bulk of computing time is inside perform_graphmatch anyway
        out = [
            [perform_graphmatch(x[batch, layer].astype(np.int8), y_onehot[batch, layer].astype(np.int8)) for layer in range(x.shape[1])]
            for batch in range(x.shape[0])
        ]

        out = np.array(out).reshape(x.shape)
        if not do_bg:
            out_fa = np.zeros_like(out)
            out = np.concatenate([out_fa, out], axis=1)

        mask_tensor = torch.tensor(out, dtype=torch.bool)

    return mask_tensor    

# warning: CPU scaling!
def perform_graphmatch(pred, label):
    connectivity = 26 if len(label.shape) > 2 else 8

    pred = ndimage.binary_dilation(pred, iterations=1)
    cc_pred = cc3d.connected_components(pred, connectivity=connectivity)
    n_pred = np.amax(cc_pred)

    diff = (label-pred) > 0
    diff = ndimage.binary_dilation(diff, iterations=2)

    cc_diff = cc3d.connected_components(diff, connectivity=connectivity)

    inc_score = np.zeros_like(label)

    diff_pred_stack = np.stack([cc_diff, cc_pred], axis=-1)
    diff_pred_stack = diff_pred_stack.reshape(-1, 2)


    # unique_combinations = np.unique(diff_pred_stack, axis=0)
    unique_combinations = _quick_row_unique(diff_pred_stack)

    # remove all rows that contain 0 (background)
    # unique_combinations = unique_combinations[np.all(unique_combinations != 0, axis=1)]
    # not needed anymore since _quick_row_unique does this

    # step 1: match disconnected predictions
    matched_diff = unique_combinations[:,0]
    diffs, n_matches = np.unique(matched_diff, return_counts=True)
    diffs_critical = diffs[n_matches >= 2]
    inc_score = np.isin(cc_diff, diffs_critical)

    # step 2: match predictions that don't overlap with any ground truth
    matched_preds = unique_combinations[:, 1]
    unmatched_preds = set(range(1, n_pred)) - set(matched_preds)
    unmatched_preds_index = np.isin(cc_pred, list(unmatched_preds))

    inc_score[unmatched_preds_index] = True

    return inc_score.astype(np.float32)