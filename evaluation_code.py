import numpy as np
from skimage.morphology import skeletonize, erosion
from sklearn.metrics import f1_score, accuracy_score

def evaluation_code(prediction, groundtruth):
    '''
    Function to evaluate the performance of AV predictions with a given ground truth
    - prediction: should be an image array of [dim1, dim2, img_channels = 3] with arteries in red and veins in blue
    - groundtruth: same as above
    '''
    
    encoded_pred = np.zeros(prediction.shape[:2], dtype=int)
    encoded_gt = np.zeros(groundtruth.shape[:2], dtype=int)
    
    # convert white pixels to green pixels (which are ignored)
    white_ind = np.where(np.logical_and(groundtruth[:,:,0] == 255, groundtruth[:,:,1] == 255, groundtruth[:,:,2] == 255))
    if white_ind[0].size != 0:
        groundtruth[white_ind] = [0,255,0]
        
    # translate the images to arrays suited for sklearn metrics
    arteriole = np.where(np.logical_and(groundtruth[:,:,0] == 255, groundtruth[:,:,1] == 0)); encoded_gt[arteriole] = 1
    venule = np.where(np.logical_and(groundtruth[:,:,2] == 255, groundtruth[:,:,1] == 0)); encoded_gt[venule] = 2
    arteriole = np.where(prediction[:,:,0] == 255); encoded_pred[arteriole] = 1
    venule = np.where(prediction[:,:,2] == 255); encoded_pred[venule] = 2
   
    # retrieve the indices for the centerline pixels present in the prediction
    center = np.where(np.logical_and(
        np.logical_or((skeletonize(groundtruth[:,:,0] > 0)),(skeletonize(groundtruth[:,:,2] > 0))),
        encoded_pred[:,:] > 0))
    
    encoded_pred_center = encoded_pred[center]
    encoded_gt_center = encoded_gt[center]
    
    # retrieve the indices for the centerline pixels present in the groundtruth
    center_comp = np.where(
        np.logical_or(skeletonize(groundtruth[:,:,0] > 0),skeletonize(groundtruth[:,:,2] > 0)))
    
    encoded_pred_center_comp = encoded_pred[center_comp]
    encoded_gt_center_comp = encoded_gt[center_comp]
    
    # retrieve the indices for discovered centerline pixels - limited to vessels wider than two pixels (for DRIVE)
    center_eroded = np.where(np.logical_and(
        np.logical_or(skeletonize(erosion(groundtruth[:,:,0] > 0)),skeletonize(erosion(groundtruth[:,:,2] > 0))),
        encoded_pred[:,:] > 0))
                             
    encoded_pred_center_eroded = encoded_pred[center_eroded]
    encoded_gt_center_eroded = encoded_gt[center_eroded]
    
    # metrics over full image
    cur1_acc = accuracy_score(encoded_gt.flatten(),encoded_pred.flatten())
    cur1_F1 = f1_score(encoded_gt.flatten(),encoded_pred.flatten(),average='weighted')
    print('Full image')
    print('Accuracy: {}\nF1: {}\n'.format(cur1_acc, cur1_F1))
    metrics1 = [cur1_acc, cur1_F1]
    
    # metrics over discovered centerline pixels
    cur2_acc = accuracy_score(encoded_gt_center.flatten(),encoded_pred_center.flatten())
    cur2_F1 = f1_score(encoded_gt_center.flatten(),encoded_pred_center.flatten(),average='weighted')
    print('Discovered centerline pixels')
    print('Accuracy: {}\nF1: {}\n'.format(cur2_acc, cur2_F1))
    metrics2 = [cur2_acc, cur2_F1]
    
    # metrics over discovered centerline pixels - limited to vessels wider than two pixels
    cur3_acc = accuracy_score(encoded_gt_center_eroded.flatten(),encoded_pred_center_eroded.flatten())
    cur3_F1 = f1_score(encoded_gt_center_eroded.flatten(),encoded_pred_center_eroded.flatten(),average='weighted')
    print('Discovered centerline pixels of vessels wider than two pixels')
    print('Accuracy: {}\nF1: {}\n'.format(cur3_acc, cur3_F1))
    metrics3 = [cur3_acc, cur3_F1]
    
    # metrics over all centerline pixels in ground truth
    cur4_acc = accuracy_score(encoded_gt_center_comp.flatten(),encoded_pred_center_comp.flatten())
    cur4_F1 = f1_score(encoded_gt_center_comp.flatten(),encoded_pred_center_comp.flatten(),average='weighted')
    print('Centerline pixels')
    print('Accuracy: {}\nF1: {}\n'.format(cur4_acc, cur4_F1))
    metrics4 = [cur4_acc, cur4_F1]
    
    # finally, compute vessel detection rate
    vessel_ind = np.where(encoded_gt>0)
    vessel_gt = encoded_gt[vessel_ind]
    vessel_pred = encoded_pred[vessel_ind]
    
    detection_rate = accuracy_score(vessel_gt.flatten(),vessel_pred.flatten())
    print('Amount of vessels detected: ' + str(detection_rate))
    
    return [metrics1,metrics2,metrics3,metrics4,detection_rate]
    
