import os
import numpy as np
import cv2
def draw_segmentation_result(segmentation_mask, image, ground_truth):
    #image is a torch tensor (3 x w x h)
    #segmentation_mask is a torch tensor (1 x w x h)
    #ground_truth is a torch tensor (1 x w x h)
    #get the numpy array of the image
    image = image.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = image * 255
    image = image.astype(np.uint8)
    # scale image such that width is 700
#     width = 700
#     height = int(image.shape[0] * width / image.shape[1])
#     image = cv2.resize(image, (width, height))
    print('image.resize.shape:', image.shape)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    #get the numpy array of the segmentation mask
    segmentation_mask = segmentation_mask.detach().cpu().numpy()
    segmentation_mask = np.transpose(segmentation_mask, (1, 2, 0))
    segmentation_mask = segmentation_mask * 255
    segmentation_mask = segmentation_mask.astype(np.uint8)
    #get the numpy array of the ground truth
    ground_truth = ground_truth.detach().cpu().numpy()
    ground_truth = np.transpose(ground_truth, (1, 2, 0))
    ground_truth = ground_truth * 255
    ground_truth = ground_truth.astype(np.uint8)
    #turn the color of segmentation mask into 3 channels
    segmentation_mask = cv2.cvtColor(segmentation_mask, cv2.COLOR_GRAY2RGB)
    #turn the color of ground truth into 3 channels
    ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_GRAY2RGB)
    #make the segmentation mask gree
    segmentation_mask[:, :, 0] = 0
    segmentation_mask[:, :, 2] = 0
    #make the ground truth red
    ground_truth[:, :, 1] = 0
    ground_truth[:, :, 2] = 0
    # scale ground truth to same size as image
    ground_truth = cv2.resize(ground_truth, (image.shape[1], image.shape[0]))
    # scale segmentation mask to same size as image
    segmentation_mask = cv2.resize(segmentation_mask, (image.shape[1], image.shape[0]))
    #add weighted the segmentation mask and the ground truth to the image
    SRandGT=cv2.addWeighted(segmentation_mask, 0.5, ground_truth, 0.5, 0)//2
    #add weighted the image and the segmentation mask and the ground truth
    result=cv2.addWeighted(image, 0.5, SRandGT, 0.5, 0)*2
    return result

def draw_segmentation_results(image_batch, segmentation_mask_batch, ground_truth_batch,show_result=False, save_filename_prefix="_result",save_dir="../results"):
    #image_batch is a torch tensor (batch_size x 3 x w x h)
    #segmentation_mask_batch is a torch tensor (batch_size x 1 x w x h)
    #ground_truth_batch is a torch tensor (batch_size x 1 x w x h)
    for i in range(image_batch.shape[0]):
        image = image_batch[i]
        print('image.shape:', image.shape)
        segmentation_mask = segmentation_mask_batch[i]
        print('segmentation.shape', segmentation_mask.shape)
        ground_truth = ground_truth_batch[i]
        print('ground_truth.shape', ground_truth.shape)
        result = draw_segmentation_result(segmentation_mask, image, ground_truth)
        #if save dir not exist, create it
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        #get the lowest filename+x that doesnt exist
        x=0
        while os.path.exists(save_dir+"/"+save_filename_prefix+str(x)+".png"):
            x+=1
        #save the result
        cv2.imwrite(save_dir+"/"+save_filename_prefix+str(x)+".png", result)
        if show_result:
            cv2.imshow("result", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
