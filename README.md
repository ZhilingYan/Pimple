# Pimple

BS6204 Deep Learning for Biomedical Science


Train set
The train set is from ISIC Challenge (isic-archive.com).

Test set
We build our own test set using SegmentIt, an interactive image segmentation tool. To get the Ground Truth of image, we manually segmented the photos.



After obtaining the segmented outline, we processed the photo and converted it to black and white. As shown in the figure below, the white part is the result of the segmentation.



Result
We implemented various metrics to evaluate the result of the model, which can be found in the .ipynb file. In addition, we also visualize the segmentation result and compare it to the ground truth.
