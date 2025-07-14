Title ==========================================
ICRA 2022 
GPR Competition (gprcompetition.com)
UAV Dataset Readme


Author ==========================================
Ivan Cisneros (icisnero@andrew.cmu.edu)


Purpose ==========================================
This document serves as a quick explainer on the dataset format and files, as well as the evaluation metrics
we are using to rank contestants of the GPR Competition in 2022.
This dataset represents a sample of a much larger UAV Visual Place Recognition dataset. Please 
refer to it for more detailed information.  We plan on releasing it on arXiv.org by the end of May 2022.
Until then, please contact the author (listed above) for more information about the paper's release or if you have
any questions. We do ask that if you use this dataset or the full dataset in any published work that you
provide proper attribution to our work.

The goal of this dataset is to promote visual place recognition and localization algorithms in the large-scale
task context. In this case, we provide a set of query and reference imagery of different domains. The 
query and reference imagery cover the same portion of the trajectory of interest, but at slightly different 
geographic coordinates. With a high accuracy algorithm, we should be able to match each query image to the 
geographically closest reference image (in UTM coordinates). The reference images serve as a grid of 
equidistant points, and so matching the query images to the appropriate reference images should provide
useful information of the position of the ego-vehicle in a GPS-denied or GPS-limited context.


Data Information and Data Splits ==========================================
This dataset contains nadir-facing RGB camera imagery captured via commercial helicopter (query), as well
as by high-altitude plane (reference, captured by USGS). We include some query telemetry information that
may be useful for preprocessing or incorporated into your algorithm as contextual information.
We provide three data splits: Train, Val, and Test. These are non-overlapping and include (24701), (3979), 
and (4209) images, respectively. These are all part of a 150km helicopter flight over a variety of
different terrains, including Urban, Suburban, Rural, Dense Forest, Rivers, and Lakes.
For ease of use with deep learning pipelines, we include the images in png format. All images are RGB
and in 500x500 pixel resolution.

The folders are organized as follows:
    Train
        - query_images
            - 000000.png
            - 000001.png
            - ...
        - reference_images
            - offset_0_None
                - 000000.png
                - ...
            - offset_20_North
                - 000000.png
                - ...
            - offset_20_South
                - 000000.png
                - ...
            - offset_40_North
                - 000000.png
                - ...
            - offset_40_South
                - 000000.png
                - ...
        - gt_matches.csv
        - query.csv
        - reference.csv


    Val
        - query_images
            - 000000.png
            - 000001.png
            - ...
        - reference_images
            - offset_0_None
                - 000000.png
                - ...
            - offset_20_North
                - 000000.png
                - ...
            - offset_20_South
                - 000000.png
                - ...
            - offset_40_North
                - 000000.png
                - ...
            - offset_40_South
                - 000000.png
                - ...
        - gt_matches.csv
        - query.csv
        - reference.csv


    Test
        - 000000.png
        - 000001.png
        - ...



The Train and Val sets contain images from the trajectory in sequential order.  As the image index increases, the ego-vehicle 
progresses through the trajectory. The query and reference imagery are organized into separate folders.  

The query images are taken at approximately 20fps, and are spaced approximately 2.5 meters apart.
The reference images are spaced 10 meters apart, and these are taken along the trajectory of interest.
The query and reference images are provided in a ratio of 3.6:1. That means that multiple query images map to the same 
reference image. 
The main reference images are included in the "offset_0_None" folder; this folder should contain all of the closest reference matches
to the query images. But we also provide "offset_20_North", "offset_20_South", "offset_40_North", and "offset_40_South", which
contain reference images that are offset from the trajectory. They run parallel to the trajectory of interest
at +/- 20 meters, or +/- 40 meters offset in the Northing direction. These are optional to use, but may serve as useful data agumentation;
since each of the offset reference images has a large overlap with the offset_0_None reference images, these may be useful for
training more accurate and informative descriptor vectors.

The Test set contains query and reference imagery that are mixed together and in random order. The Test set does not contain any offset 
reference images; the reference images contained here should be the closest possible to the query images. Contestants will need to provide 
the feature vectors from their system in the same order as the Test set so that they can be evaluated properly. If we have 1000 Test images, 
and your algorithm produces a (512,) dimensional descriptor vector for a single image, then you should providea (1000, 512) dimensional tensor 
for submission.  



In addition to the imagery in the Train and Val sets, we also provide a few csv files.  Their contents are as follows:
    - gt_matches.csv: This file provides the ground truth best match between the query images and the reference images.
        As states above, multiple query images may map to a single reference image. We determine "best match" by using the L2 
        distance of the UTM coordinates associated with the respective images.
        - query_ind: The index of the query image.
        - query_name: The name of the query image.
        - ref_ind: The index of the reference image that best matches the query image.
        - ref_name: The name of the reference image that best matches the query image.
        - distance: The distance (meters) between the query and best matching reference image.
    
    - query.csv: Telemetry information about each query image frame.
        - easting: The ground truth Easting coordinate (meters) where the image was taken.
        - northing: The ground truth Northing coordinate (meters) where the image was taken.
        - altitude: The ground truth Altitude (meters) above the WGS84 ellipsoid surface.
        - orient_x, orient_y, orient_z, orient_w: The orientation (scalar last quaternion) of the camera with respect the the ECEF reference frame.
        - name: The name of the query image.

    - reference.csv: Information about each reference image frame. Contains information about all offset reference images as well.
        - easting: The ground truth Easting coordinate (meters) where the image was taken.
        - northing: The ground truth Northing coordinate (meters) where the image was taken.
        - name: The directory and name of the reference image.



Evaluation Criteria ==========================================
We will be evaluating contestants using Top5 recall:
    for each query image vector, is the ground truth reference image vector in the Top 5 closest matching vectors?

    Example:
    Query image 000001.png produces a vector x.
    When we compare vector x to all the vectors in the reference image set, we find that the Top 5 closest vectors (in Euclidean distance)
    are of index [0, 123, 456, 789, 1112].
    From gt_matches.csv we see that the ground truth best match to query 000001.png is reference index 0.
    Since index 0 is in our Top 5 closest vectors list, we count this as correct.

    We use the following function for this evaluation, where:
        predictions: a (N,M) matrix where N is the number of query images, and M is the M closest matching vectors in vector space.
        gt: a (N,) vector with the ground truth best matching reference index for each query.

    """
    num_query = len(predictions)
    top_n_vals = [1, 5]  # evaluate Top1 and Top5 recall
    gt_df = pd.read_csv(os.path.join(input_path, gt_matches))  # gt csv contains only the index of the best match
    gt = gt_df["ref_ind"].to_numpy()

    correct_at_n = np.zeros(len(top_n_vals))
    for q_ind, pred in enumerate(predictions):
        for i, n in enumerate(top_n_vals):
            if np.any(np.in1d(pred[:n], gt[q_ind])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / num_query
    """
