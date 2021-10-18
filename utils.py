import cv2
import time
import numpy as np
from labels import coco_category_index


def resize_preserving_ar(image, new_shape):
    """
    Resize and pad the input image in order to make it usable by an object detection model (e.g. centernet 512x512)

    Args:
        :image (numpy.ndarray): The image that will be resized and padded
        :new_shape (tuple): The shape of the image output (height, width)

    Returns:
        :res_image (numpy.ndarray): The image resized and padded to have the new shape
        :pad (tuple): Tuple that contains the information about the padding operation (right padding and bottom padding) and the new shape of the image computed in order to
                    maintain the aspect ratio (without the padding)
    """
    (old_height, old_width, _) = image.shape
    (new_height, new_width) = new_shape

    if old_height != old_width:  # rectangle
        ratio_h, ratio_w = new_height / old_height, new_width / old_width

        if ratio_h > ratio_w:
            dim = (new_width, int(old_height * ratio_w))
            res_image = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
            bottom_padding = int(new_height - int(old_height * ratio_w)) if int(new_height - int(old_height * ratio_w)) >= 0 else 0
            res_image = cv2.copyMakeBorder(res_image, 0, bottom_padding, 0, 0, cv2.BORDER_CONSTANT)
            pad = (0, bottom_padding, dim)

        else:
            dim = (int(old_width * ratio_h), new_height)
            res_image = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
            right_padding = int(new_width - int(old_width * ratio_h)) if int(new_width - int(old_width * ratio_h)) >= 0 else 0
            res_image = cv2.copyMakeBorder(res_image, 0, 0, 0, right_padding, cv2.BORDER_CONSTANT)
            pad = (right_padding, 0, dim)

    else:  # square
        res_image = cv2.resize(image, new_shape, new_height, new_width)
        pad = (0, 0, (new_height, new_width))

    return res_image, pad


def delete_items_from_array_aux(arr, i):
    """
    Auxiliary function that delete the item at a certain index from a numpy array

    Args:
        :arr (numpy.ndarray): Array of array where each element correspond to the four coordinates of bounding box expressed in percentage
        :i (int): Index of the element to be deleted

    Returns:
        :arr_ret: the array without the element at index i
    """

    aux = arr.tolist()
    aux.pop(i)
    arr_ret = np.array(aux)
    return arr_ret


def filter_detections(detections, min_score_thresh, shape, new_old_shape=None):
    """
    Filter the detections based on a minimum threshold value and modify the bounding box coordinates if the image was resized for the detection

    Args:
        :detections (dict): The dictionary that outputs the model
        :min_score_thresh (float): The minimum score for the detections (detections with a score lower than this value will be discarded)
        :shape (tuple): The shape of the image
        :new_old_shape (tuple): The first element represents the right padding (applied by resize_preserving_ar() function);
                                the second element represents the bottom padding (applied by resize_preserving_ar() function) and
                                the third element is a tuple that is the shape of the image after resizing without the padding (this is useful for
                                the coordinates changes that we have to do)
            (default is None)

    Returns:
        :filtered_detections (dict): dictionary with detection scores, classes, centroids and bounding box coordinates ordered by score in descending order
    """
    allowed_categories = ["person"]

    im_height, im_width, _ = shape
    center_net = False

    classes = detections['detection_classes'][0].numpy().astype(np.int32)
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    key_points_score = None
    key_points = None

    if 'detection_keypoint_scores' in detections:
        key_points_score = detections['detection_keypoint_scores'][0].numpy()
        key_points = detections['detection_keypoints'][0].numpy()
        center_net = True

    sorted_index = np.argsort(scores)[::-1]
    scores = scores[sorted_index]
    boxes = boxes[sorted_index]
    classes = classes[sorted_index]

    i = 0
    while i < 10000:
        if scores[i] < min_score_thresh:  # sorted
            break
        if coco_category_index[classes[i]]["name"] in allowed_categories:
            i += 1
        else:
            scores = np.delete(scores, i)
            boxes = delete_items_from_array_aux(boxes, i)
            classes = np.delete(classes, i)
            if center_net:
                key_points_score = delete_items_from_array_aux(key_points_score, i)
                key_points = delete_items_from_array_aux(key_points, i)

    filtered_detections = dict()
    filtered_detections['detection_classes'] = classes[:i]

    rescaled_boxes = (boxes[:i])

    if new_old_shape:
        rescale_bb(rescaled_boxes, new_old_shape, im_width, im_height)
        if center_net:
            rescaled_key_points = key_points[:i]
            rescale_key_points(rescaled_key_points, new_old_shape, im_width, im_height)

    filtered_detections['detection_boxes'] = rescaled_boxes
    filtered_detections['detection_scores'] = scores[:i]

    if center_net:
        filtered_detections['detection_keypoint_scores'] = key_points_score[:i]
        filtered_detections['detection_keypoints'] = rescaled_key_points

    aux_centroids = []
    for bb in boxes[:i]:  # y_min, x_min, y_max, x_max
        centroid_x = (bb[1] + bb[3]) / 2.
        centroid_y = (bb[0] + bb[2]) / 2.
        aux_centroids.append([centroid_x, centroid_y])

    filtered_detections['detection_boxes_centroid'] = np.array(aux_centroids)

    return filtered_detections


def detect(model, image, min_score_thresh, new_old_shape):
    """
    Detect objects in the image running the model

    Args:
        :model (tensorflow.python.saved_model): The Tensorflow object detection model
        :image (numpy.ndarray): The image that is given as input to the object detection model
        :min_score_threshold (float): The minimum score for the detections (detections with a score lower than this value will be discarded)
        :new_old_shape (tuple): The first element represents the right padding (applied by resize_preserving_ar() function);
                                the second element represents the bottom padding (applied by resize_preserving_ar() function) and
                                the third element is a tuple that is the shape of the image after resizing without the padding (this is useful for
                                the coordinates changes)

    Returns:
        :detections (dict): dictionary with detection scores, classes, centroids and bounding box coordinates ordered by score in descending order
        :inference_time (float): inference time for one image (expressed in seconds)
    """
    image = np.array(image).astype(np.uint8)
    input_tensor = np.expand_dims(image, axis=0)

    start_time = time.time()
    det = model(input_tensor)
    end_time = time.time()

    detections = filter_detections(det, min_score_thresh, image.shape, new_old_shape)
    inference_time = end_time - start_time
    return detections, inference_time


def rescale_bb(boxes, pad, im_width, im_height):
    """
    Modify in place the bounding box coordinates (percentage) to the new image width and height

    Args:
        :boxes (numpy.ndarray): Array of bounding box coordinates expressed in percentage [y_min, x_min, y_max, x_max]
        :pad (tuple): The first element represents the right padding (applied by resize_preserving_ar() function);
                        the second element represents the bottom padding (applied by resize_preserving_ar() function) and
                        the third element is a tuple that is the shape of the image after resizing without the padding (this is useful for
                        the coordinates changes)
        :im_width (int): The new image width
        :im_height (int): The new image height

    Returns:
    """

    right_padding = pad[0]
    bottom_padding = pad[1]

    if bottom_padding != 0:
        for box in boxes:
            y_min, y_max = box[0] * im_height, box[2] * im_height  # to pixels
            box[0], box[2] = y_min / (im_height - pad[1]), y_max / (im_height - pad[1])  # back to percentage

    if right_padding != 0:
        for box in boxes:
            x_min, x_max = box[1] * im_width, box[3] * im_width  # to pixels
            box[1], box[3] = x_min / (im_width - pad[0]), x_max / (im_width - pad[0])  # back to percentage


def rescale_key_points(key_points, pad, im_width, im_height):
    """
    Modify in place the bounding box coordinates (percentage) to the new image width and height

    Args:
        :key_points (numpy.ndarray): Array of bounding box coordinates expressed in percentage [y_min, x_min, y_max, x_max]
        :pad (tuple): The first element represents the right padding (applied by resize_preserving_ar() function);
                        the second element represents the bottom padding (applied by resize_preserving_ar() function) and
                        the third element is a tuple that is the shape of the image after resizing without the padding (this is useful for
                        the coordinates changes)
        :im_width (int): The new image width
        :im_height (int): The new image height

    Returns:
    """

    right_padding = pad[0]
    bottom_padding = pad[1]

    print("Rescale kpts", right_padding, bottom_padding, pad, im_width, im_height)
    print(key_points)
    # exit()

    if bottom_padding != 0:
        for aux in key_points:
            for point in aux:  # x 1 y 0
                y = point[0] * im_height
                point[0] = y / (im_height)
                # x = point[1] * im_width
                # point[1] = int(x / (im_width - pad[0]))

    if right_padding != 0:
        for aux in key_points:
            for point in aux:
                x = point[1] * im_width
                point[1] = int(x / im_width)
    print(key_points)
    return key_points


def percentage_to_pixel(shape, bb_boxes, bb_boxes_scores, key_points=None, key_points_score=None):
    """
    Convert the detections from percentage to pixels coordinates; it works both for the bounding boxes and for the key points if passed

    Args:
        :img_shape (tuple): the shape of the image (heigth, width, channels)
        :bb_boxes (numpy.ndarray): list of list each one representing the bounding box coordinates for each object detected expressed in percentage
            [[y_min_perc, x_min_perc, y_max_perc, x_max_perc], .., [..]]
        :bb_boxes_scores (numpy.ndarray): list of score for each bounding box in bb_boxes in range [0, 1]
        :key_points (numpy.ndarray): list of list of list each one representing the key points coordinates expressed in percentage associated to each detection
            [[[y_perc, x_perc], .., [..]], [[..], .., [..]]]
            (default is None)
        :key_points_score (numpy.ndarray): list of list each one representing the score associated to each key point in range [0, 1]; e.g. [[s1, s2, ...] , .., [..]]
            (default is None)

    Returns:
        :det (numpy.ndarray): list of lists each one representing the bounding box coordinates in pixels and the score associated to each bounding box
            [[x_min, y_min, x_max, y_max, score], .., [..]]
        :kpt (list): list of lists of lists each one representing the key points detected in pixels and the score associated to that point [[[x, y, score], .., []], ..]
    """

    im_height, im_width = shape[0], shape[1]
    print("BBB", shape, key_points)
    det, kpt = [], []

    if key_points is not None:
        key_points = key_points
        key_points_score = key_points_score

    for i, _ in enumerate(bb_boxes):
        y_min, x_min, y_max, x_max = bb_boxes[i]
        x_min_rescaled, x_max_rescaled, y_min_rescaled, y_max_rescaled = x_min * im_width, x_max * im_width, y_min * im_height, y_max * im_height
        det.append([int(x_min_rescaled), int(y_min_rescaled), int(x_max_rescaled), int(y_max_rescaled), bb_boxes_scores[i]])

        if key_points is not None:
            aux_list = []
            for n, key_point in enumerate(key_points[i]):
                aux = [int(key_point[0] * im_width), int(key_point[1] * im_height), key_points_score[i][n]]
                aux_list.append(aux)
            kpt.append(aux_list)

    det = np.array(det)

    return det, kpt


def get_interest_points(kpts, detector=''):
    """
    Get the interest points (the five points of the face). Based on the detector used the number of points may differ; we assume two possible detectors
    Centernet and OpenPose

    Args:
        :kpts (list): list of the key points predicted by the detector
        :detector (str): detector used; possible values are centernet (for Centernet) otherwise we assume the detector used is OpenPose
            (default is '')

    Returns:
        # TO CHECK
        :res_kpts (list): list of interest points (points of the face) formalized as [xy, xy, c]
    """
    if detector == 'centernet':
        face_points = [0, 1, 2, 3, 4]
    else:
        face_points = [0, 16, 15, 18, 17]  # OpenPose

    res_kpts = []

    for index in face_points:
        res_kpts.append(kpts[index][1])
        res_kpts.append(kpts[index][0])
        res_kpts.append(kpts[index][2])

    return res_kpts


def compute_centroid_list(points):
    """
    #TO CHECK
    Compute the centroid from a list of points expressed by three coordinates (x,y,c) where x and y are the horizontal and vertical
    components while c is the confidence.
    It loop through the list, take only points with confidence higher than 0 and the return a list containing the means if at least
    one point with c>0 is found, otherwise it returns the list [None, None]

    Args:
        :points (list)

    Returns:
        :mean_list (list): list containing the mean for x and y
    """
    x, y = [], []
    mean_list = [None, None]
    for i in range(0, len(points), 3):
        if points[i + 2] > 0.0:  # confidence openpose
            x.append(points[i])
            y.append(points[i + 1])

    if x == [] or y == []:
        return mean_list

    mean_x = np.mean(x)
    mean_y = np.mean(y)
    mean_list = [mean_x, mean_y]

    return mean_list


def normalize_wrt_maximum_distance_point(points, file_name=''):
    """
    Normalize each with respect to the maximum distance point (keeping separate x's and y's): we find the maximum x distance and y distance,
    then we normalize each x and y with the maximum (this is done to avoid a possible loss of information if data not well distributed).
    The resulting points are in range [-1, 1]
    #TO CHECK RISCRIVERE BENE

    Args:
        :points (list)

    Returns:
        :points (list): points normalized to bea in range [-1, 1]
    """
    centroid = compute_centroid_list(points)

    max_dist_x, max_dist_y = 0, 0
    for i in range(0, len(points), 3):
        if points[i + 2] > 0.0:  # confidence openpose: take only valid keypoints (if not detected (0, 0, 0)
            distance_x = abs(points[i] - centroid[0])
            distance_y = abs(points[i+1] - centroid[1])
            if distance_x > max_dist_x:
                max_dist_x = distance_x
            if distance_y > max_dist_y:
                max_dist_y = distance_y
        elif points[i + 2] == 0.0:
            points[i] = 0
            points[i+1] = 0

    for i in range(0, len(points), 3):
        if points[i + 2] > 0.0:
            if max_dist_x != 0.0:
                points[i] = (points[i] - centroid[0]) / max_dist_x
            if max_dist_y != 0.0:
                points[i + 1] = (points[i + 1] - centroid[1]) / max_dist_y
            if max_dist_x == 0.0:  # only one point valid with some confidence value so it become (0,0, confidence)
                points[i] = 0.0
            if max_dist_y == 0.0:
                points[i + 1] = 0.0
    # print(points)
    # exit()

    return points