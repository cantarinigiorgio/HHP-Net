import tensorflow as tf
import cv2
import numpy as np
from utils.img_utils import draw_key_points_pose


def initialize_interpreter(model_path):
    """

    Args:
        :model_path (str): The file location of the spreadsheet

    Returns:
        :interpreter:
        :input_shape_model:
        :input_details:
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    input_shape_model = input_details[0]['shape'][1], input_details[0]['shape'][2]
    return interpreter, input_shape_model, input_details


def inference_pose(img_person_resized, interpreter, input_details):
    """

    Args:
        :img_person_resized ():
        :interpreter ():
        :input_details ():

    Returns:
        :heatmaps
        :offsets:
    """
    aux_img = img_person_resized.copy()  # in 0 1
    interpreter.set_tensor(input_details[0]['index'], aux_img)
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    heatmaps = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
    offsets = np.squeeze(interpreter.get_tensor(output_details[1]['index']))
    return heatmaps, offsets


def parse_output_pose(heatmaps, offsets, threshold):
    """
    Parse the output pose (auxiliary function for tflite models)
    Args:
        :

    Returns:
        :
    """
    #
    # heatmaps: 9x9x17 probability of appearance of each keypoint in the particular part of the image (9,9) -> used to locate position of the joints
    # offsets: 9x9x34 used for calculation of the keypoint's position (first 17 x coords, the second 17 y coords)
    #
    joint_num = heatmaps.shape[-1]
    pose_kps = np.zeros((joint_num, 3), np.uint32)

    for i in range(heatmaps.shape[-1]):
        joint_heatmap = heatmaps[..., i]
        max_val_pos = np.squeeze(np.argwhere(joint_heatmap == np.max(joint_heatmap)))
        remap_pos = np.array(max_val_pos / 8 * 257, dtype=np.int32)
        pose_kps[i, 0] = int(remap_pos[0] + offsets[max_val_pos[0], max_val_pos[1], i])
        pose_kps[i, 1] = int(remap_pos[1] + offsets[max_val_pos[0], max_val_pos[1], i + joint_num])
        max_prob = np.max(joint_heatmap)

        if pose_kps[i, 0] > 257:
            pose_kps[i, 0] = 257

        if pose_kps[i, 1] > 257:
            pose_kps[i, 1] = 257

        if max_prob > threshold:

            if pose_kps[i, 0] < 257 and pose_kps[i, 1] < 257:
                pose_kps[i, 2] = 1

    return pose_kps


def change_coordinates_aspect_ratio(aux_key_points_array, img_person, padding):
    """

    Args:
        :

    Returns:
        :
    """

    aux_key_points_array_ratio = []
    ratio_h, ratio_w = img_person.shape[1] / (padding[2][0]), (img_person.shape[0]) / (padding[2][1])

    for elem in aux_key_points_array:

        x = int((elem[1]) * ratio_w)
        y = int(elem[0] * ratio_h)
        c = float(elem[2])
        aux_key_points_array_ratio.append([x, y, c])

    return aux_key_points_array_ratio


def resize_preserving_ar(image, new_shape):
    """
    Resize and pad the input image in order to make it usable by an object detection model (e.g. mobilenet 640x640)

    Args:
        :image (numpy.ndarray): The image that will be resized and padded
        :new_shape (tuple): The shape of the image output (height, width)

    Returns:
        :res_image (numpy.ndarray): The image modified to have the new shape
    """
    (old_width, old_height, _) = image.shape
    (new_width, new_height) = new_shape
    # print("SHAPES", old_width, old_height, new_width, new_height)

    if old_width != old_height:  # rectangle
        ratio_w, ratio_h = new_width / old_width, new_height / old_height
        # print("RATIOS", ratio_w, ratio_h)
        # print("EE", int(old_height * ratio_w), int(old_width * ratio_h))

        if ratio_h < ratio_w:
            # print("!")
            # dim = (int(old_width * ratio_h), new_height)
            dim = (new_width, int(old_width * ratio_h))
            img = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
            bottom_padding = int(new_width - int(old_width * ratio_h)) if int(new_width - int(old_width * ratio_h)) >= 0 else 0
            img = cv2.copyMakeBorder(img, 0, bottom_padding, 0, 0, cv2.BORDER_CONSTANT)
            pad = (0, bottom_padding, dim)

        else:
            # print("£")
            dim = (int(old_height * ratio_w), new_height)
            img = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
            right_padding = int(new_height - int(old_height * ratio_w)) if int(new_height - int(old_height * ratio_w)) >= 0 else 0
            img = cv2.copyMakeBorder(img, 0, 0, 0, right_padding, cv2.BORDER_CONSTANT)
            pad = (right_padding, 0, dim)

    else:  # square
        img = cv2.resize(image, new_shape)
        pad = (0, 0, (new_width, new_height))

    return img, pad


def pose_from_det(image, boxes, classes, scores, interpreter, input_shape_interpreter=None, input_details=None, aux_img=None, new_old_shape=None, visualize=False, score_threshold=0.2):
    """

    Args:
        :image ():
        :detections ():
        :interpreter ():
        :input_shape_interpreter ():
        :input_details ():
        :visualize (bool)
            (default is False)

    Returns:
        :kpt_list: a list of
    """
    im_width, im_height = image.shape[0], image.shape[1]
    # print("pòose from det", im_width, im_height)

    if not interpreter:
        return

    kpt_list = []
    i = 0
    while i < len(classes):
        [y_min, x_min, y_max, x_max] = boxes[i]
        (left, right, top, bottom) = (int(x_min * im_width), int(x_max * im_width), int(y_min * im_height), int(y_max * im_height))
        # cv2.imshow("", ima)
        #
        # exit()

        if classes[i] == 0.0 and scores[i] > score_threshold:  # person
            img_person = image[top:bottom, left:right]

            # img_person_resized, padding = resize_and_padding_preserving_ar(img_person, input_shape_interpreter)
            img_person_resized, padding = resize_preserving_ar(img_person, input_shape_interpreter)
            # cv2.imshow("person_resized", img_person_resized[:, :, :])
            # cv2.waitKey(0)
            img_person_resized = img_person_resized.astype(np.float32) / 255.
            img_person_resized = np.expand_dims(img_person_resized, 0)

            # print("3", img_person_resized.shape)

            # start_time = time.time()
            heatmaps, offsets = inference_pose(img_person_resized, interpreter, input_details)
            # end_time = time.time()
            # print("Pose inference time: ", end_time-start_time)

            threshold = 0
            aux_kept_array = parse_output_pose(heatmaps, offsets, threshold)
            # print("A", aux_kept_array)

            aux_kept_array_ratio = change_coordinates_aspect_ratio(aux_kept_array, img_person, padding)
            # print("B", aux_kept_array_ratio)

            # print("PADDING", padding, new_old_shape)
            # exit()

            # print(aux_kept_array_ratio)

            # img_kpts = draw_key_points_pose(img_person, aux_kept_array_ratio, 2, 2)
            # cv2.imshow("kpts1", img_kpts)
            # cv2.waitKey(0)
            # exit()

            for elem in aux_kept_array_ratio:  # coordinates back to original image size
                elem[0] += left
                elem[1] += top

                elem[0] /= (new_old_shape[2][1])
                elem[1] /= (new_old_shape[2][0])

            # print("CCC", aux_kept_array_ratio)
            #
            # img_kpts = draw_key_points_pose(fuck_img, aux_kept_array_ratio, 2, 2)

            kpt_list.append(aux_kept_array_ratio)

            # cv2.imshow("aaa", img_kpts)
            # cv2.imshow("aaa", cv2.resize(img_kpts, (966, 703)))
            # cv2.waitKey(0)
            # exit()
        i += 1

    # print(kpt_list)
    return kpt_list


def inference_interpreter(interpreter, img, input_details):
    output_details = interpreter.get_output_details()

    input_data = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    det_boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    det_classes = interpreter.get_tensor(output_details[1]['index'])[0]
    det_scores = interpreter.get_tensor(output_details[2]['index'])[0]
    num_det = interpreter.get_tensor(output_details[3]['index'])[0]
    return det_boxes, det_classes, det_scores, num_det
