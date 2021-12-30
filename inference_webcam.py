import argparse
import tensorflow as tf
import cv2
import os
import time
from utils.utils import resize_preserving_ar, detect, percentage_to_pixel, get_face_points, normalize_wrt_maximum_distance_point
from utils.img_utils import draw_key_points_pose, draw_axis
import numpy as np


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-dm", "--detection-model", type=str, default=None, help="PATH_DETECTION_MODEL", required=True)
    ap.add_argument("-hm", "--hhp-model", type=str, default=None, help="PATH_HHPNET", required=True)
    config = ap.parse_args()

    tf.keras.backend.clear_session()
    model_detection = tf.saved_model.load(os.path.join(config.detection_model, 'saved_model'))
    input_shape_od_model = (512, 512)
    min_score_thresh, max_boxes_to_draw = .47, 50
    length_axis = 50
    prev_frame_time, new_frame_time = 0, 0

    hhp_model = tf.keras.models.load_model(config.hhp_model, custom_objects={"tf": tf})

    vid = cv2.VideoCapture(0)

    while True:

        ret, img = vid.read()

        img_resized, new_old_shape = resize_preserving_ar(img, input_shape_od_model)

        new_frame_time = time.time()
        detections, _ = detect(model_detection, img_resized, min_score_thresh, new_old_shape)

        kpt = percentage_to_pixel(img.shape, detections['detection_keypoints'], detections['detection_keypoint_scores'])

        img_res = img.copy()

        i = 0
        for kpt_person in kpt:
            img_res = draw_key_points_pose(img_res, kpt_person)

            face_kpt = get_face_points(kpt_person, 'centernet')

            mean_x = np.mean([face_kpt[i] for i in range(0, 15, 3) if face_kpt[i] != 0.0])
            mean_y = np.mean([face_kpt[i + 1] for i in range(0, 15, 3) if face_kpt[i + 1] != 0.0])

            face_kpt_normalized = np.array(normalize_wrt_maximum_distance_point(face_kpt, mean_x, mean_y))

            input_kpts = tf.cast(np.expand_dims(face_kpt_normalized, 0), tf.float32)

            y, p, r = hhp_model(input_kpts, training=False)

            yaw, yaw_unc = y[:, 0].numpy()[0], y[:, 1].numpy()[0]
            pitch, pitch_unc = p[:, 0].numpy()[0], p[:, 1].numpy()[0]
            roll, roll_unc = r[:, 0].numpy()[0], r[:, 1].numpy()[0]

            img_res = draw_axis(yaw, pitch, roll, img_res, mean_x, mean_y, length_axis, yaw_unc, pitch_unc, roll_unc)

        fps = 1. / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        print("FPS: ", fps)
        cv2.imshow("", img_res)
        # cv2.waitKey(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()
