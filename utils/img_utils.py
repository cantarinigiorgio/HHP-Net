import cv2
import numpy as np
import math


body_parts = [(5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (11, 12), (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)]
face_points = [0, 1, 2, 3, 4]

color_pose = {
    "purple": (255, 0, 100),
    "light_pink": (80, 0, 255),
    "dark_pink": (220, 0, 255),
    "light_orange": (0, 80, 255),
    "dark_orange": (0, 220, 255),
    "blue": (255, 0, 0)
}


def draw_key_points_pose(image, kpts, radius=3, thickness=5):
    """
    Draw the key points and the lines connecting the body points

    Args:
        :image (numpy.ndarray): The image where the key points will be drawn
        :kpts (list): list of lists of points detected for each person [[x1, y1, c1], [x2, y2, c2],...] where x and y
            represent the coordinates of each point while c represents the confidence
        :radius (int): The radius value passed to cv2.circle function that represents the radius of the circle
            (default is 3)
        :thickness (int): The thickness value passed to cv2.circle function that represents the thickness of the circle
            (default is 5)

    Returns:
        :image (numpy.ndarray): The image with key points drawn
    """
    overlay = image.copy()

    for j in range(len(kpts)):
        color = color_pose["blue"]

        if j == face_points[0]:  # nose
            color = color_pose["purple"]
            cv2.circle(image, (int(kpts[j][0]), int(kpts[j][1])), radius, color, thickness)
        elif j == face_points[1]:  # left eye
            color = color_pose["light_pink"]
            cv2.circle(image, (int(kpts[j][0]), int(kpts[j][1])), radius, color, thickness)
        elif j == face_points[2]:  # right eye
            color = color_pose["dark_pink"]
            cv2.circle(image, (int(kpts[j][0]), int(kpts[j][1])), radius, color, thickness)
        elif j == face_points[3]:  # left ear
            color = color_pose["light_orange"]
            cv2.circle(image, (int(kpts[j][0]), int(kpts[j][1])), radius, color, thickness)
        elif j == face_points[4]:  # right eye
            color = color_pose["dark_orange"]
            cv2.circle(image, (int(kpts[j][0]), int(kpts[j][1])), radius, color, thickness)

    alpha = 0.2
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    return image


def draw_axis(yaw, pitch, roll, image, tdx=None, tdy=None, length_axis=50, yaw_uncertainty=-1, pitch_uncertainty=-1, roll_uncertainty=-1):
    """
    Project yaw pitch and roll on the image plane and draw them on the image if passed as input.

    Args:
        :yaw (float): value that represents the yaw rotation of the head
        :pitch (float): value that represents the pitch rotation of the head
        :roll (float): value that represents the roll rotation of the head
        :image (numpy.ndarray): The image where the three vector will be drawn
        :tdx (float): x coordinate from where the vector drawing start expressed in pixel coordinates
            (default is None)
        :tdy (float): y coordinate from where the vector drawing start expressed in pixel coordinates
            (default is None)
        :length_axis (int): value that will be multiplied to each x, y and z value that change the length of each vector that will be drawn
            (default is 50)
        :yaw_uncertainty (float): uncertainty value associated to yaw
            (default is -1)
        :pitch_uncertainty (float): uncertainty value associated to pitch
            (default is -1)
        :roll_uncertainty (float): uncertainty value associated to roll
            (default is -1)

    Returns:
        :image (numpy.ndarray): The image with the three axis drawn
    """
    res_image = image.copy()

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx is not None and tdy is not None:
        tdx = tdx
        tdy = tdy
    else:
        tdx = image.shape[0] / 2
        tdy = image.shape[1] / 2

    # PROJECT 3D TO 2D XY plane (Z = 0)

    # X-Axis pointing to right. drawn in red
    x1 = length_axis * (math.cos(yaw) * math.cos(roll)) + tdx
    y1 = length_axis * (math.cos(pitch) * math.sin(roll) + math.cos(roll) * math.sin(pitch) * math.sin(yaw)) + tdy

    # Y-Axis | drawn in green
    x2 = length_axis * (-math.cos(yaw) * math.sin(roll)) + tdx
    y2 = length_axis * (math.cos(pitch) * math.cos(roll) - math.sin(pitch) * math.sin(yaw) * math.sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = length_axis * (math.sin(yaw)) + tdx
    y3 = length_axis * (-math.cos(yaw) * math.sin(pitch)) + tdy
    # z3 = size * (cos(pitch) * cos(yaw)) + tdy

    if image is not None:
        cv2.line(res_image, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 2)
        cv2.line(res_image, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.line(res_image, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

    return res_image
