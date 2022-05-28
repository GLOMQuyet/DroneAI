#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
from collections import Counter
from collections import deque
import cv2 as cv
import mediapipe as mp

from utils import CvFpsCalc
from utils import Select
from utils import Login
from model import KeyPointClassifier
from model import PointHistoryClassifier
from calc import Calc
from Pre import Preprocess
from Draw import *

from deepface import DeepFace
use_brect = True

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args

def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence


    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


    keypoint_classifier = KeyPointClassifier()


    point_history_classifier = PointHistoryClassifier()

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    #Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)

        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]

    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0

    while True:
        fps = cvFpsCalc.get()

        key = cv.waitKey(5)
        if key & 0xFF == ord('q'):
            break
        S = Select(key,mode)
        number, mode = S.select_mode()

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break

        image = cv.flip(image, 1)  # Mirror display

        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)

        image.flags.writeable = True

        #  ####################################################################
        # V = DeepFace.verify(img1_path='image/WIN_20220519_22_15_29_Pro.jpg', img2_path=image, model_name='Facenet',
        #                     detector_backend='retinaface', align = True, normalization = 'Facenet',enforce_detection=False)
        # print(V)
        # if V['verified'] == True:
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                c = Calc(debug_image, hand_landmarks)
                # Bounding box calculation
                brect = c.calc_bounding_rect()

                # Landmark calculation
                landmark_list = c.calc_landmark_list()

                # Conversion to relative coordinates / normalized coordinates
                p = Preprocess(debug_image,point_history,landmark_list)
                pre_processed_landmark_list = p.pre_process_landmark()
                pre_processed_point_history_list = p.pre_process_point_history()
                # Write to the dataset file
                L = Login(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)
                L.logging_csv()

                # Hand sign classification

                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                if hand_sign_id == 2:  # Point gesture
                    point_history.extend((landmark_list[4],landmark_list[8],landmark_list[12],landmark_list[16],landmark_list[20]))
                else:
                    point_history.append([0,0])

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                # Drawing part
                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )




        else:
            point_history.append([0,0])
        # else:
        #     cv.imshow('veri false',image)
        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
