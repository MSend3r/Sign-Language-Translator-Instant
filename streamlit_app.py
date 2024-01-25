# %%
#Import necessary libraries
import streamlit as st
import threading
import numpy as np
import os
import string
import mediapipe as mp
import cv2
from my_functions import *
import keyboard
from tensorflow.keras.models import load_model
import language_tool_python as grammer_checker
from variables import actions, frames
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer
import av

# Locking frames from multi threading
lock = threading.Lock()

# Initialize variables
sentence, keypoints, last_prediction, grammar_result = [], [], [], []
parser = grammer_checker.LanguageTool('en-US')
fig_place = st.empty()
log = st.empty()
img_container = {"img": None}

# Load the trained model
model = load_model('baseline_model')

# Deployment configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def video_frame_callback(frame):
    global sentence, keypoints, last_prediction, grammar_result, lock

    image = frame.to_ndarray(format="bgr24")
    # with lock:
    #     img_container["img"] = img

    # return frame

    # Create a holistic object for sign prediction
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # image = cv2.flip(img,1)

        # Process the image and obtain sign landmarks using image_process function from my_functions.py
        results = image_process(image, holistic)
        # Draw the sign landmarks on the image using draw_landmarks function from my_functions.py
        draw_landmarks(image, results)

        # Process the image and obtain sign landmarks using image_process function from my_functions.py
        results = image_process(image, holistic)

        landmarks = keypoint_extraction(results)

        # Extract keypoints from the pose landmarks using keypoint_extraction function from my_functions.py
        with lock:
            keypoints.append(landmarks)
            if len(keypoints) > frames:
                keypoints = keypoints[-frames:]

            cv2.putText(image, str(len(keypoints)), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
            # Check if 10 frames have been accumulated
            if len(keypoints) == frames:
                # Convert keypoints list to a numpy array
                keypoints = np.array(keypoints)
                # Make a prediction on the keypoints using the loaded model
                prediction = model.predict(keypoints[np.newaxis, :, :])
                # Clear the keypoints list for the next set of frames
                keypoints = []
                cv2.putText(image, actions[np.argmax(prediction)], (3,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Check if the maximum prediction value is above 0.9
                if np.amax(prediction) > 0.9:
                    # Check if the predicted sign is different from the previously predicted sign
                    if last_prediction != actions[np.argmax(prediction)] and actions[np.argmax(prediction)] != 'blank':
                        # Append the predicted sign to the sentence list
                        sentence.append(actions[np.argmax(prediction)])
                        # Record a new prediction to use it on the next cycle
                        last_prediction = actions[np.argmax(prediction)]

            # Limit the sentence length to 7 elements to make sure it fits on the screen
            if len(sentence) > 7:
                sentence = sentence[-7:]
                # Moving grammar check to 7 signs detected
                text = ' '.join(sentence)
                grammar_result = parser.correct(text) 

            # # Reset if the "Spacebar" is pressed
            # if keyboard.is_pressed(' '):
            #     sentence, keypoints, last_prediction, grammar_result = [], [], [], []

            # Check if the list is not empty
            if sentence:
                # Capitalize the first word of the sentence
                sentence[0] = sentence[0].capitalize()

            # Check if the sentence has at least two elements
            if len(sentence) >= 2:
                # Check if the last element of the sentence belongs to the alphabet (lower or upper cases)
                if sentence[-1] in string.ascii_lowercase or sentence[-1] in string.ascii_uppercase:
                    # Check if the second last element of sentence belongs to the alphabet or is a new word
                    if sentence[-2] in string.ascii_lowercase or sentence[-2] in string.ascii_uppercase or (sentence[-2] not in actions and sentence[-2] not in list(x.capitalize() for x in actions)):
                        # Combine last two elements
                        sentence[-1] = sentence[-2] + sentence[-1]
                        sentence.pop(len(sentence) - 2)
                        sentence[-1] = sentence[-1].capitalize()

            # # Perform grammar check if "Enter" is pressed
            # if keyboard.is_pressed('enter'):
            #     with lock:
            #         # Record the words in the sentence list into a single string
            #         text = ' '.join(sentence)
            #         grammar_result = parser.correct(text) 

            if grammar_result:
                # Calculate the size of the text to be displayed and the X coordinate for centering the text on the image
                textsize = cv2.getTextSize(grammar_result, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_X_coord = (image.shape[1] - textsize[0]) // 2

                # Draw the sentence on the image
                cv2.putText(image, grammar_result, (text_X_coord, 470),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                # Calculate the size of the text to be displayed and the X coordinate for centering the text on the image
                textsize = cv2.getTextSize(' '.join(sentence), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_X_coord = (image.shape[1] - textsize[0]) // 2

                # Draw the sentence on the image
                cv2.putText(image, ' '.join(sentence), (text_X_coord, 470),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            img_container["img"] = image
                            
        return av.VideoFrame.from_ndarray(image,format="bgr24")

def run_sign_detector():
    # global sentence, keypoints, last_prediction, grammar_result, lock

    cam = webrtc_streamer(
        key="Sign-Language-Detector",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        # video_processor_factory=OpenCVVideoProcessor,
        async_processing=True,
        video_frame_callback=video_frame_callback
    )

    # while cam.state.playing:
        # with lock:
        #     img = img_container["img"]
        # if img is None:
        #     continue
        # fig_place.image(img)

def main():
    st.title("Real Time Sign Language to Text by Maheep Singh")
    st.header("Description of the app")

    sign_language_det = "Sign Language Live Detector"
    app_mode = st.sidebar.selectbox( "Choose the app mode",
        [
            sign_language_det
        ],
    )

    st.subheader(app_mode)

    if app_mode == sign_language_det:
        run_sign_detector()

if __name__ == "__main__":
    main()