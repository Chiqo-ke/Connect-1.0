import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time
import pickle

# Initialize MediaPipe Hand and Pose tracking
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# Load the saved motion data with equations
with open("motion_data_with_equations.pkl", "rb") as f:
    motion_data_with_equations = pickle.load(f)

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to calculate velocity and acceleration
def calculate_velocity(landmark_sequence):
    return np.gradient(landmark_sequence, axis=0)

def calculate_acceleration(landmark_sequence):
    return np.gradient(np.gradient(landmark_sequence, axis=0), axis=0)

# Function to compare motion based on landmarks, velocity, and acceleration
def compare_motion(current_landmarks, saved_landmarks, saved_velocity, saved_acceleration, tolerance=0.2):
    current_velocity = calculate_velocity(current_landmarks)
    current_acceleration = calculate_acceleration(current_landmarks)
    position_diff = np.linalg.norm(current_landmarks - saved_landmarks)
    velocity_diff = np.linalg.norm(current_velocity - saved_velocity)
    acceleration_diff = np.linalg.norm(current_acceleration - saved_acceleration)
    total_diff = position_diff + velocity_diff + acceleration_diff
    max_diff = tolerance * 3  # Adjust this value for flexibility
    similarity = max(0, 1 - (total_diff / max_diff))
    return similarity * 100  # Return as percentage similarity

# Initialize webcam for real-time tracking
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands, mp_pose.Pose() as pose:
    last_check_time = time.time()
    best_label_display = ""  # Store the best label for display
    
    # New variables for sign capture duration
    current_sign = None
    sign_start_time = None
    min_capture_duration = 3  # Minimum time to capture a sign (in seconds)
    recognition_cooldown = 5  # Cooldown time after recognizing a sign
    last_recognition_time = 0
    recognition_in_progress = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = time.time()
        
        # Convert to RGB for MediaPipe processing
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect hands and pose
        result_hands = hands.process(rgb_image)
        result_pose = pose.process(rgb_image)
        
        if result_hands.multi_hand_landmarks:
            for hand_landmarks in result_hands.multi_hand_landmarks:
                current_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                best_similarity = 0
                best_label = None
                
                for saved_landmarks, saved_velocity, saved_acceleration, label in motion_data_with_equations:
                    similarity = compare_motion(current_landmarks, saved_landmarks, saved_velocity, saved_acceleration)
                    # Track the best match
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_label = label
                
                # Check if enough time has passed since last recognition
                if current_time - last_recognition_time >= recognition_cooldown:
                    # New sign capture logic
                    if best_similarity >= 15:
                        # If no current sign or a new sign is detected
                        if current_sign != best_label:
                            current_sign = best_label
                            sign_start_time = current_time
                        
                        # Check if the sign has been held long enough
                        if current_time - sign_start_time >= min_capture_duration:
                            print(f"Recognized: {current_sign} with {best_similarity:.2f}% similarity.")
                            engine.say(current_sign)  # Speak out the label
                            engine.runAndWait()
                            best_label_display = current_sign
                            
                            # Set last recognition time to start cooldown
                            last_recognition_time = current_time
                            
                            # Reset for next sign
                            current_sign = None
                            sign_start_time = None
                    else:
                        # Reset if no similar sign is detected
                        current_sign = None
                        sign_start_time = None
        
        # Display the frame with recognized sign
        if best_label_display:
            cv2.putText(frame, f"Recognized: {best_label_display}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Show the video feed
        cv2.imshow('Webcam Feed', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()