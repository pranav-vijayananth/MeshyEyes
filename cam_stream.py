### IMPORTS ###

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

### CONSTANTS ###

landmarker_model_path = "landmarker_model/face_landmarker.task"

### LANDMARKER MODEL/OBJECT ###

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def landmarker_result(
        result: FaceLandmarkerResult, 
        output_image: mp.Image, 
        timestamp_ms: int
): 
    print(f"Face Landmarker: {result}")

options = FaceLandmarkerOptions(
    base_options = BaseOptions(model_asset_path=landmarker_model_path), 
    running_mode = VisionRunningMode.LIVE_STREAM,
    result_callback=landmarker_result 
)

### VIDEO STREAM ###

cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, image = cap.read()

    if not success: 
        break

    cv2.imshow("Video Stream", image)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    with FaceLandmarker.create_from_options(options) as landmarker:
        landmarker.detect_async(mp_image, 45)


    if cv2.waitKey(100) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()