import cv2
import mediapipe as mp
import pygame

pygame.init()

screen=pygame.display.set_mode((800,600))

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path="hand_landmarker.task"
    )
)

detector = vision.HandLandmarker.create_from_options(options)


cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    if not success:
        break

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(image)

    if len(result.hand_landmarks) > 0:
        landmark = result.hand_landmarks[0][8]

        x = int(landmark.x * img.shape[1])
        y = int(landmark.y * img.shape[0])

        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        ball_x=int((1-landmark.x) * 800)
        ball_y=int(landmark.y * 600)
        
        screen.fill((0,0,0))
        pygame.draw.circle(screen,(255, 182, 193),(ball_x,ball_y),10)
        pygame.display.flip()
        
    
    cv2.imshow("Camera", img)

    if cv2.waitKey(1) == ord("q"):
        break
