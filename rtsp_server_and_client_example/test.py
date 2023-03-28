import cv2
import os

RTSP_URL = 'rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4'

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print('Cannot open RTSP stream')
    exit(-1)

while True:
    _, frame = cap.read()
    # resize = cv2.resize(frame, (1920, 1080)) 
    cv2.imshow('RTSP stream', frame)
    # cv2.imshow('RTSP stream resized', resize)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()