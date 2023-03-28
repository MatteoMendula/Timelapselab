import cv2
import os

RTSP_URL_0 = 'rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4'
RTSP_URL_1 = 'rtsp://localhost:8554/stream1'
RTSP_URL_2 = 'rtsp://localhost:8554/stream2'
RTSP_URL_3 = 'rtsp://localhost:8554/stream3'
RTSP_URL_4 = 'rtsp://localhost:8554/stream4'

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

cap_0 = cv2.VideoCapture(RTSP_URL_0, cv2.CAP_FFMPEG)
cap_1 = cv2.VideoCapture(RTSP_URL_1, cv2.CAP_FFMPEG)
cap_2 = cv2.VideoCapture(RTSP_URL_2, cv2.CAP_FFMPEG)
cap_3 = cv2.VideoCapture(RTSP_URL_3, cv2.CAP_FFMPEG)
cap_4 = cv2.VideoCapture(RTSP_URL_4, cv2.CAP_FFMPEG)

if not cap_0.isOpened():
    print('Cannot open RTSP stream')
    exit(-1)

while True:
    _, frame0 = cap_0.read()
    _, frame1 = cap_1.read()
    _, frame2 = cap_2.read()
    _, frame3 = cap_3.read()
    _, frame4 = cap_4.read()
    # resize = cv2.resize(frame, (1920, 1080)) 
    cv2.imshow('RTSP stream0', frame0)
    cv2.imshow('RTSP stream1', frame1)
    cv2.imshow('RTSP stream2', frame2)
    cv2.imshow('RTSP stream3', frame3)
    cv2.imshow('RTSP stream4', frame4)
    # cv2.imshow('RTSP stream resized', resize)

    if cv2.waitKey(50) == 13 :      # Specifying (Enter) button to break the loop
        break

cap_1.release()
cap_2.release()
cap_3.release()
cap_4.release()
cv2.destroyAllWindows()