import cv2
import os

RTSP_URL = 'rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4'

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print('Cannot open RTSP stream')
    exit(-1)

width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

print(width, height)

# Dividing the original video in 4 parts by slicing the frame array
while True :
    photo = cap.read()[1]           # Storing the frame in a variable photo
    # photo = cv2.flip(photo,1)       # Fliping the photo for mirror view


    # cropu1 = photo[:240,0:320]      # Top left part of the photo
    # cropu2 = photo[:240,320:]       # Top right part of the photo
    # cropd1 = photo[240:,0:320]      # Bottom left part of the photo
    # cropd2 = photo[240:,320:]       # Bottom right part of the photo

    cropu1 = photo[:int(width/2),:int(height/2)]        # Top left part of the photo
    cropu2 = photo[:int(width/2),int(height/2):]        # Top right part of the photo
    cropd1 = photo[int(width/2):,:int(height/2)]      # Bottom left part of the photo
    cropd2 = photo[int(width/2):,int(height/2):]       # Bottom right part of the photo


    cv2.imshow("cropu1",cropu1)     # It will show cropu1 part in a window     
    cv2.imshow("cropu2",cropu2)     # It will show cropu2 part in a window
    cv2.imshow("cropd1",cropd1)     # It will show cropd1 part in a window
    cv2.imshow("cropd2",cropd2)     # It will show cropd2 part in a window
    
    cv2.imshow("Live",photo)        # It will show complete video in a window

    if cv2.waitKey(50) == 13 :      # Specifying (Enter) button to break the loop
        break


cv2.destroyAllWindows()            # To destroy all windows 