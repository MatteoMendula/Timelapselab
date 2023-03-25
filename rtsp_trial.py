import rtsp

RTSP_URL = 'rtsp://admin:Mp010201@10.111.45.211:554/Streaming/channels/101'

with rtsp.Client(RTSP_URL) as client:
    client.preview()