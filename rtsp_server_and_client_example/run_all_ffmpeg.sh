ffmpeg -rtsp_transport tcp -i "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4" -rtsp_transport tcp -an -vf crop=in_w/2:in_h/2:0:0 -c:v h264 -c:a copy -f rtsp rtsp://localhost:8554/stream1 &
ffmpeg -rtsp_transport tcp -i "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4" -rtsp_transport tcp -an -vf crop=in_w/2:in_h/2:in_w:0 -c:v h264 -c:a copy -f rtsp rtsp://localhost:8554/stream2 &
ffmpeg -rtsp_transport tcp -i "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4" -rtsp_transport tcp -an -vf crop=in_w/2:in_h/2:0:in_h -c:v h264 -c:a copy -f rtsp rtsp://localhost:8554/stream3 &
ffmpeg -rtsp_transport tcp -i "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4" -rtsp_transport tcp -an -vf crop=in_w/2:in_h/2:in_w:in_h -c:v h264 -c:a copy -f rtsp rtsp://localhost:8554/stream4
