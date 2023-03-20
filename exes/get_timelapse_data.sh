mkdir -p ./datasets/timelapse
curl -L "https://universe.roboflow.com/ds/dF828PMiOz?key=OF0eFoDZDa" > roboflow.zip; unzip roboflow.zip -d ./datasets/timelapse; rm roboflow.zip