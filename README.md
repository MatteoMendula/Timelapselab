# Timelapselab


## Commands for running trackers

```
python track.py --tracking_source='rtsp://admin:Mp010201@10.111.45.211:554/Streaming/channels/101' --img_size 640 640 --tracking_method='bytetrack' --save_vid --save_crop --inference_mode='torch'
```

#### Options:
- `--tracking source`: defines the tracking source path. For our purposes it could be either a rtsp link, an http link or '0' for grabbing video directly from the camera of the machine running the script.
- `--images size`: defines the size to which the source is resized to. Models available now run only with 640x640 images.
- `--tracking_method`: defines the sota approach used to track objects. It could be either 'bytetrack' (preferred), 'strongsort', 'ocsort', 'botsort' or 'deepocsort'.
- `--save_vid`: if passed saves a video of tracked objects into experiments directory.
- `--save_crop`: if passed saves a copy of each object tracked for each video frame cropped around its detection area.
- `--inference_mode`: defines the mode (deeplearning library) used for inference. This was tested only with 'torch' and 'onnx' at the moment.