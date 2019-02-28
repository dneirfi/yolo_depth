#Same as darknet_detect_and_depth_cam.sh
#	Add -start : starting distance (millimeter)
#	    -interval : interval (millimeter)
#Press "tab" key, you can capture a image.
#./darknet detector measure cfg/coco.data cfg/yolov3-tiny.cfg /home/shinyeong/stereo_depth_and_detection/yolov3-tiny.weights -ocam -gray4stereo -c "$3" -thresh 0.4 -w 1280 -h 360 -lrf -1 -depth 1 -both 0 -path_agg "$2" -alfa -1 -percent_closest 20 -class "$1" -shrink 1.0 -poly /home/shinyeong/stereo_depth_and_detection_master/first_disparity_second_millimeter_3rd_polynomial_fit_first_constant_last_3rd_180920_shrink_0.5.txt -cam_intr /home/shinyeong/stereo_depth_and_detection_master/intrinsics.yml -cam_extr /home/shinyeong/stereo_depth_and_detection_master/extrinsics.yml -start 500 -interval 50

#./darknet detector measure cfg/coco.data cfg/yolov3-tiny.cfg /home/shinyeong/stereo_depth_and_detection/yolov3-tiny.weights sbs_images -ocam 1 -c "$3" -thresh 0.4 -lrf -1 -w 1280 -h 360 -class clock -start 500 -interval 50

./darknet detector measure cfg/coco.data cfg/yolov3-tiny.cfg yolov3-tiny.weights sbs_images -ocam 1 -c 1 -thresh 0.4 -lrf -1 -w 1280 -h 360 -class clock -start 850 -interval 50
