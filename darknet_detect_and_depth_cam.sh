#   -c : camera index
#   -thres : threshold for detection probability
#   -w : 2 X left(right) camera image width
#   -h : left(right) camera image height
#   -lrf : left(negative integer such as -1), right(positive integer such as 1) or full(zero) image for detection input 
#   -depth : non-zero for estismating depth, zero for no-depth estimation
#   -both : when estmating depth, non-zero for computing both left and right disparities, zero for only left disprity.
#   -path_agg : when semi-global block matching, 8 for 8-way path aggregation, 4 for 4-way path aggregation.
#   -alfa : alpha (alpha = -1(default) or 0 <= alpha <= 1) value for stereoRectify function
#   -percent_closest : for generalized median filter, 0 for the closest pixel from the camera, 100 for the furthest pixel from the camera.
#   -class : the class name of object to measure the ditance from camera.  It should one of the coco classes such as 'person'
#   -cam_intr : path to the yml file of stereo camera intrinsic parameters.
#   -cam_extr : path to the yml file of stereo camera extrinsic parameters.

#   ZED WVGA, 8-way path aggregation
#./darknet detector demo cfg/coco.data cfg/yolov3-tiny.cfg /mnt/sdcard/darknet_ori/yolov3-tiny.weights -gray4stereo -c 1 -thresh 0.4 -w 1344 -h 376 -lrf -1 -depth 1 -both 0 -path_agg 8 -alfa -1 -percent_closest 20 -class person -shrink 0.5 -cam_intr /mnt/sdcard/opencv_stereo_calibration/data/zed_672x376/intrinsics.yml -cam_extr /mnt/sdcard/opencv_stereo_calibration/data/zed_672x376/extrinsics.yml

#   ZED WVGA, 4-way path aggregation
#./darknet detector demo cfg/coco.data cfg/yolov3-tiny.cfg /mnt/sdcard/darknet_ori/yolov3-tiny.weights -gray4stereo -c 1 -thresh 0.4 -w 1344 -h 376 -lrf -1 -depth 1 -both 0 -path_agg "$2" -alfa -1 -percent_closest 20 -class "$1" -shrink 1.0 -poly first_disparity_second_millimeter_3rd_polynomial_fit_first_constant_last_3rd_180920_shrink_0.5.txt -cam_intr intrinsics.yml -cam_extr extrinsics.yml

#   ZED HD, 8-way path aggregation
#./darknet detector demo cfg/coco.data cfg/yolov3-tiny.cfg /mnt/sdcard/darknet_ori/yolov3-tiny.weights -c 1 -thresh 0.4 -gray4stereo -w 2560 -h 720 -lrf -1 -depth 1 -both 0 -path_agg 8 -alfa -1 -percent_closest 20 -class person -cam_intr /mnt/sdcard/opencv_stereo_calibration/data/zed_1280x720/intrinsics.yml -cam_extr /mnt/sdcard/opencv_stereo_calibration/data/zed_1280x720/extrinsics.yml

#   ZED HD, 4-way path aggregation
#./darknet detector demo cfg/coco.data cfg/yolov3-tiny.cfg /mnt/sdcard/darknet_ori/yolov3-tiny.weights -c 1 -thresh 0.4 -gray4stereo -w 2560 -h 720 -lrf -1 -depth 1 -both 0 -path_agg 4 -alfa -1 -percent_closest 20 -class person -cam_intr /mnt/sdcard/opencv_stereo_calibration/data/zed_1280x720/intrinsics.yml -cam_extr /mnt/sdcard/opencv_stereo_calibration/data/zed_1280x720/extrinsics.yml


#  oCamS-1CGN-U WVGA, 4-way path aggregation, shinyeong.
./darknet detector demo cfg/coco.data cfg/yolov3-tiny.cfg /home/shinyeong/stereo_depth_and_detection/yolov3-tiny.weights -ocam 0 -gray4stereo -c "$3" -thresh 0.4 -w 1280 -h 360 -lrf -1 -depth 1 -both 0 -path_agg "$2" -alfa -1 -percent_closest 20 -class "$1" -shrink 0.3 -poly /home/shinyeong/stereo_depth_and_detection_master/first_disparity_second_millimeter_3rd_polynomial_fit_first_constant_last_3rd_180920_shrink_0.5.txt -cam_intr /home/shinyeong/stereo_depth_and_detection_master/intrinsics.yml -cam_extr /home/shinyeong/stereo_depth_and_detection_master/extrinsics.yml



