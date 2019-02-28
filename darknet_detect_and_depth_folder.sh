#   -dir_stereo : threshold for detection probability
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
#   -result : path to the result file of which each line shows [the image file name, millimeter in depth of target, disparity of target, x coord of target rect in left image, y coord of target rect in left image, width of target rect in left image, height of terget in left image.
#   -cam_intr : path to the yml file of stereo camera intrinsic parameters.
#   -cam_extr : path to the yml file of stereo camera extrinsic parameters.

#file_gt = "$1"
#   ZED WVGA, 8-way path aggregation
 ./darknet detector folder cfg/coco.data cfg/yolov3-tiny.cfg /mnt/sdcard/darknet_ori/yolov3-tiny.weights "$2" -ext png -gray4stereo -thresh 0.4 -w 1344 -h 376 -lrf -1 -depth 1 -both 0 -path_agg 8 -alfa -1 -percent_closest 20 -class "$1" -shrink 1.0 -result left_"$1"_millimeter_disparity_x_y_width_height_shrink_1.0.txt -cam_intr /mnt/sdcard/opencv_stereo_calibration/data/zed_672x376/intrinsics.yml -cam_extr /mnt/sdcard/opencv_stereo_calibration/data/zed_672x376/extrinsics.yml

#   ZED WVGA, 4-way path aggregation
#./darknet detector folder cfg/coco.data cfg/yolov3-tiny.cfg /mnt/sdcard/darknet_ori/yolov3-tiny.weights /media/ubuntu/ZED/zed_raw_images/temp -ext png -gray4stereo -thresh 0.4 -w 1344 -h 376 -lrf -1 -depth 1 -both 1 -path_agg 4 -alfa -1 -percent_closest 20 -class person -shrink 1.0 -result left_person_millimeter_disparity_x_y_width_height.txt -cam_intr /mnt/sdcard/opencv_stereo_calibration/data/zed_672x376/intrinsics.yml -cam_extr /mnt/sdcard/opencv_stereo_calibration/data/zed_672x376/extrinsics.yml

#   ZED HD, 8-way path aggregation
#./darknet detector demo cfg/coco.data cfg/yolov3-tiny.cfg /mnt/sdcard/darknet_ori/yolov3-tiny.weights -c 1 -thresh 0.4 -gray4stereo -w 2560 -h 720 -lrf -1 -depth 1 -both 0 -path_agg 8 -alfa -1 -percent_closest 20 -class person -cam_intr /mnt/sdcard/opencv_stereo_calibration/data/zed_1280x720/intrinsics.yml -cam_extr /mnt/sdcard/opencv_stereo_calibration/data/zed_1280x720/extrinsics.yml

#   ZED HD, 4-way path aggregation
#./darknet detector demo cfg/coco.data cfg/yolov3-tiny.cfg /mnt/sdcard/darknet_ori/yolov3-tiny.weights -c 1 -thresh 0.4 -gray4stereo -w 2560 -h 720 -lrf -1 -depth 1 -both 0 -path_agg 4 -alfa -1 -percent_closest 20 -class person -cam_intr /mnt/sdcard/opencv_stereo_calibration/data/zed_1280x720/intrinsics.yml -cam_extr /mnt/sdcard/opencv_stereo_calibration/data/zed_1280x720/extrinsics.yml




