#   darknet_detect_and_depth_folder.sh [class of interest] [path to left-right side-by-side image folder]
#sh darknet_detect_and_depth_folder.sh clock /media/ubuntu/ZED/zed_raw_images/captured_180920
#   darknet_detect_and_depth_cam.sh [class of interest] [path aggregation 4 or 8] [cam index]



sh darknet_detect_and_depth_measure.sh clock 4 0
#sh darknet_detect_and_depth_cam.sh clock 4 0
#sh darknet_detect_and_depth_cam.sh all 4 0 
#sh darknet_detect_and_depth_folder.sh clock 4 0
