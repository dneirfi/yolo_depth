# Repo. for governmental project. #

Related jira post : http://support.eyedea.co.kr:8200/browse/VIS-8

Tasks : 

Real time object dectection and depth estimation to the objects from stereo camera streams.

![tasks](http://support.eyedea.co.kr:8200/secure/thumbnail/18266/_thumb_18266.png)

In breif, depth estimation codes are appended to Darknet's detector module.

In detail, the main body of codes is based on the yolo detection codes of darknet[1]. And depth estimation codes (based on [2]) is added to the demo part of darknet.
Since the depth estimation codes are written C++, the darkent C codes had to be converted in C++.

Many ideas for experimental settings are borrowed from [3]. 

[1] https://github.com/pjreddie/darknet

[2] https://github.com/dhernandez0/sgm

[3] https://www.youtube.com/watch?v=xjx4mbZXaNc

# Build and run #
To build
    ```
    ```
    ```make```

To run
    ```
    ```
    ```sh commands.sh```
    
# Parameters #
About commands.sh


    darknet_detect_and_depth_cam.sh [class of interest] [path aggregation 4 or 8] [cam index]
    
About darknet_detetct_and_depth_cam.sh


    ./darknet detector demo cfg/coco.data cfg/yolov3-tiny.cfg 
    [ path "yolov3-tiny.weights"] 
    [ -ocam ] 
    [ -gray4stereo] 
    [ -c : camera index] 
    [ -thres : threshold for detection probability] 
    [ -w : 2 X left(right) camera image width] 
    [ -h : left(right) camera image height]
    [ -lrf : left(negative integer such as -1), right(positive integer such as 1) or full(zero) image for detection input]
    [ -depth : non-zero for estismating depth, zero for no-depth estimation] 
    [ -both : when estmating depth, non-zero for computing both left and right disparities, zero for only left disprity.]
    [ -path_agg : when semi-global block matching, 8 for 8-way path aggregation, 4 for 4-way path aggregation.]
    [ -alfa : alpha (alpha = -1(default) or 0 <= alpha <= 1) value for stereoRectify function]
    [ -percent_closest : for generalized median filter, 0 for the closest pixel from the camera, 100 for the furthest pixel from the camera.]
    [ -class : the class name of object to measure the ditance from camera.  It should one of the coco classes such as 'person']
    [ -shrink 1.0 ] [ -poly : path "first_disparity_second_millimeter_3rd_polynomial_fit_first_constant_last_3rd_180920_shrink_0.5.txt"]
    [ -cam_intr : path "intrinsics.yml"] 
    [ -cam_extr : path "extrinsics.yml"]


# end2end.sh #
To run
    ```
    ```
    ```sh end2end.sh```
* Set camera option and directories
* Process  
  1. Stereo camera Calibration and Get rectified_stereo image files  
  2. Save ground truth distance images ( press 'tab' keys )  
  3. Rectify saved images and Get disparity. Make comparison table between ground truth distance and estimated distance  
  4. Make real distance mapping  
  5. Display distance of target object
