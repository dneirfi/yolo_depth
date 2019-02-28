#!/bin/sh

set -e

#if [ 1 -eq 0 ]; then

IS_OCAM=0

#CAM_IDX=0

CAM_IDX=1

ZED=zed
OCAMS=ocams

#CAMERA=$ZED                                                                                  
CAMERA=$OCAMS  
if [ $CAMERA = $ZED ]; then 
	WIDTH=672
	HEIGHT=376
elif [ $CAMERA = $OCAMS ]; then
	WIDTH=640
	HEIGHT=360
	IS_OCAM=1
else 
	WIDTH=640
	HEIGHT=480
fi
echo "width : $WIDTH , height : $HEIGHT"      
echo "IS_OCAM:$IS_OCAM..."   

#fi #[ 1 -eq 0 ]

SQUARE_MM=24.95
WIDTH_CHESS=10    
HEIGHT_CHESS=7
OBJECT=clock
ALPHA=0
SHRINK=0.5
CAM_ENV="$CAMERA"_"$WIDTH"x"$HEIGHT"
echo "CAM_ENV : $CAM_ENV"

CALIB_ENV="$CAM_ENV"_"$WIDTH_CHESS"x"$HEIGHT_CHESS"_"$SQUARE_MM"
echo "CALIB_ENV : $CALIB_ENV"

#DIR_STEREO=~/work/eyedea/stereo_camera_calibration/master
DIR_STEREO=/mnt/sdcard/gov_project/stereo_camera_calibration/master
#DIR_STEREO=$PWD
echo "DIR_STEREO : $DIR_STEREO"
[ ! -d "$DIR_STEREO" ] && echo "Error: directory '$DIR_STEREO' does NOT exist." && exit
DIR_CALIB="$DIR_STEREO"/data/"$CALIB_ENV"
echo "DIR_CALIB : $DIR_CALIB"
[ ! -d "$DIR_CALIB" ] && echo "Error: directory '$DIR_CALIB' does NOT exist." && exit

#DIR_DISTANCE=~/work/eyedea/stereo_depth_and_detection/master
DIR_DISTANCE=$PWD
echo "DIR_DISTANCE : $DIR_DISTANCE"
[ ! -d "$DIR_DISTANCE" ] && echo "Error: directory '$DIR_DISTANCE' does NOT exist." && exit

DIR_CAM="$DIR_DISTANCE"/data/"$CAM_ENV"
echo "DIR_CAM : $DIR_CAM"
[ ! -d "$DIR_CAM" ] && echo "Error: directory '$DIR_CAM' does NOT exist." && exit

DIR_SBS_IMG=$DIR_DISTANCE/data/$CAM_ENV
echo "DIR_SBS_IMG : $DIR_SBS_IMG"
[ ! -d "$DIR_SBS_IMG" ] && echo "Error: directory $DIR_SBS_IMG does NOT exist." && exit

FILE_GT_VS_EST="$DIR_CAM"/left_"$OBJECT"_mm_disp_x_y_w_h_shrink_"$SHRINK".txt
echo "FILE_GT_VS_EST : $FILE_GT_VS_EST"
#[ -e "$FILE_GT_VS_EST" ] && echo "$FILE_GT_VS_EST does already exists. So it is deleted." && rm -- $FILE_GT_VS_EST

FILE_INTR="$DIR_CALIB"/intrinsics_"$CALIB_ENV".yml
echo "FILE_INTR : $FILE_INTR"

FILE_EXTR="$DIR_CALIB"/extrinsics_"$CALIB_ENV".yml
echo "FILE_EXTR : $FILE_EXTR"

FILE_CLASS=cfg/coco.data
echo "FILE_CLASS : $FILE_CLASS"
[ ! -f "$FILE_CLASS" ] && echo "$FILE_CLASS does NOT exist." && exit

DETECTOR=yolov3-tiny
echo "DETECTOR : $DETECTOR"

FILE_CONFIG=cfg/"$DETECTOR".cfg
echo "FILE_CONFIG : $FILE_CONFIG"
[ ! -f "$FILE_CONFIG" ] && echo "$FILE_CONFIG does NOT exist." && exit

FILE_WEIGHT=/mnt/sdcard/darknet_ori/"$DETECTOR".weights
#FILE_WEIGHT=/home/shinyeong/stereo_depth_and_detection/"$DETECTOR".weights
echo "FILE_WEIGHT : $FILE_WEIGHT"
[ ! -f "$FILE_WEIGHT" ] && echo "$FILE_WEIGHT does NOT exist." && exit

FILE_IMG_LIST="$DIR_CALIB"/stereo_image_list_$CALIB_ENV.xml
echo "FILE_IMG_LIST : $FILE_IMG_LIST"
[ ! -f "$FILE_IMG_LIST" ] && echo "$FILE_IMG_LIST does NOT exist." && exit

FILE_POLYFIT="$DIR_CALIB"/1st_disp_2nd_mm_polyfit_1st_const_last_3rd_"$CAM_ENV"_alfa_"$ALPHA"_shrink_"$SHRINK".txt
echo "FILE_POLYFIT : $FILE_POLYFIT"

# 스테레오 카메라 calibration
echo "save_stereo_images starts."
#$DIR_STEREO/save_stereo_images_exe -ocam=$IS_OCAM -cam=$CAM_IDX -s_mm=$SQUARE_MM -w=$WIDTH_CHESS -h=$HEIGHT_CHESS -width=$WIDTH -height=$HEIGHT -image_list=$FILE_IMG_LIST -dir_img=$DIR_CALIB -th_overlap=0.6 -sec_int=7
echo "save_stereo_images finishes."
echo "stereo_calib_eyedea starts."
#$DIR_STEREO/stereo_calib_eyedea_exe -alfa=$ALPHA -s=$SQUARE_MM -w=$WIDTH_CHESS -h=$HEIGHT_CHESS -dir_img=$DIR_CALIB -dir_calib=$DIR_CALIB -input=$FILE_IMG_LIST -postfix=$CALIB_ENV                                   
echo "stereo_calib_eyedea finishes."
echo "get_rectified_stereo starts."
#   image file version 
#$DIR_STEREO/get_rectified_stereo_exe -input=$FILE_IMG_LIST -int=$FILE_INTR -ext=$FILE_EXTR -alfa=$ALPHA -post=alfa_$ALPHA -sec=5 -dir_img=$DIR_CALIB -dir_rect=$DIR_CALIB/rectified_result                                                                    
#   camera version
#$DIR_STEREO/get_rectified_stereo_exe -ocam=$IS_OCAM -cam=$CAM_IDX -width=$WIDTH -height=$HEIGHT -int=$FILE_INTR -ext=$FILE_EXTR -alfa=$ALPHA -post=alfa_$ALPHA -sec=1 -dir_rect=$DIR_CALIB/rectified_result
echo "get_rectified_stereo finishes."

# ground truth 거리마다 이미지를 저장
#echo $DIR_DISTANCE/darknet detector measure "$FILE_CLASS" "$FILE_CONFIG" "$FILE_WEIGHT" "$DIR_SBS_IMG" -ocam "$IS_OCAM" -c "$CAM_IDX" -thresh 0.4 -lrf -1 -w "$((WIDTH * 2))" -h "$HEIGHT" -class "$OBJECT" -start 500 -interval 50
#$DIR_DISTANCE/darknet detector measure "$FILE_CLASS" "$FILE_CONFIG" "$FILE_WEIGHT" "$DIR_SBS_IMG" -ocam "$IS_OCAM" -c "$CAM_IDX" -thresh 0.4 -lrf -1 -w "$((WIDTH * 2))" -h "$HEIGHT" -class "$OBJECT" -start 500 -interval 50

# 저장된 이미지를 캘리브레이션 정보로 rectify 시킨 후 disparity를 구하고, callibrationt시 구한 focal lengh로 distance estimation을 해서 ground truth와 estimated distance와의 비교표를 만든다.
#$DIR_DISTANCE/darknet detector folder "$FILE_CLASS" "$FILE_CONFIG" "$FILE_WEIGHT" "$DIR_SBS_IMG" -ext png -gray4stereo -thresh 0.4 -lrf -1 -depth 1 -both 0 -path_agg 8 -alfa "$ALPHA" -percent_closest 20 -class "$OBJECT" -shrink "$SHRINK" -result "$FILE_GT_VS_EST" -cam_intr $FILE_INTR -cam_extr $FILE_EXTR

# ground truth와 estimated distance와의 비교표로 부터 3차원 fitting을 통해 estimated -> real distance mapping을 만든다.
#echo python3 $DIR_DISTANCE/ransac_polyfit.py $FILE_GT_VS_EST $FILE_POLYFIT
#python3 $DIR_DISTANCE/ransac_polyfit.py $FILE_GT_VS_EST $FILE_POLYFIT
# 카메라로 부터 이미지를 읽어 들이면서 estimated -> real distance mapping을 적용하여 해당 target object에 대해 distance를 display한다.
printf "\n"
#echo $DIR_DISTANCE/darknet detector demo "$FILE_CLASS" "$FILE_CONFIG" "$FILE_WEIGHT" -ocam "$IS_OCAM" -gray4stereo -c "$CAM_IDX" -thresh 0.4 -w "$((WIDTH * 2))" -h "$HEIGHT" -lrf -1 -depth 1 -both 0 -path_agg 4 -alfa "$ALPHA" -percent_closest 20 -class "$OBJECT" -shrink "$SHRINK" -poly $FILE_POLYFIT -cam_intr $FILE_INTR -cam_extr $FILE_EXTR
printf "\n"
$DIR_DISTANCE/darknet detector demo "$FILE_CLASS" "$FILE_CONFIG" "$FILE_WEIGHT" -ocam "$IS_OCAM" -gray4stereo -c "$CAM_IDX" -thresh 0.4 -w "$((WIDTH * 2))" -h "$HEIGHT" -lrf -1 -depth 1 -both 0 -path_agg 4 -alfa "$ALPHA" -percent_closest 20 -class "$OBJECT" -shrink "$SHRINK" -poly $FILE_POLYFIT -cam_intr $FILE_INTR -cam_extr $FILE_EXTR
