#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include <sys/time.h>
#include "disparity_method.h"
#include <stdlib.h>
#include <unistd.h>


#include "withrobot_camera.hpp"
#include <dirent.h>
#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <cstdlib>

#include <string.h>
#include <stdio.h>
#include <time.h>
#include <stdarg.h>

#define DEFAULT_EXPOSURE        150
#define DEFAULT_BRIGHTNESS      100
#define MAX_EXPOSURE            2000
#define MIN_EXPOSURE            10
#define MAX_BRIGHTNESS          255
#define MIN_BRIGHTNESS          10
#define INT_EXPOSURE            10
#define INT_BRIGHTNESS          5

int g_exposure = DEFAULT_EXPOSURE, g_brightness = DEFAULT_BRIGHTNESS;

pthread_cond_t cond_depth = PTHREAD_COND_INITIALIZER;
pthread_mutex_t mutex_lock_contour, mutex_lock_disp, mutex_lock_millimeter, mutex_lock_rect, mutex_lock_mask_remap;
#define DEMO 1



#ifdef OPENCV

#define SEC_INTERVAL        2
//#define SIZE_BUFFER         3

//#define ALL

std::vector <cv::Mat_< uchar > > mask_rect_remap_vector;
std::vector < float >  millimeter_quant_vector_copy;
std::vector <cv::Point3i> x_y_disp_quant_vector_copy;
char depth_vector[1000];
char *g_name_person = "person";
int g_idx_person, g_idx_coi;


static image im_raw;
static image im_object;
pthread_mutex_t mutex_lock_display;
pthread_mutex_t mutex_lock_image;
pthread_mutex_t mutex_lock_ipl;
static image im_letter;
static IplImage *ipl_l, *ipl_r;

//#define NO_RECTIFIED_MAP

#define LOG

#ifdef LOG


std::string getCurrentDateTime( std::string s )
{
    time_t now = time(0);
    struct tm tstruct;
    char log_buf[80];
    tstruct = *localtime(&now);
    if(s=="now"){
        strftime(log_buf, sizeof(log_buf), "%Y-%m-%d %X", &tstruct);
    } 	
    else if(s=="date"){
        strftime(log_buf, sizeof(log_buf), "%Y-%m-%d", &tstruct);
    }
    return std::string(log_buf);
};

//void Logger( std::string logMsg, int id, char *names )
void Logger(std::string& filePath, std::string logMsg, int id, char *names = NULL)
{
    time_t t;
    struct tm *tm;
    struct timeval tv;
    struct tm  tstruct;
    int id_img = id;
    char pb[1024];
    std::ofstream ofs(filePath.c_str(), std::ios::out |  std::ios::app );
    std::string now = getCurrentDateTime("now");

    gettimeofday(&tv,NULL);
    long int ms=tv.tv_sec*1000+tv.tv_usec/1000;
    sprintf(pb,"%06d", id_img);
    if (names)
    {
        //char dnames[4069];
        //strcpy(dnames, names);
        //ofs << now <<" "<< ms << " ms "<< '\t' << logMsg <<" "<< dnames <<" "<< pb <<'\n';
        ofs << now << " " << ms << " ms "<< '\t' << logMsg << " " << names <<" "<< pb <<'\n';
    }
    else
    {
        ofs << now << " " << ms << " ms "<< '\t' << logMsg << " " << pb <<'\n';
    }
    ofs.close();
}

int g_id_img = -1;
std::string g_fn_log = "Log_" + getCurrentDateTime("date") + ".txt";	


//void log_detected_objects(detection *dets, float demo_thresh, int nboxes, int demo_classes, int id_img)
void log_detected_objects(detection *dets, float th_prob, int n_box, int n_class, int id_img, char **obj_names)
{
	//char labelstr[4096] = {0};
	char labelstr[4096];
	int n_detected = 0;
	//for (int i = 0; i < nboxes; ++i)
	for (int i = 0; i < n_box; ++i)
	{
		//float prob_max = demo_thresh;
		float prob_max = th_prob;
		int idx_obj_max = -1; 
		//for(int j = 0; j < demo_classes; ++j)
		for(int j = 0; j < n_class; ++j)
		{
			if(dets[i].prob[j] > prob_max)
			{
				prob_max = dets[i].prob[j];
				idx_obj_max = j;
			}
		}
		if(idx_obj_max >= 0) 
		{
			if(n_detected)
			{
				strcat(labelstr, ", ");
				//strcat(labelstr, demo_names[j]);
				strcat(labelstr, obj_names[idx_obj_max]);
			}
			else
			{
				//strcpy(labelstr, demo_names[j]);
				strcpy(labelstr, obj_names[idx_obj_max]);
			}
			n_detected++;
		}
	}
	//printf("OOOOOOOOOO %s\n ", demo_names[j]);
	Logger(g_fn_log, "Detected objects are", id_img, labelstr);
	return;
}



#define TRIGGER

#endif	//	LOG





#define Y_RECT                  100
#define W_RECT                  300
#define H_RECT                  200

#define NMS                     0.4
#define X_OFFSET                -35
#define Y_OFFSET_DEPTH          55
#define Y_OFFSET_CLASS          22

#define MAX_LINE                512
//#define FONT_SCALE            0.7
//#define FONT_SCALE_CAM_OBJECT   0.8
#define FONT_SCALE_CAM_OBJECT   1.1
#define FONT_SCALE_CAM_NO_OBJ   1.2
#define FONT_SCALE_FOLDER       0.7

static int starting, interval;
static bool compute_depth = false;

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static network *net;


/*static*/ cv::Mat mat_disp;
/*static*/ cv::Rect rect_tgt;
cv::Rect rect_tgt_ratio;
/*static*/ cv::Mat mask_rect_remap;
std::vector < cv::Rect > rectvector(80);
std::vector <int> classnumvec(80);

#ifdef NO_RECTIFIED_MAP
#else   //  NO_RECTIFIED_MAP
std::vector<std::vector<cv::Point> > contour_remap;
std::vector<std::vector<cv::Point> > contour_remap_ratio;
#endif  //  NO_RECTIFIED_MAP
/*static*/ cv::Point3i x_y_millimeter;
std::vector<cv::Point3i> x_y_millimeter_vector;
static CvCapture * cap_ocv;
static Withrobot::Camera *cap_ocam;
static IplImage  * ipl;
//static float fps_detect = 0;
//static float fps_depth = 0;
static float demo_thresh = 0;
static float demo_hier = .5;
//static int running = 0;

static int demo_frame = 3;
static int demo_index = 0;
static float **predictions;
static float *avg;
static int demo_done = 0;
static int demo_total = 0;

char *cam_name[20];
bool is_measure=false;


int left_top_x,left_top_y,right_top_x,right_top_y,left_bot_x,left_box_y,right_bot_x,right_bot_y, lwidth, lheight, kount=0;
char left_name[100], right_name[100],full_name[1000];

#include <string>
#include <sstream>

namespace patch
{
	template < typename T > std::string to_string( const T& n )
	{
		std::ostringstream stm ;
		stm << n ;
		return stm.str() ;
	}
}




void get_boundingbox_coordinates(int x1, int y1, int x2, int y2, int x3, int y3, int x4, int y4, int w, int h)
{

	left_top_x=x1, left_top_y=y1;
	right_top_x=x2, right_top_y=y2;
	left_bot_x=x3, left_box_y=y3;
	right_bot_x=x4, right_bot_y=y4;
	lwidth=w; lheight=h;

}
void func_capture(const char *dir_img, int s, int i)
{	
	//int imw = ipl_l->width + ipl_r->width;
	//int imh = ipl_l->height + ipl_r -> height;
	//if ( compute_depth){
		//printf("CAPTURE IN**************************************");
		starting=s;
		interval=i;
		if ((starting + (interval*kount))<1000) { 
			//sprintf(left_name, "left_0%d_%d_%d_%d_%d.png",starting + (interval * kount), left_top_x, left_top_y,lwidth, lheight);	
			//sprintf(right_name, "right_0%d_%d_%d_%d_%d.png",starting + (interval * kount), left_top_x, left_top_y,lwidth, lheight);

			sprintf(full_name, "%s/stereo_0%d_%d_%d_%d_%d.png", dir_img, starting + (interval * kount), left_top_x, left_top_y,lwidth, lheight);
		}
		else {
			//sprintf(left_name, "%s/left_%d_%d_%d_%d_%d.png", dir_img, starting + (interval * kount), left_top_x, left_top_y,lwidth, lheight);	
			//sprintf(right_name, "%s/right_%d_%d_%d_%d_%d.png", dir_img, starting + (interval * kount), left_top_x, left_top_y,lwidth, lheight);
			sprintf(full_name, "%s/stereo_%d_%d_%d_%d_%d.png", dir_img, starting + (interval * kount), left_top_x, left_top_y,lwidth, lheight);
		}

		//sprintf(full_name, "data/ocams_640x360/stereo_0%d_%d_%d_%d_%d.png",starting + (interval * kount), left_top_x, left_top_y,lwidth, lheight);}
		/*	else {
			sprintf(left_name, "left_%d_%d_%d_%d_%d.png",starting + (interval * kount), left_top_x, left_top_y,lwidth, lheight);	
			sprintf(right_name, "right_%d_%d_%d_%d_%d.png",starting + (interval * kount), left_top_x, left_top_y,lwidth, lheight);
			sprintf(full_name, "data/ocams_640x360/stereo_%d_%d_%d_%d_%d.png",starting + (interval * kount), left_top_x, left_top_y,lwidth, lheight);}
			*/
		//frame = cvQueryFrame(cap);

		//IplImage *left_raw = cvSetImageROI(frame, cvRect(0, 0, frame->widthStep / 2, frame->height));
		//IplImage *right_raw = cvSetImageROI(frame, cvRect(frame->widthStep / 2, 0, frame->height / 2, frame->height));
		//cvSaveImage(left_name, ipl_l,0);
		//cvSaveImage(right_name, ipl_r,0);

		IplImage *ipl_full = cvCreateImage(cvSize(im_object.w*2,im_object.h),ipl_l->depth, ipl_l->nChannels);
		//printf("----ipl_full w: %d h: %d , ipl_l w: %d h: %d ,  ipl_r : %d h:%d\n", ipl_full->width, ipl_full->height, ipl_l->width, ipl_l->height, ipl_r->width, ipl_r->height);
		cvSetImageROI(ipl_full, cvRect(0,0,ipl_l->width,ipl_l->height));
		cvCopy(ipl_l, ipl_full);
		cvSetImageROI(ipl_full, cvRect(ipl_l->width,0,ipl_r->width*2,ipl_r->height));
		cvCopy(ipl_r,ipl_full);
		cvResetImageROI(ipl_full);
		//merge_horizontally(ipl_full, ipl_l_buff[idx], ipl_r_buff[idx]);

		cvSaveImage(full_name, ipl_full, 0);
		//cvReleaseImage(&ipl_full);
		//cvResetImageROI(ipl_l_copy);          
		//cvResetImageROI(ipl_r_copy);

		kount++;
		printf("----------------------------------Saved Image\n");

	//}
}

#ifdef SHINYEONG

void setoCams(){

	const char* devPath = "/dev/video0";
	Withrobot::Camera camera(devPath);

	/* 8-bit Bayer pattern GRBG image 1280 x 720 60 fps */
	camera.set_format(1280, 720, Withrobot::fourcc_to_pixformat('Y','U','Y','V'), 1, 60);   // for 1CGN
	/*
	 * get current camera format (image size and frame rate)
	 */
	Withrobot::camera_format camFormat;
	camera.get_current_format(camFormat);
	std::string camName = camera.get_dev_name();
	std::string camSerialNumber = camera.get_serial_number();
	int brightness = camera.get_control("Gain");
	int exposure = camera.get_control("Exposure (Absolute)");

	camera.set_control("Gain", brightness);
	camera.set_control("Exposure (Absolute)", exposure);

	cv::Mat srcImage(cv::Size(camFormat.width, camFormat.height), CV_8UC2);
	bool quit = false;
	while(!quit)
	{
		int size = cap_ocam->get_frame(srcImage.data, camFormat.image_size, 1);
		if(size == -1)
		{
			print("error number: %d\n", errno);
			perror("Cannot get image from camera");
			camera.stop();
			camera.start();
			continue;
		}
		cv::Mat dstImage[2];
		cv::Mat left_image, right_image;
		cv::split(srcImage, dstImage);
		cv::cvtColor(dstImage[0], right_image, CV_BayerBG2BGR);
		cv::cvtColor(dstImage[1], left_image, CV_BayerBG2BGR);

		cv::imshow(windowName + "_right", right_image);
		cv::imshow(windowName + "_left", left_image);
		cv::waitKey();


	}
}
#endif //SHINYEONG

//double demo_time;


int size_network(network *net)
{
	int i;
	int count = 0;
	for(i = 0; i < net->n; ++i){
		layer l = net->layers[i];
		if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
			count += l.outputs;
		}
	}
	return count;
}

void remember_network(network *net)
{
	int i;
	int count = 0;
	for(i = 0; i < net->n; ++i){
		layer l = net->layers[i];
		if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
			memcpy(predictions[demo_index] + count, net->layers[i].output, sizeof(float) * l.outputs);
			count += l.outputs;
		}
	}
}

detection *avg_predictions(network *net, int *nboxes)
{
	int i, j;
	int count = 0;
	fill_cpu(demo_total, 0, avg, 1);
	for(j = 0; j < demo_frame; ++j){
		axpy_cpu(demo_total, 1./demo_frame, predictions[j], 1, avg, 1);
	}
	for(i = 0; i < net->n; ++i){
		layer l = net->layers[i];
		if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
			memcpy(l.output, avg + count, sizeof(float) * l.outputs);
			count += l.outputs;
		}
	}
	detection *dets = get_network_boxes(net, im_raw.w, im_raw.h, demo_thresh, demo_hier, 0, 1, nboxes);
	return dets;
}


void get_min_max_disparity(cv::Point3i& p_min_dis, cv::Point3i& p_max_dis, cv::Mat& mat_disp, cv::Mat& mask01)
{
	cv::Point minLoc, maxLoc;
	double minVal, maxVal;
	cv::minMaxLoc(mat_disp, &minVal, &maxVal, &minLoc, &maxLoc, mask01);
	p_min_dis.x = minLoc.x; p_min_dis.y = minLoc.y; p_min_dis.z = (int)minVal;
	p_max_dis.x = maxLoc.x; p_max_dis.y = maxLoc.y; p_max_dis.z = (int)maxVal;
	return;
}

void draw_contour_on_iplImage_from_vector_of_point(IplImage *ipl, std::vector<cv::Point>& li_point, CvScalar kolor)
{
	int iP, n_pt = li_point.size();
	CvPoint pt;
	CvMemStorage *memStorage = cvCreateMemStorage(0);
	CvSeq* seq = cvCreateSeq(CV_SEQ_KIND_CURVE | CV_SEQ_ELTYPE_POINT | CV_SEQ_FLAG_CLOSED, sizeof(CvSeq), sizeof(CvPoint), memStorage);
	for(iP = 0; iP < n_pt; iP++)
	{
		pt.x = li_point[iP].x;  pt.y = li_point[iP].y;
		cvSeqPush(seq, &pt);
	}
	cvDrawContours(ipl, seq, kolor, kolor, 0, 1);
	cvReleaseMemStorage(&memStorage);
	//cvClearSeq(seq);
	return;
}

void draw_no_coi(IplImage *ipl, char *class_of_interest, char *prephix, CvScalar kolor, CvFont& phont)
{
	char text[100];
	sprintf(text, "%s'%s'", prephix, class_of_interest);
	cvPutText(ipl, text, CvPoint(20, ipl->height / 2), &phont, kolor);
}


int set_font_color(IplImage *ipl, int x_min, int y_min, int y_max)
{
    int wid = ipl->width * 0.2;
    int hei = y_max - y_min;
    //printf("x_min : %d, y_min : %d, wid : %d, hei : %d\n", x_min, y_min, wid, hei);
    cvSetImageROI(ipl, cvRect(x_min, y_min, wid, hei));
    CvScalar intensity_avg = cvAvg(ipl);
    //printf("intensity_avg.val[0] : %f\n", intensity_avg.val[0]);
    //printf("intensity_avg.val[1] : %f\n", intensity_avg.val[1]);
    //printf("intensity_avg.val[2] : %f\n", intensity_avg.val[2]);
    //printf("intensity_avg.val[3] : %f\n", intensity_avg.val[3]);
    cvResetImageROI(ipl);
    return intensity_avg.val[0] > 128 ? 0 : 255;
}    


void draw_depth_in_millimeter(IplImage *ipl, char *postphix, CvPoint& xy, double depth_millimeter, CvScalar kolor, CvFont& phont)
{        
    char text[100];
    sprintf(text, "%d mm", cvRound(depth_millimeter));
    int intensity = set_font_color(ipl, xy.x + X_OFFSET, xy.y + Y_OFFSET_CLASS, xy.y + Y_OFFSET_DEPTH);    
    kolor = cvScalarAll(intensity);
    cvPutText(ipl, text, CvPoint(xy.x + X_OFFSET, xy.y + Y_OFFSET_DEPTH), &phont, kolor);
    cvPutText(ipl, postphix, CvPoint(xy.x + X_OFFSET, xy.y + Y_OFFSET_CLASS), &phont, kolor);
/*
	sprintf(text, "%d mm %s", cvRound(depth_millimeter), postphix);
	cvPutText(ipl, text, CvPoint(xy.x + X_OFFSET_TEXT, xy.y + Y_OFFSET_TEXT), &phont, kolor);
*/
    cvCircle(ipl, xy, 3, kolor, 1);
} 

void draw_depth_in_millimeter(cv::Mat& mat, char *postphix, int x, int y, double millimeter, cv::Scalar kolor, int fontFace, double fontScale, int fontThickness)
{
	//std::string text = patch::to_string(cvRound(millimeter)) + " mm " + std::string(postphix);
	std::string txt_depth = patch::to_string(cvRound(millimeter)) + " mm";// + " mm " + std::string(postphix);
    cv::putText(mat, txt_depth, cv::Point(x + X_OFFSET, y + Y_OFFSET_DEPTH), fontFace, fontScale, kolor, fontThickness, 8);
	cv::putText(mat, std::string(postphix), cv::Point(x + X_OFFSET, y + Y_OFFSET_CLASS), fontFace, fontScale, kolor, fontThickness, 8);
	cv::circle(mat, cv::Point(x, y), 3, kolor, 1);
	return;

}


void draw_no_coi(cv::Mat& mat, char *class_of_interest, char *prephix, cv::Scalar kolor, int fontFace, double fontScale, int fontThickness)
{
	std::string text = std::string(prephix) + std::string("'") + std::string(class_of_interest) + std::string("'");
	cv::putText(mat, text, cv::Point(20, mat.rows / 2), fontFace, fontScale, kolor, fontThickness, 8);
}
/*
void draw_depth_in_millimeter(cv::Mat& mat, char *postphix, std::vector <cv::Point3i > &point, std::vector<float> &millimeter, cv::Scalar kolor, int fontFace, double fontScale, int fontThickness)
{
	std::string text = patch::to_string(cvRound(millimeter)) + " mm " + std::string(postphix);
	cv::putText(mat, text, cv::Point(x + X_OFFSET_TEXT, y + Y_OFFSET_TEXT), fontFace, fontScale, kolor, fontThickness, 8);
	cv::circle(mat, cv::Point(x, y), 3, kolor, 1);
	return;

}*/
int get_index_of_class(char *class_of_interest, char **names, int n_class)
{
	if(class_of_interest)
	{ 
		for(int j = 0; j < n_class; j++)
		{
			if(!strcmp(names[j], class_of_interest))
			{
				return j;
			}
		}
	}
	return -1;         
}


cv::Point3i get_quantile_disparity(cv::Mat& mat_disp, cv::Mat& mask01, int percent_closest)
{
	double m = (float)(100 - percent_closest) * (float)cv::countNonZero(mask01) / 100.0;
	//printf("DOUBLE m : %f \n ", m);
	int bin = 0;
	double med = -1.0;
	int histSize = 256;
	float range[] = {0, 256};
	const float* histRange = {range};
	bool uniform = true;
	bool accumulate = false;
	cv::Mat hist;//, mat_disp_masked = mat_disp.mul(mask01);
	//mask01.convertTo(mask01, CV_8UC1);
	//std::cout<< "||||||||||||"<<mask01.type()<<std::endl;	
	cv::calcHist(&mat_disp, 1, 0, mask01, hist, 1, &histSize, &histRange, uniform, accumulate);

	//std::cout << "m : " << m << std::endl;
	for(int iH = 0; iH < histSize; iH++)
	{
		bin += cvRound(hist.at<float>(iH));
		if (bin > m)
		{
			med = iH;
			break;
			//m = INT_MAX;
		}
	}
	//std::cout << "bin : " << bin << std::endl;
	cv::Mat mask_med;
	cv::inRange(mat_disp, cv::Scalar::all(med), cv::Scalar::all(med), mask_med);
	double minV, maxV;    cv::Point minL, maxL;
	cv::minMaxLoc(mask_med, &minV, &maxV, &minL, &maxL, mask01);
	//printf("IN DEPTH METHOD MAXL.x : %d, y : %d , med : %d\n", maxL.x, maxL.y, med);
	return cv::Point3i(maxL.x, maxL.y, med);

}


struct_poly *load_polynomial_coefficient(char *fn_poly)
{
	//bool is_coef_loaded = false;
	struct_poly *param_poly = NULL;
	if (fn_poly)
	{
		std::ifstream ifs(fn_poly);
		if(!ifs)
		{
			std::cerr << "Unable to open fn_poly : " << fn_poly << std::endl;
			exit(0);
		}
		param_poly = new struct_poly;
		char *p;
		char line[MAX_LINE];
		if(ifs.getline(line, MAX_LINE));
		{
			p = strtok(line, " ");
			while(p)
			{
				printf("disparity token: %s\n", p);
				param_poly->coef_disparity.push_back(atof(p));
				p = strtok(NULL, " ");
			}

		}
		if(ifs.getline(line, MAX_LINE));
		{
			p = strtok(line, " ");
			while(p)
			{
				printf("millimeter token: %s\n", p);
				param_poly->coef_millimeter.push_back(atof(p));
				p = strtok(NULL, " ");
			}
		}
		ifs.close();
		std::cout << "param_poly->coef_millimeter : " << std::endl;
		for(size_t i = 0; i < param_poly->coef_millimeter.size(); ++i) std::cout << param_poly->coef_millimeter[i] << std::endl;
		std::cout << "param_poly->coef_disparity : " << std::endl;
		for(size_t i = 0; i < param_poly->coef_disparity.size(); ++i) std::cout << param_poly->coef_disparity[i] << std::endl;
	}
	return param_poly;

}


void compute_mask_rect_remap(cv::Mat& mask_remap, cv::Mat& mask_remap_l, cv::Mat& mask_remap_r, cv::Rect& rect, struct_stereo *param_stereo, bool from_both)
{

	mask_remap_l.setTo(cv::Scalar::all(0));
	cv::rectangle(mask_remap_l, rect, cv::Scalar::all(1), -1, 8); 
	if (from_both)
	{

		undistort_and_rectify(mask_remap_l, param_stereo, true);
		mask_remap_r.setTo(cv::Scalar::all(0));
		cv::rectangle(mask_remap_r, rect, cv::Scalar::all(1), -1, 8); 
		undistort_and_rectify(mask_remap_r, param_stereo, false);
		mask_remap = mask_remap_l.mul(mask_remap_r, 255.);
	}
	else
	{
		mask_remap = 255 * mask_remap_l;
	}
	return;
}

void *depth_in_thread(void *ptr) {
	struct_depth *param_depth = (struct_depth *)ptr;
	int n_frm = 0;
	bool is_resumed;
	double fps_depth, sec_cur, sec_pre, sec_start = what_time_is_it_now();//demo_time_detect = what_time_is_it_now();
	sec_pre = sec_start; 
	cv::Mat mat_dis_ori, mat_dis_l, mask_rect_remap_copy;//)at_focal_mul_baseline;
	IplImage *ipl_l_copy = cvCreateImage(CvSize(ipl_l->width,ipl_l->height),ipl_l->depth,ipl_l->nChannels), *ipl_r_copy=cvCreateImage(CvSize(ipl_r->width, ipl_r->height),ipl_r->depth, ipl_r->nChannels);
#ifdef LOG
	int id_img_copy = -1;
#endif  //  LOG
	init_disparity_method(param_depth->p1, param_depth->p2);

	while (!demo_done) {  
		n_frm++;
		float elaps_ms, elaps_ms_l;

		pthread_mutex_lock(&mutex_lock_image);

		cvCopy(ipl_l, ipl_l_copy);
		cvCopy(ipl_r, ipl_r_copy);
#ifdef LOG
		id_img_copy = g_id_img;
#endif	//	LOG
		pthread_mutex_unlock(&mutex_lock_image);

#ifdef LOG
		if(id_img_copy >= 0)
		{
			Logger(g_fn_log, "Depth thread just received the image", id_img_copy);
		}
#endif	//	LOG

		mat_dis_l = compute_disparity_method(cv::cvarrToMat(ipl_l_copy), cv::cvarrToMat(ipl_r_copy), &elaps_ms_l, param_depth->path_agg);

		//mat_dis_l = compute_disparity_method(cv::cvarrToMat(ipl_l_buff), cv::cvarrToMat(ipl_r_buff), &elaps_ms_l, param_depth->path_agg);


		if (param_depth->both_disparity) 
		{
			float elaps_ms_r;
			cv::Mat mat_dis_r = compute_disparity_method(cv::cvarrToMat(ipl_r_copy), 
			cv::cvarrToMat(ipl_l_copy), &elaps_ms_r, param_depth->path_agg);
			cv::Mat mat_im_l = cv::cvarrToMat(ipl_l_copy);
			post_disparity_filter(mat_dis_ori, mat_dis_l, mat_dis_r, mat_im_l, param_depth->wls_filter);
			elaps_ms = elaps_ms_l + elaps_ms_r;
			//cv::imshow("mat_dis_r", mat_dis_r);	cv::imshow("mat_dis", mat_dis); cv::waitKey(1);
		}
		else 
		{
			mat_dis_ori = mat_dis_l;
			elaps_ms = elaps_ms_l;
		}
#ifdef LOG
		if(id_img_copy >= 0)
		{
			Logger(g_fn_log, "Depth map is computed", id_img_copy);
		}
#endif  //LOG
		is_resumed = false;
		pthread_mutex_lock(&mutex_lock_mask_remap);
		//while(!compute_depth) 
		if(!compute_depth) 
		{
			is_resumed = true;
			pthread_cond_wait(&cond_depth, &mutex_lock_mask_remap);
		}
		if(is_resumed)
		{
			n_frm = 1;
			sec_start = what_time_is_it_now();//demo_time_detect = what_time_is_it_now();
			sec_pre = sec_start; 

		}
		mask_rect_remap.copyTo(mask_rect_remap_copy);
		pthread_mutex_unlock(&mutex_lock_mask_remap);

		int n_pxl_mask = cv::countNonZero(mask_rect_remap_copy); 

		//printf("#######n_pxl_mask : %d\n", n_pxl_mask);
		cv::Point3i x_y_disp_quant(-1, -1, -1);
		double millimeter_quant = 0;
#ifdef ALL
//printf("MASK_RECT_REMAP_VETOR_SIZE : %d \n", mask_rect_remap_vector.size());
		std::vector < float >  millimeter_quant_vector;
		std::vector <cv::Point3i> x_y_disp_quant_vector;
		//millimeter_quant_vector[0] = '\0';
		if( n_pxl_mask)
		{
			
			//printf("qqqqqqqqqqqqqq\n");
			for(int i =0; i<rectvector.size(); i++)
			{
				//x_y_disp_quant = get_quantile_disparity(mat_dis_ori, mask_rect_remap_vector[i], param_depth->percent_closest);
				//millimeter_quant = disparity_2_millimeter(x_y_disp_quant.z, param_depth->parameter_stereo->focal_in_pixel, param_depth->parameter_stereo->baseline_in_millimeter, param_depth->parameter_poly);
			

				x_y_disp_quant_vector.push_back( get_quantile_disparity(mat_dis_ori, mask_rect_remap_vector[i], param_depth->percent_closest));
				//x_y_disp_quant_vector.resize((int)(x_y_disp_quant_vector.size()));
				x_y_disp_quant_vector_copy.clear();
				x_y_disp_quant_vector_copy.assign( x_y_disp_quant_vector.begin(), x_y_disp_quant_vector.end() );

				//std::copy( x_y_disp_quant_vector.begin(), x_y_disp_quant_vector.end(), x_y_disp_quant_vector_copy.begin() );

			
				//printf("xxxxxDEPTH IN THREAD : %d  /  %d\n", x_y_disp_quant_vector.size(),x_y_disp_quant_vector_copy.size());
				
				millimeter_quant_vector.push_back(cvRound(disparity_2_millimeter(x_y_disp_quant_vector[i].z,param_depth->parameter_stereo->focal_in_pixel, param_depth->parameter_stereo->baseline_in_millimeter, param_depth->parameter_poly)));
				
				millimeter_quant_vector_copy.clear();
				millimeter_quant_vector_copy.assign(millimeter_quant_vector.begin(), millimeter_quant_vector.end() );
				//printf("yyyyyDEPTH IN THREAD : %d  /  %d\n", millimeter_quant_vector.size(), millimeter_quant_vector_copy.size());
				//std::copy( millimeter_quant_vector.begin(), millimeter_quant_vector.end(), millimeter_quant_vector_copy.begin() );

//printf("MMMMMMMMMMMMMMMMMMMMMMMMMMM : %d\n", millimeter_quant_vector.size());
				//millimeter_quant_vector.push_back( cvRound(millimeter_quant_vector[i]));
				//printf("$$$$$$$$$$$$$$$$$$ %d //  %d\n",i,millimeter_quant_vector[i]);
				//x_y_disp_quant_vector[i].z=cvRound(millimeter_quant_vector[i]);
				//x_y_millimeter_vector.push_back(x_y_disp_quant_vector[i]);
				//depth_vector.push_back(cvRound(millimeter_quant_vector[i]));
				//depth_vector[i]=cvRound(millimeter_quant_vector[i]);
				//printf("------------------------------_%d\n", depth_vector[i]);
			}
			//millimeter_quant = disparity_2_millimeter(x_y_disp_quant.z, param_depth->parameter_stereo->focal_in_pixel, param_depth->parameter_stereo->baseline_in_millimeter, param_depth->parameter_poly);
			//printf("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@22\n"); 	
		
		}
#else //ALL
		//double millimeter_quant =0;
		if (n_pxl_mask)
		{
			//printf("qqqqqqqqqqqqqq\n");
			x_y_disp_quant = get_quantile_disparity(mat_dis_ori, mask_rect_remap_copy, param_depth->percent_closest);
			//std::cout << "x_y_disp_med : " << x_y_disp_med << std::endl;
			millimeter_quant = disparity_2_millimeter(x_y_disp_quant.z, param_depth->parameter_stereo->focal_in_pixel, param_depth->parameter_stereo->baseline_in_millimeter, param_depth->parameter_poly);
			
		}
		

		//printf("$$$$$$$$x_y_ x : %d, y : %d \n", x_y_disp_quant.x, x_y_disp_quant.y);
#endif  //  ALL
		pthread_mutex_lock(&mutex_lock_millimeter);
		x_y_millimeter.x = x_y_disp_quant.x;   x_y_millimeter.y = x_y_disp_quant.y;  x_y_millimeter.z = cvRound(millimeter_quant);


		pthread_mutex_unlock(&mutex_lock_millimeter);

		pthread_mutex_lock(&mutex_lock_disp);
		mat_dis_ori.copyTo(mat_disp);
		pthread_mutex_unlock(&mutex_lock_disp);


		sec_cur = what_time_is_it_now();
		if ((sec_cur - sec_pre) > SEC_INTERVAL) 
		{
			fps_depth = (double)(n_frm) / (sec_cur - sec_start);
			//printf("\033[2J");        printf("\033[1;1H");
			printf("\nFPS 4 depth : %.1f\n", fps_depth);
			//printf("Objects:\n\n");
			sec_pre = sec_cur;
		}
	}
	pthread_exit((void *) 0);
}



void *detect_in_thread(void *ptr)
{
	struct_detect *param_detect = (struct_detect *)ptr;
	cv::Rect rect_tgt_ori;
	cv::Rect rect_tgt_ori_ratio;
		//float nms = .4;
	float nms = NMS;
	layer l = net->layers[net->n - 1];
	int n_frm = 0;
#ifdef ALL
	std::vector <cv::Rect> rectvector_ori(80);
	for (int i=0; i < 80; i++){
		classnumvec.push_back(i);
	}
#else //ALL
	g_idx_coi = get_index_of_class(param_detect->class_of_interest, demo_names, demo_classes);
#endif  // ALL
	double fps_detect, sec_cur, sec_pre, sec_start = what_time_is_it_now();//demo_time_detect = what_time_is_it_now();
	sec_pre = sec_start; 

	image im_object_origin = copy_image(im_raw), im_letter_copy = copy_image(im_letter);
#ifdef LOG
	int id_img_copy = -1;
#endif	//	LOG

	while (!demo_done)
	{
		n_frm++;
		pthread_mutex_lock(&mutex_lock_image);
		if(!compute_depth)
		{
			copy_image_into(im_raw, im_object_origin); 
		}
		copy_image_into(im_letter, im_letter_copy);
#ifdef LOG
		id_img_copy = g_id_img;
#endif	//	LOG
		pthread_mutex_unlock(&mutex_lock_image);
#ifdef LOG
		if(id_img_copy >= 0)
		{
			Logger(g_fn_log, "Detection thread just received the image", id_img_copy);
		}
#endif	//	LOG
		float *X = im_letter_copy.data;
		network_predict(net, X);
		remember_network(net);
		detection *dets = 0;
		int nboxes = 0;
		dets = avg_predictions(net, &nboxes);
		if (nms > 0) do_nms_obj(dets, nboxes, l.classes, nms);
#ifdef LOG
		if(id_img_copy >= 0 && nboxes > 0)
		{
			log_detected_objects(dets, demo_thresh, nboxes, demo_classes, id_img_copy, demo_names);
		}
#endif	//	LOG


		if(compute_depth)
		{

#ifdef ALL

//			for(int i=0; i< 80; i++){
//				get_rect_of_object_of_interest(rectvector_ori[i], im_object_origin.w, im_object_origin.h, dets, nboxes, demo_thresh, classnumvec[i], param_detect->ratio_shrink);
//			}
			draw_detections(im_object_origin, dets, nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes,rectvector_ori);
			//printf("%%%%%%RECT x: %d , y : %d ", rectvector_ori[0].x, rectvector_ori[1]);
			pthread_mutex_lock(&mutex_lock_display);
			copy_image_into(im_object_origin, im_object);
			pthread_mutex_unlock(&mutex_lock_display);	
#else   //  ALL
			get_rect_of_object_of_interest(rect_tgt_ori, im_object_origin.w, im_object_origin.h,dets, nboxes, demo_thresh, g_idx_coi, param_detect->ratio_shrink);
			get_rect_of_object_of_interest(rect_tgt_ori_ratio,
			im_object_origin.w, im_object_origin.h,
			dets,nboxes,demo_thresh, g_idx_coi, 1);
#endif //ALL
			pthread_mutex_lock(&mutex_lock_rect);
			rect_tgt.x = rect_tgt_ori.x;    rect_tgt.y = rect_tgt_ori.y;    
			rect_tgt.width = rect_tgt_ori.width;    rect_tgt.height = rect_tgt_ori.height;
			rect_tgt_ratio.x = rect_tgt_ori_ratio.x; rect_tgt_ratio.y = rect_tgt_ori_ratio.y;
			rect_tgt_ratio.width = rect_tgt_ori_ratio.width; rect_tgt_ratio.height = rect_tgt_ori_ratio.height;
#ifdef ALL
			rectvector.assign(rectvector_ori.begin(), rectvector_ori.end());
			//printf("REEEEEEEEEEEEEEEEEEEEEEEE : %d \n", rectvector.size());
#endif  //  ALL
			pthread_mutex_unlock(&mutex_lock_rect);
			//printf("$$$$$$$$$$%d %d $$$$$$$$$$$$%d %d\n", rect_tgt.x, rect_tgt.y, rect_tgt_ratio.x, rect_tgt_ratio.y);
		}
		else
		{
			printf("Objects:\n\n");
			draw_detections(im_object_origin, dets, nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes);
			pthread_mutex_lock(&mutex_lock_display);
			copy_image_into(im_object_origin, im_object);
			pthread_mutex_unlock(&mutex_lock_display);	
		}
		free_detections(dets, nboxes);


		sec_cur = what_time_is_it_now();
		if ((sec_cur - sec_pre) > SEC_INTERVAL)
		{
			fps_detect = (double)(n_frm) / (sec_cur - sec_start);
			//printf("\033[2J");        printf("\033[1;1H");
			printf("\nFPS 4 detect : %.1f\n", fps_detect);
			sec_pre = sec_cur;
		}


		demo_index = (demo_index + 1) % demo_frame;
		//running = 0;
		printf("\n\n");
	}
	pthread_exit((void *) 0);
}


void *fetch_in_thread(void *ptr)
{
	// printf("fetch in thread in");

	struct_fetch *param_fetch = (struct_fetch *)ptr;
	cv::Rect rect_tgt_copy;
	std::vector <cv::Rect> rectvector_copy(80); 
	cv::Rect rect_tgt_ratio_copy;
	cv::Mat mask_rect_remap_ori = mask_rect_remap.clone(), mask_rect_remap_l = mask_rect_remap.clone(), 
		mask_rect_remap_r = mask_rect_remap.clone();

	bool is_there_detection;
	int id_img_orgin, n_frm = 0, status, l_r_or_all = param_fetch->left_right_or_full;
	struct_stereo *param_stereo = param_fetch->parameter_stereo;
	double fps_fetch, sec_cur, sec_pre, sec_start = what_time_is_it_now();
	sec_pre = sec_start; 

	image im_raw_copy=copy_image(im_raw), im_letter_origin=copy_image(im_letter);
	IplImage *ipl_l_origin = cvCreateImage(cvSize(ipl_l->width, ipl_l->height), ipl_l->depth, ipl_l->nChannels), *ipl_r_origin = cvCreateImage(cvSize(ipl_r->width, ipl_r->height), ipl_r->depth, ipl_r->nChannels);
#ifdef LOG
	int id_img_origin = 0;
#endif	//	LOG


	while (!demo_done)
	{    
		n_frm++;
		if(param_fetch->is_ocams)
		{
			status = fill_image_from_stream_ocam(cap_ocam, im_raw_copy, *(param_fetch->srcImg), param_fetch->image_size, &(ipl_l_origin), &(ipl_r_origin), l_r_or_all, param_fetch->gray_4_stereo);
		}
		else
		{
			status = fill_image_from_stream_ocv(cap_ocv, im_raw_copy, &(ipl_l_origin), &(ipl_r_origin), l_r_or_all, param_fetch->gray_4_stereo);
		}
		letterbox_image_into(im_raw_copy, net->w, net->h, im_letter_origin);
		pthread_mutex_lock(&mutex_lock_image);
		copy_image_into(im_raw_copy, im_raw);
		copy_image_into(im_letter_origin, im_letter);
#ifdef LOG
		g_id_img = id_img_origin;
#endif	//	LOG
		
        if (is_measure)
        {
            cvCopy(ipl_l_origin, ipl_l);
            cvCopy(ipl_r_origin, ipl_r);
        }
        pthread_mutex_unlock(&mutex_lock_image);

#ifdef LOG
		Logger(g_fn_log, "Captured frame", id_img_origin);
		id_img_origin++;
#endif	//	LOG


		if(compute_depth && param_stereo) 
		{
            if(is_measure)
            {
                undistort_and_rectify(ipl_l_origin, ipl_r_origin, param_stereo);
            }
            pthread_mutex_lock(&mutex_lock_ipl);
			cvCopy(ipl_l_origin, ipl_l);
			cvCopy(ipl_r_origin, ipl_r);
			pthread_mutex_unlock(&mutex_lock_ipl);

#ifndef ALL
			pthread_mutex_lock(&mutex_lock_rect);
			rect_tgt_copy.x = rect_tgt.x;    rect_tgt_copy.y = rect_tgt.y;    
			rect_tgt_copy.width = rect_tgt.width;    rect_tgt_copy.height = rect_tgt.height;
			rect_tgt_ratio_copy.x = rect_tgt_ratio.x; rect_tgt_ratio_copy.y = rect_tgt_ratio.y;
			rect_tgt_ratio_copy.width = rect_tgt_ratio.width; rect_tgt_ratio_copy.height = rect_tgt_ratio.height;
#else   //  ALL
			rectvector_copy.assign(rectvector.begin(), rectvector.end());
#endif  //  ALL
			pthread_mutex_unlock(&mutex_lock_rect);
			mask_rect_remap_ori.setTo(cv::Scalar::all(0));
			
			is_there_detection = rect_tgt_copy.width > 0 && rect_tgt_copy.height > 0;
			std::vector<std::vector<cv::Point> > contour_remap_ori;
			std::vector<std::vector<cv::Point> > contour_remap_ori_ratio;
#ifdef ALL
			is_there_detection = true;
#endif  //  ALL  
			//std::vector <double> mask_rect_remap_vector(80);
			if (is_there_detection) 
			{ 
				//compute_mask_rect_remap(mask_rect_remap_ori, mask_rect_remap_l, mask_rect_remap_r, rect_tgt_copy, param_stereo, true);
#ifdef ALL
				
				for(int i=0; i< rectvector.size(); i++){
					compute_mask_rect_remap(mask_rect_remap_ori, mask_rect_remap_l, mask_rect_remap_r, rectvector[i], param_stereo, true);
					//mask_rect_remap_vector.assign(mask_rect_remap_ori.datastart, mask_rect_remap_ori.dataend);
					mask_rect_remap_vector.push_back(mask_rect_remap_ori);
				}

#else   //  ALL

				compute_mask_rect_remap(mask_rect_remap_ori, mask_rect_remap_l, mask_rect_remap_r, rect_tgt_ratio_copy, param_stereo, true);
				
//compute_mask_rect_remap(mask_rect_remap_ori, mask_rect_remap_l, mask_rect_remap_r, rect_tgt_copy, param_stereo, true);
				cv::findContours(mask_rect_remap_ori, contour_remap_ori, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
				cv::findContours(mask_rect_remap_ori, contour_remap_ori_ratio, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
#endif  //  ALL
			}           
			pthread_mutex_lock(&mutex_lock_mask_remap);
			mask_rect_remap_ori.copyTo(mask_rect_remap);
			pthread_mutex_unlock(&mutex_lock_mask_remap);
#ifndef ALL
			pthread_mutex_lock(&mutex_lock_contour);
			contour_remap = contour_remap_ori;
			contour_remap_ratio = contour_remap_ori_ratio;
			pthread_mutex_unlock(&mutex_lock_contour);
#endif  //  ALL
		} 

		if(status == 0) demo_done = 1;
		sec_cur = what_time_is_it_now();
		if ((sec_cur - sec_pre) > SEC_INTERVAL) 
		{   
			fps_fetch = (double)(n_frm) / (sec_cur - sec_start);
			//printf("\033[2J");        printf("\033[1;1H");
			printf("\nFPS 4 fetch : %.1f\n", fps_fetch);
			//printf("Objects:\n\n");
			sec_pre = sec_cur; 
		}
	}
	pthread_exit((void *) 0);
}



void *fetch_in_thread_measure(void *ptr)
{
	// printf("fetch in thread in");

	struct_fetch *param_fetch = (struct_fetch *)ptr;
	cv::Rect rect_tgt_copy ;
	cv::Mat mask_rect_remap_ori = mask_rect_remap.clone(), mask_rect_remap_l = mask_rect_remap.clone(), 
		mask_rect_remap_r = mask_rect_remap.clone();
	bool is_there_detection;
	int id_img_orgin, n_frm = 0, status, l_r_or_all = param_fetch->left_right_or_full;
	struct_stereo *param_stereo = param_fetch->parameter_stereo;
	double fps_fetch, sec_cur, sec_pre, sec_start = what_time_is_it_now();//demo_time_detect = what_time_is_it_now();
	sec_pre = sec_start; 

	//image im_detect_copy;
	//IplImage *ipl_detect_l_copy, *ipl_detect_r_copy;
	image im_raw_copy=copy_image(im_raw), im_letter_origin=copy_image(im_letter);
	IplImage *ipl_l_origin=cvCreateImage(cvSize(ipl_l->width, ipl_l->height),ipl_l->depth,ipl_l->nChannels), *ipl_r_origin=cvCreateImage(cvSize(ipl_r->width, ipl_r->height), ipl_r->depth, ipl_r->nChannels);

#ifdef LOG
	int id_img_origin = 0;
#endif //LOG
	while (!demo_done)
	{    
		//printf(" fetch in thread while in");
		n_frm++;
		if(param_fetch->is_ocams)

			/*
			   pthread_mutex_lock(&mutex_lock_image);
			   im_detect_copy = im_detect;
			   ipl_detect_l_copy=cvCloneImage(ipl_detect_l);
			   ipl_detect_r_copy=cvCloneImage(ipl_detect_r);
			   pthread_mutex_unlock(&mutex_lock_image);
			   */
			//if(param_fetch->is_ocams)
		{

			status = fill_image_from_stream_ocam(cap_ocam, im_raw_copy, *(param_fetch->srcImg), param_fetch->image_size, &(ipl_l_origin), &(ipl_r_origin), l_r_or_all, param_fetch->gray_4_stereo);
			//status = fill_image_from_stream_ocam(cap_ocam, im_detect, *(param_fetch->srcImg), param_fetch->image_size, &(ipl_l_buff), &(ipl_r_buff), l_r_or_all, param_fetch->gray_4_stereo);

		}
		else
		{
			status = fill_image_from_stream_ocv(cap_ocv, im_raw_copy, &(ipl_l_origin), &(ipl_r_origin), l_r_or_all, param_fetch->gray_4_stereo);

			//status = fill_image_from_stream_ocv(cap_ocv, im_detect, &(ipl_l_buff), &(ipl_r_buff), l_r_or_all, param_fetch->gray_4_stereo);

		}
		letterbox_image_into(im_raw_copy, net->w, net->h, im_letter_origin);
		pthread_mutex_lock(&mutex_lock_image);
		copy_image_into(im_raw_copy, im_raw);
		copy_image_into(im_letter_origin, im_letter);
#ifdef LOG

		g_id_img = id_img_origin;
#endif  //  LOG
		cvCopy(ipl_l_origin, ipl_l);		cvCopy(ipl_r_origin, ipl_r);
		pthread_mutex_unlock(&mutex_lock_image);

#ifdef LOG
		//Logger(g_fn_log, "Get Frame ", id_img_origin)
		Logger(g_fn_log, "Captured frame ", id_img_origin, "");
		id_img_origin++;
#endif //LOG

		if(compute_depth && param_stereo) 
		{
			//cvShowImage("before undistort_and_rectify", ipl_l_buff[buff_index]);
			//undistort_and_rectify(ipl_l_origin, ipl_r_origin, param_stereo);
			pthread_mutex_lock(&mutex_lock_ipl);
			cvCopy(ipl_l_origin, ipl_l);		
			cvCopy(ipl_r_origin, ipl_r);
			pthread_mutex_unlock(&mutex_lock_ipl);

			pthread_mutex_lock(&mutex_lock_rect);
			rect_tgt_copy.x = rect_tgt.x;    rect_tgt_copy.y = rect_tgt.y;    rect_tgt_copy.width = rect_tgt.width;    rect_tgt_copy.height = rect_tgt.height;
			pthread_mutex_unlock(&mutex_lock_rect);
			mask_rect_remap_ori.setTo(cv::Scalar::all(0));
			is_there_detection = rect_tgt_copy.width > 0 && rect_tgt_copy.height > 0;
			std::vector<std::vector<cv::Point> > contour_remap_ori;
			std::vector<std::vector<cv::Point> > contour_remap_ori_ratio;
			if (is_there_detection) 
			{ 
			   compute_mask_rect_remap(mask_rect_remap_ori, mask_rect_remap_l, mask_rect_remap_r, rect_tgt_copy, param_stereo, true); 
			   cv::findContours(mask_rect_remap_ori, contour_remap_ori, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
				cv::findContours(mask_rect_remap_ori, contour_remap_ori_ratio, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
			}         
			pthread_mutex_lock(&mutex_lock_mask_remap);
			mask_rect_remap_ori.copyTo(mask_rect_remap);
			pthread_mutex_unlock(&mutex_lock_mask_remap);

			pthread_mutex_lock(&mutex_lock_contour);
			contour_remap = contour_remap_ori;
			contour_remap_ratio = contour_remap_ori_ratio;
			pthread_mutex_unlock(&mutex_lock_contour);



		} 
		if(status == 0) demo_done = 1;
		sec_cur = what_time_is_it_now();
		if ((sec_cur - sec_pre) > SEC_INTERVAL) 
		{   
			fps_fetch = (double)(n_frm) / (sec_cur - sec_start);
			//printf("\033[2J");        printf("\033[1;1H");
			printf("\nFPS 4 fetch : %.1f\n", fps_fetch);
			//printf("Objects:\n\n");
			sec_pre = sec_cur; 
		}
		//buff_index = (buff_index + 1) %3;
		//buff_index = (buff_index + 1) % SIZE_BUFFER;
	}
	pthread_exit((void *) 0);
}




void *display_in_thread(void *ptr)
{
	struct_display *param_display = (struct_display *)ptr;
	cv::Mat /*mask_rect_remap_copy,*/ mat_dis_copy;
	std::vector<std::vector<cv::Point> > contour_remap_copy;
	std::vector<std::vector<cv::Point> > contour_remap_ratio_copy;
    bool show_depth = false;
	char text[100];
	CvPoint xy_tgt;
	int fontThickness = 1, fontFace = cv::FONT_HERSHEY_DUPLEX, depth_millimeter, n_frm = 0;
	double /*fontScale = FONT_SCALE_CAM,*/ fps_display, sec_cur, sec_pre, sec_start = what_time_is_it_now();//demo_time_detect = what_time_is_it_now();
	CvFont font_obj, font_no_obj;
	cvInitFont(&font_obj, CV_FONT_HERSHEY_DUPLEX, FONT_SCALE_CAM_OBJECT, FONT_SCALE_CAM_OBJECT, 0, fontThickness);
	cvInitFont(&font_no_obj, CV_FONT_HERSHEY_DUPLEX, FONT_SCALE_CAM_NO_OBJ, FONT_SCALE_CAM_NO_OBJ, 0, fontThickness);
	sec_pre = sec_start; 

	image im_object_copy=copy_image(im_object); 
	image im_letter_copy; 
	IplImage *ipl_l_copy=cvCreateImage(cvSize(ipl_l->width, ipl_l->height), ipl_l->depth, ipl_l->nChannels), *ipl_r_copy=cvCreateImage(cvSize(ipl_r->width, ipl_r->height), ipl_r->depth, ipl_r->nChannels);
#ifdef TRIGGER
	int id_img_copy = -1;	
#endif	//	TRIGGER
	while (!demo_done)
	{     
		n_frm++;
		pthread_mutex_lock(&mutex_lock_image);
		copy_image_into(im_object, im_object_copy);
		cvCopy(ipl_l, ipl_l_copy);
		cvCopy(ipl_r, ipl_r_copy);
#ifdef TRIGGER
		id_img_copy = g_id_img;
#endif	//	TRIGGER
		pthread_mutex_unlock(&mutex_lock_image);
		if(!compute_depth)
		{
			show_image_cv(im_object_copy, "Demo", ipl);
		}
		else
		{
			pthread_mutex_lock(&mutex_lock_millimeter);
			xy_tgt.x = x_y_millimeter.x;    xy_tgt.y = x_y_millimeter.y;    depth_millimeter = x_y_millimeter.z;
	
			pthread_mutex_unlock(&mutex_lock_millimeter);

			pthread_mutex_lock(&mutex_lock_disp);
			mat_disp.copyTo(mat_dis_copy);
			pthread_mutex_unlock(&mutex_lock_disp);

			pthread_mutex_lock(&mutex_lock_contour);
			contour_remap_copy = contour_remap;
			contour_remap_ratio_copy = contour_remap_ratio;
			pthread_mutex_unlock(&mutex_lock_contour);
#ifdef ALL
			for( int i=0; i<rectvector.size(); i++)
			{
				//printf("****************** %d \n", i);
				rectangle(mat_dis_copy,rectvector[i], cv::Scalar(255, 255, 255), 1);
                CvPoint pUL(rectvector[i].x, rectvector[i].y);
                CvPoint pLR(rectvector[i].x + rectvector[i].width - 1, rectvector[i].y + rectvector[i].height - 1);
                cvRectangle(ipl_l_copy, pUL, pLR, cv::Scalar(255, 255, 255), 1);

				if ( i< x_y_disp_quant_vector_copy.size() ) {
                    CvPoint pDepth(x_y_disp_quant_vector_copy[i].x, x_y_disp_quant_vector_copy[i].y); 
					draw_depth_in_millimeter(mat_dis_copy, g_idx_coi == g_idx_person ? "person " : param_display->class_of_interest, pDepth.x, pDepth.y, millimeter_quant_vector_copy[i], CV_RGB(255, 255, 255), fontFace, FONT_SCALE_CAM_OBJECT, fontThickness); 
                    draw_depth_in_millimeter(ipl_l_copy, g_idx_coi == g_idx_person ? "person" : param_display->class_of_interest, pDepth, millimeter_quant_vector_copy[i], CV_RGB(255, 0, 0), font_obj);
				}
			}
#else   //  ALL
			if (!contour_remap_copy.empty())
			{
				cv::drawContours(mat_dis_copy, contour_remap_ratio_copy, 0, CV_RGB(255, 255, 255), 1);
				draw_contour_on_iplImage_from_vector_of_point(ipl_l_copy, contour_remap_ratio_copy[0], CV_RGB(128, 128, 128));

			}
			if (depth_millimeter > 0)
			{
				draw_depth_in_millimeter(ipl_l_copy, g_idx_coi == g_idx_person ? g_name_person : param_display->class_of_interest, xy_tgt, depth_millimeter, CV_RGB(255, 0, 0), font_obj);
				draw_depth_in_millimeter(mat_dis_copy, g_idx_coi == g_idx_person ? g_name_person : param_display->class_of_interest, xy_tgt.x, xy_tgt.y, depth_millimeter, CV_RGB(255, 255, 255), fontFace, FONT_SCALE_CAM_OBJECT, fontThickness); 
			}
			else
			{
              
                draw_no_coi(ipl_l_copy, g_idx_coi == g_idx_person ? g_name_person :  param_display->class_of_interest, "I don't see any ", CV_RGB(255, 255, 255), font_no_obj);
				draw_no_coi(mat_dis_copy, g_idx_coi == g_idx_person ? g_name_person : param_display->class_of_interest, "I don't see any ", CV_RGB(255, 255, 255), fontFace, FONT_SCALE_CAM_NO_OBJ, fontThickness);
               
            }

#endif //ALL

            if (is_measure)
            {
                cvShowImage("LEFT",ipl_l_copy);
                cvShowImage("RIGHT", ipl_r_copy);
            }

            if(show_depth)
            {
                //printf("mat_dis_copy !!\n");
                cvShowImage("rectified left", ipl_l_copy);  
                cv::imshow("mat_dis_copy", mat_dis_copy);
            }
            else
            {
                //printf("rectified left !!\n");
                cvShowImage("rectified left", ipl_l_copy);  
                cv::imshow("mat_dis_copy", mat_dis_copy);
                //cvShowImage("rectified right", ipl_r_copy);
            }

        }
        int c = cvWaitKey(1);
        if (c != -1) c = c % 256;
        //printf("c is %d\n", c);
        //printf("g_idx_person : %d, g_idx_coi : %d\n", g_idx_person, g_idx_coi); //   exit(0);
        if (c == 27)                //  ESC key
        {
            demo_done = 1;
            return 0;
        }

        else if( c == 97 )          //  'a' key
        {
            func_capture(param_display->dir_img, starting, interval);
        }

        //else if (9 == c)            // 'Tab' key
        //{
        //    cvDestroyAllWindows();
        //    show_depth = !show_depth;
        //}
        else if (9 == c)            // 'Tab' key
        {
            //int idx_person = get_index_of_class("person", demo_names, demo_classes);
            //printf("idx_person : %d, g_idx_coi : %d\n", idx_person, g_idx_coi);    exit(0);
            g_idx_coi = g_idx_person != g_idx_coi ? g_idx_person : get_index_of_class(param_display->class_of_interest, demo_names, demo_classes);
        }
       else if(56 == c || 50 == c || 101 == c)              //  56-up, 50-down, e-default. exposure
        {
            //int brightness = cap_ocam->get_control("Gain");
            //int exposure = cap_ocam->get_control("Exposure (Absolute)");
            int exposure_b4 = g_exposure;
            g_exposure = 56 == c ? MIN(MAX_EXPOSURE, g_exposure + INT_EXPOSURE) : (50 == c ? MAX(MIN_EXPOSURE, g_exposure - INT_EXPOSURE) : DEFAULT_EXPOSURE);
            if(g_exposure != exposure_b4)
            {
                cap_ocam->set_control("Exposure (Absolute)", g_exposure);   
                g_exposure = cap_ocam->get_control("Exposure (Absolute)");   
                if(g_exposure != exposure_b4)
                {
                    printf("Exposure has been changed from %d to %d\n", exposure_b4, g_exposure);
                }
                else
                {
                    printf("The exposure value %d was NOT changed by set_control function\n", exposure_b4);
                }
            }
            else
            {
                printf("The requested exposure value %d is the same as the existing value\n", g_exposure);
            }
        }
        else if(54 == c || 52 == c || 98 == c)          //  54-right, 52-left, b-default. brightness
        {
            int brightness_b4 = g_brightness;
            g_brightness = 54 == c ?  MIN(MAX_BRIGHTNESS, g_brightness + INT_BRIGHTNESS) :(52 == c ? MAX(MIN_BRIGHTNESS, g_brightness - INT_BRIGHTNESS) : DEFAULT_BRIGHTNESS);
            if(g_brightness != brightness_b4)
            {
                cap_ocam->set_control("Gain", g_brightness); 
                g_brightness = cap_ocam->get_control("Gain");
                if(g_brightness != brightness_b4)
                {
                    printf("Brightness has been changed from %d to %d\n", brightness_b4, g_brightness);
                }
                else
                {
                    printf("The brightness value %d was NOT changed by set_control function\n", brightness_b4);
                }
            }
            else
            {
                printf("The requested brightness value %d is the same as the existing value\n", g_brightness);
            }
        }

#ifdef TRIGGER
        else if (c == 8)			
        {
            if(id_img_copy >= 0)
            {
                for(int i = 0; i < 20; i++) printf("Detection test is just started for %s at %d\n", param_display->class_of_interest, id_img_copy);
                Logger(g_fn_log, "Detection test is just started for", id_img_copy, param_display->class_of_interest);
            }
        }
#endif  //  TRIGGER
        else if (c == 32)           //  SPACE key 
        {
            cvDestroyAllWindows();
            compute_depth = !compute_depth;
            if(compute_depth)
            {
                pthread_cond_signal(&cond_depth);
            }
        }
        else if (c == 82)           //  'R' key
        {
            demo_thresh += .02;
        } 
		else if (c == 84)           //  'T' key
		{
			demo_thresh -= .02;
			if(demo_thresh <= .02) demo_thresh = .02;
		} 
		else if (c == 83)           //  'S' key
		{
			demo_hier += .02;
		} 
		else if (c == 81)           //  'Q' key
		{
			demo_hier -= .02;
			if(demo_hier <= .0) demo_hier = .0;
		}

		sec_cur = what_time_is_it_now();
		if ((sec_cur - sec_pre) > SEC_INTERVAL) 
		{   
			fps_display = (double)(n_frm) / (sec_cur - sec_start);
			//printf("\033[2J");        printf("\033[1;1H");
			printf("\nFPS 4 display : %.1f\n", fps_display);
			//printf("Objects:\n\n");
			sec_pre = sec_cur; 
		}
	}
	pthread_exit((void *) 0);
}

void *display_in_thread_measure(void *ptr)
{
	struct_display *param_display = (struct_display *)ptr;
	cv::Mat /*mask_rect_remap_copy,*/ mat_dis_copy;
	std::vector<std::vector<cv::Point> > contour_remap_copy;
	char text[100];
	CvPoint xy_tgt;
	int fontThickness = 1, fontFace = cv::FONT_HERSHEY_DUPLEX, depth_millimeter, n_frm = 0;
	double /*fontScale = FONT_SCALE_CAM,*/ fps_display, sec_cur, sec_pre, sec_start = what_time_is_it_now();//demo_time_detect = what_time_is_it_now();
	CvFont phont;
	cvInitFont(&phont, CV_FONT_HERSHEY_DUPLEX, FONT_SCALE_CAM_NO_OBJ, FONT_SCALE_CAM_NO_OBJ, 0, fontThickness);
	sec_pre = sec_start; 

	image im_object_copy=copy_image(im_object); 
	image im_letter_copy; 
	IplImage *ipl_l_copy=cvCreateImage(cvSize(ipl_l->width, ipl_l->height),ipl_l->depth,ipl_l->nChannels), *ipl_r_copy=cvCreateImage(cvSize(ipl_r->width, ipl_r->height), ipl_r->depth, ipl_r->nChannels);
#ifdef TRIGGER
	int id_img_copy = -1;
#endif //TRIGGER

	while (!demo_done)
	{     
		n_frm++;
		//show_image_cv(buff[(buff_index + 1)%3], "Demo", ipl);
		//if(!compute_depth)
		//	{


		pthread_mutex_lock(&mutex_lock_image);
		//im_detect_copy = im_detect;
		//copy_image_into(im_detect, im_detect_copy);
		copy_image_into(im_object, im_object_copy);
		//im_letter_copy = im_letter;
		cvCopy(ipl_l,ipl_l_copy);
		cvCopy(ipl_r,ipl_r_copy);
#ifdef TRIGGER
		id_img_copy = g_id_img;
#endif //TRIGGER
		pthread_mutex_unlock(&mutex_lock_image);
		if(!compute_depth)
		{
			show_image_cv(im_object_copy, "Demo", ipl);
		}
		else
		{ 
			//std::cout << "display bbb" << std::endl;
			pthread_mutex_lock(&mutex_lock_millimeter);
			xy_tgt.x = x_y_millimeter.x;    xy_tgt.y = x_y_millimeter.y;    depth_millimeter = x_y_millimeter.z;
			pthread_mutex_unlock(&mutex_lock_millimeter);
			//std::cout << "display ccc" << std::endl;
			///*
			pthread_mutex_lock(&mutex_lock_disp);
			mat_disp.copyTo(mat_dis_copy);
			pthread_mutex_unlock(&mutex_lock_disp);
			//std::cout << "display ddd" << std::endl;

			pthread_mutex_lock(&mutex_lock_contour);
			contour_remap_copy = contour_remap;
			pthread_mutex_unlock(&mutex_lock_contour);
			//std::cout << "display eee" << std::endl;
			if (!contour_remap_copy.empty())
			{
				cv::drawContours(mat_dis_copy, contour_remap_copy, 0, CV_RGB(255, 255, 255), 1);

				//std::cout << "display fff" << std::endl;
				//*/

				//pthread_mutex_lock(&mutex_lock_image);
				//im_detect_copy = im_detect;
				//ipl_detect_l_copy = cvCloneImage(ipl_detect_l);
				//ipl_detect_r_copy = cvCloneImage(ipl_detect_r);
				//cvCopy(ipl_detect_l, ipl_detect_l_copy);
				//cvCopy(ipl_detect_r, ipl_detect_r_copy);
				//pthread_mutex_unlock(&mutex_lock_image);

				draw_contour_on_iplImage_from_vector_of_point(ipl_l_copy, contour_remap_copy[0], CV_RGB(255,255,255));
				//draw_contour_on_iplImage_from_vector_of_point(ipl_l_buff, contour_remap_copy[0], CV_RGB(255,255,255));
			}
			if (depth_millimeter > 0)
			{                
				draw_depth_in_millimeter(ipl_l_copy, g_idx_coi == g_idx_person ? g_name_person : param_display->class_of_interest /*"quant : "*/, xy_tgt, depth_millimeter, CV_RGB(255, 255, 255), phont);
				//draw_depth_in_millimeter(ipl_l_buff, param_display->class_of_interest /*"quant : "*/, xy_tgt, depth_millimeter, CV_RGB(255, 255, 255), phont);
				draw_depth_in_millimeter(mat_dis_copy, g_idx_coi == g_idx_person ? g_name_person : param_display->class_of_interest, xy_tgt.x, xy_tgt.y, depth_millimeter, CV_RGB(255, 255, 255), fontFace, FONT_SCALE_CAM_NO_OBJ, fontThickness); 

            }
			else
			{
				//draw_no_coi(ipl_l_copy, param_display->class_of_interest, "I don't see any ", CV_RGB(255, 255, 255), phont);
				//draw_no_coi(ipl_l_buff, param_display->class_of_interest, "I don't see any ", CV_RGB(255, 255, 255), phont);
				//draw_no_coi(mat_dis_copy, param_display->class_of_interest, "I don't see any ", CV_RGB(255, 255, 255), fontFace, fontScale, fontThickness);
			}
			//std::cout << "display ggg" << std::endl;
			//	cvShowImage("rc left", ipl_l); cvShowImage("rc right", ipl_r);
			cvShowImage("rectified left", ipl_l_copy);  cvShowImage("rectified right", ipl_r_copy);
			//cvShowImage("rectified left", ipl_l_buff);  cvShowImage("rectified right", ipl_r_buff);
			//std::cout << "display hhh" << std::endl;
			//std::cout << "mat_dis_copy.size() : " << mat_dis_copy.size() << std::endl;
			//cv::imshow("mat_dis_copy", mat_dis_copy);
			//std::cout << "display iii" << std::endl;
		}


		int c = cvWaitKey(1);
		if (c != -1) c = c % 256;
		if (c == 27)                //  ESC key
		{
			demo_done = 1;
			return 0;
		}	 
		else if (c == 32)           //  SPACE key 
		{
			//param_display->show_all_detection = !param_display->show_all_detection;
			//param_display->comp_depth = !param_display->comp_depth;
			cvDestroyAllWindows();
			compute_depth = !compute_depth;
			if(compute_depth)
			{
				pthread_cond_signal(&cond_depth);
			}
		}
		else if (c == 9)			// 'Tab' key	
		{
			//printf("tab key\n");	exit(0);
			func_capture(param_display->dir_img, starting, interval);
		}
		else if (c == 82)           //  'R' key
		{
			demo_thresh += .02;
		}	 
		else if (c == 84)           //  'T' key
		{
			demo_thresh -= .02;
			if(demo_thresh <= .02) demo_thresh = .02;
		} 
		else if (c == 83)           //  'S' key
		{
			demo_hier += .02;
		} 
		else if (c == 81)           //  'Q' key
		{
			demo_hier -= .02;
			if(demo_hier <= .0) demo_hier = .0;
		}

		sec_cur = what_time_is_it_now();
		if ((sec_cur - sec_pre) > SEC_INTERVAL) 
		{   
			fps_display = (double)(n_frm) / (sec_cur - sec_start);
			//printf("\033[2J");        printf("\033[1;1H");
			printf("\nFPS 4 display : %.1f\n", fps_display);
			//printf("Objects:\n\n");
			sec_pre = sec_cur; 
		}
	}
	pthread_exit((void *) 0);
	//return 0;
}

bool is_this_file(const char *path)
	//bool is_this_file(std::string& path)
{
	struct stat s;
	if(stat(path, &s) == 0 )
	{
		if( s.st_mode & S_IFDIR )
		{
			//it's a directory
			//return false;
		}
		else if( s.st_mode & S_IFREG )
		{
			//it's a file
			return true;
		}
		else
		{
			//something else
			//return false;
		}
	}
	else
	{
		perror("path does not exist");
	}
	return false;
}

/*
//------------ c++ counterpart of python function "join" --------------  
#include <experimental/filesystem>
std::string python_join_equivalent(const std::string& dir_to_file, const std::string& filename)
{
std::experimental::filesystem::path dir(dir_to_file);
std::experimental::filesystem::path fn(filename);    
std::experimental::filesystem::path full_path = dir / fn;
return full_path.u8string();                                                                               
}
*/

std::string python_join_equivalent(const std::string& dir_to_file, const std::string& filename)
{
	return dir_to_file + "/" + filename;
}



void get_left_and_right_image(cv::Mat& mat_l_raw, cv::Mat& mat_r_raw, const cv::Mat& mat_full_raw, int left_right_or_full) 
{
	if(left_right_or_full)
	{
		int w = mat_full_raw.cols, h = mat_full_raw.rows;
		mat_full_raw(cv::Rect(0, 0, w / 2, h)).copyTo(mat_l_raw);
		mat_full_raw(cv::Rect(w / 2, 0, w / 2, h)).copyTo(mat_r_raw);
	}
	return;
}




bool detect_class_of_interest(cv::Rect& rect_target, char *class_of_interest, image& im, network *net, float thresh, float hier_thresh, float nms, int idx_coi, float ratio_shrink)
{
	image sized = letterbox_image(im, net->w, net->h);
	layer l = net->layers[net->n - 1];
	float *X = sized.data;
	//time=what_time_is_it_now();
	network_predict(net, X);
	//printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
	int nboxes = 0;
	detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
	if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
	bool is_coi_detected = get_rect_of_object_of_interest(rect_target, im.w, im.h, dets, nboxes, thresh, idx_coi, ratio_shrink); 
	//draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
	free_detections(dets, nboxes);
	return is_coi_detected;
}

cv::Size get_image_size_of_folder(struct dirent **namelist, int n_frm, std::string& str_dir, std::string& str_ext, int l_r_or_all)
{
	cv::Size saiz;
	std::string str_path;
	std::cout << "l_r_or_all : " << l_r_or_all << std::endl;
	std::cout << "n_frm : " << n_frm << std::endl;
	for(int iF = 0; iF < n_frm; iF++)
	{
		std::string str_file(namelist[iF]->d_name);
		str_path = python_join_equivalent(str_dir, str_file);
		if(!is_this_file(str_path.c_str())) continue;
		if(str_file.substr(str_file.find_last_of(".") + 1) != str_ext) continue;

		//printf("%s\n", str_path);
		std::cout << "str_path : " << str_path << std::endl;
		cv::Mat mat_full_raw = cv::imread(str_path);
		if (mat_full_raw.empty()) continue;
		if(!l_r_or_all)
		{
			saiz = mat_full_raw.size();
		}
		else
		{
			saiz.width = mat_full_raw.cols / 2;
			saiz.height = mat_full_raw.rows;
		}
		break;
	}

	std::cout << "saiz : " << saiz << std::endl;
	return saiz;

}



bool save_depth_result(std::ofstream& ofs, std::string& fn_img, double millimeter, int disparity, cv::Rect& rect)
{
	ofs << fn_img << " " << millimeter << " " << disparity << " " << rect.x << " " << rect.y << " " << rect.width << " " << rect.height << std::endl;
	return true;         
}




void demo_folder(char *cfgfile, char *weightfile, float thresh, const char *folder, char *ext, char **names, int classes, char *prefix, float hier, int fullscreen, int left_right_or_full, int comp_depth, int both_disparity, int path_agg, int percent_closest, double alfa, bool gray_4_stereo, float ratio_shrink, char *class_of_interest, char *fn_res, char *yml_intrinsic, char *yml_extrinsic)
{

#ifdef LOG
	std::ofstream ofs(g_fn_log.c_str(), std::ios::out | std::ios::trunc); ofs.close();
#endif  //  LOG

	image **alphabet = load_alphabet();
	demo_names = names;
	demo_alphabet = alphabet;
	demo_classes = classes;
	demo_thresh = thresh;
	demo_hier = hier;
	printf("Demo folder\n");
	net = load_network(cfgfile, weightfile, 0);
	set_batch_network(net, 1);

	srand(2222222);
	struct dirent **namelist;
	bool isVerticalStereo, is_coi_detected;
	std::string text, str_path, str_dir(folder), str_ext(ext);
	std::cout << "folder : " << folder << std::endl;
	int fontThickness = 1, iF, n_frm = scandir(folder, &namelist, 0, alphasort), fontFace = cv::FONT_HERSHEY_DUPLEX;
    g_idx_coi = get_index_of_class(class_of_interest, demo_names, demo_classes);
	std::cout << "n_frm : " << n_frm << std::endl;
	if(n_frm < 0)
	{
		perror("scandir");   
		//fprintf(stderr, "%s 디렉토리 조회 오류: %s\n", folder, strerror(errno));
	}
	for (iF = 0; iF < n_frm; iF++) printf("%s\n", namelist[iF]->d_name);

	std::ofstream ofs_res(fn_res);

	std::cout << "left_right_or_full : " << left_right_or_full << std::endl;
	std::cout << "comp_depth : " << comp_depth << std::endl;

	printf("net->w : %d, net->h : %d\n", net->w, net->h);

	//demo_time = what_time_is_it_now();
	struct_depth param_depth;
	if(left_right_or_full && comp_depth)
	{
		param_depth.percent_closest = percent_closest;
		param_depth.wsize = 13; param_depth.max_disp = 160; param_depth.lambda = 10000.0;   param_depth.sigma = 1.0;
		param_depth.both_disparity = both_disparity;
		param_depth.path_agg = path_agg;
		if (8 == path_agg)
		{
			param_depth.p1 = 6; param_depth.p2 = 96;
		}
		else if(4 == path_agg)
		{
			param_depth.p1 = 7; param_depth.p2 = 86;
		}
		else
		{
			std::cout << "path_agg is " << path_agg << ", which is wrong !!" << std::endl;
			exit(0);
		}
		init_disparity_method(param_depth.p1, param_depth.p2);
	}
	std::cout << "demo aaa" << std::endl;
	std::cout << "comp_depth : " << comp_depth << std::endl;
	struct_stereo param_stereo;    
	if (left_right_or_full && comp_depth) 
	{
		param_depth.wls_filter = initDisparityWLSFilter(param_depth.wsize, param_depth.max_disp, param_depth.lambda, param_depth.sigma);

		//param_stereo.img_size = cv::Size(buff[0].w, buff[0].h);
		param_stereo.img_size = get_image_size_of_folder(namelist, n_frm, str_dir, str_ext, left_right_or_full);
		std::cout << "param_stereo.img_size : " << param_stereo.img_size << std::endl;
		param_stereo.fn_ext = yml_extrinsic;
		param_stereo.fn_int = yml_intrinsic;
		param_stereo.alfa = alfa;
		isVerticalStereo = load_stereo_calibration_info(&param_stereo);
		param_depth.parameter_stereo = &param_stereo;
		mask_rect_remap.create(param_stereo.img_size, CV_8UC1); 
	}
	else 
	{
		param_depth.parameter_stereo = NULL;
	}   
	//std::cout << "demo ccc" << std::endl;
	cv::Mat mat_mask_rect_remap_l(param_stereo.img_size, CV_8UC1), mat_mask_rect_remap_r(param_stereo.img_size, CV_8UC1), mat_mask_rect_remap_int(param_stereo.img_size, CV_8UC1), mat_dis, mat_dis_l, mat_dis_r, mat_full_raw, mat_l_remap_gray, mat_r_remap_gray, mat_l_remap, mat_r_remap, mat_l_raw, mat_r_raw;
	float nms = NMS, elaps_ms_l, elaps_ms_r, elaps_ms;
	double fontScale = FONT_SCALE_FOLDER, millimeter_quant = 0, millimeter_min, millimeter_max;
	//cv::Size fontSize;
	cv::Point3i x_y_disp_min(-1, -1, -1), x_y_disp_max(-1, -1, -1), x_y_disp_quant(-1, -1, -1);
	cv::Point xy_tgt;
	cv::Rect rect_target;
	for(iF = 0; iF < n_frm; iF++)
	{
		std::string str_file(namelist[iF]->d_name);
		str_path = python_join_equivalent(str_dir, str_file);
		if(!is_this_file(str_path.c_str())) continue;
		if(str_file.substr(str_file.find_last_of(".") + 1) != str_ext) continue;

		//printf("%s\n", str_path);
		std::cout << "str_path : " << str_path << std::endl;
		//mat_full_raw = cv::imread(str_path);
		image im = load_image_cv_2(mat_full_raw, str_path.c_str(), 3, left_right_or_full);
		if (mat_full_raw.empty()) continue;
		std::cout << "mat_full_raw.size() : " << mat_full_raw.size() << std::endl;
		std::cout << "left_right_or_full : " << left_right_or_full << std::endl;
		//cv::imshow("mat_full_raw_" + str_file, mat_full_raw);
		if(left_right_or_full)
		{
			get_left_and_right_image(mat_l_raw, mat_r_raw, mat_full_raw, left_right_or_full);
		}
		//std::cout << "mat_l_raw.size() : " << mat_l_raw.size() << std::endl;
		is_coi_detected = detect_class_of_interest(rect_target, class_of_interest, im, net, thresh, hier, nms, g_idx_coi, ratio_shrink);
		//std::cout << "is_coi_detected : " << is_coi_detected << std::endl;
		if(left_right_or_full && comp_depth)
		{
			mat_l_raw.copyTo(mat_l_remap);
			//std::cout << "mat_l_remap.size() : " << mat_l_remap.size() << std::endl;
			undistort_and_rectify(mat_l_remap, &param_stereo, true);
			//cv::imshow("mat_r_raw_" + str_file, mat_r_raw);
			mat_r_raw.copyTo(mat_r_remap);
			//cv::imshow("mat_r_remap_copied_" + str_file, mat_r_remap);
			undistort_and_rectify(mat_r_remap, &param_stereo, false);
			//cv::imshow("mat_r_remap_rectified_" + str_file, mat_r_remap);
			//cv::waitKey();
			cv::cvtColor(mat_l_remap, mat_l_remap_gray, CV_BGR2GRAY);
			cv::cvtColor(mat_r_remap, mat_r_remap_gray, CV_BGR2GRAY);


			mat_dis_l = compute_disparity_method(mat_l_remap_gray, mat_r_remap_gray, &elaps_ms_l, path_agg);            
			if(both_disparity) 
			{
				mat_dis_r = compute_disparity_method(mat_r_remap_gray, mat_l_remap_gray, &elaps_ms_r, path_agg);            
				post_disparity_filter(mat_dis, mat_dis_l, mat_dis_r, mat_l_remap, param_depth.wls_filter);
				elaps_ms = elaps_ms_r + elaps_ms_l;
			}
			else
			{
				mat_dis = mat_dis_l;
				elaps_ms = elaps_ms_l;
			}
			if (is_coi_detected)
			{
				//std::cout << "mat_mask_rect_remap.size() : " << mat_mask_rect_remap.size() << std::endl;
				compute_mask_rect_remap(mat_mask_rect_remap_int, mat_mask_rect_remap_l, mat_mask_rect_remap_r, rect_target, &param_stereo, true);
				x_y_disp_quant = get_quantile_disparity(mat_dis, mat_mask_rect_remap_int, percent_closest);
				millimeter_quant = disparity_2_millimeter(x_y_disp_quant.z, param_stereo.focal_in_pixel, param_stereo.baseline_in_millimeter, NULL);
				//std::cout << "x_y_disp_med : " << x_y_disp_med << std::endl;
				get_min_max_disparity(x_y_disp_min, x_y_disp_max, mat_dis, mat_mask_rect_remap_int);
				millimeter_min = disparity_2_millimeter(x_y_disp_min.z, param_stereo.focal_in_pixel, param_stereo.baseline_in_millimeter, NULL);
				millimeter_max = disparity_2_millimeter(x_y_disp_max.z, param_stereo.focal_in_pixel, param_stereo.baseline_in_millimeter, NULL);
				draw_depth_in_millimeter(mat_l_remap, class_of_interest /*"quant : "*/, x_y_disp_quant.x, x_y_disp_quant.y, /*x_y_disp_quant.z*/ millimeter_quant, CV_RGB(255, 0, 0), fontFace, fontScale, fontThickness);
				draw_depth_in_millimeter(mat_dis, class_of_interest /*"quant : "*/, x_y_disp_quant.x, x_y_disp_quant.y, /*x_y_disp_quant.z*/ millimeter_quant, CV_RGB(255, 255, 255), fontFace, fontScale, fontThickness);
				draw_depth_in_millimeter(mat_l_remap, "min", x_y_disp_min.x, x_y_disp_min.y, /*x_y_disp_min.z*/millimeter_min, CV_RGB(255, 0, 0), fontFace, fontScale, fontThickness);
				draw_depth_in_millimeter(mat_dis, "min", x_y_disp_min.x, x_y_disp_min.y, /*x_y_disp_min.z*/ millimeter_min, CV_RGB(255, 255, 255), fontFace, fontScale, fontThickness);

				draw_depth_in_millimeter(mat_l_remap, "max", x_y_disp_max.x, x_y_disp_max.y, /*x_y_disp_max.z*/millimeter_max, CV_RGB(255, 0, 0), fontFace, fontScale, fontThickness);
				draw_depth_in_millimeter(mat_dis, "max", x_y_disp_max.x, x_y_disp_max.y, /*x_y_disp_max.z*/ millimeter_max, CV_RGB(255, 255, 255), fontFace, fontScale, fontThickness);
				std::vector<std::vector<cv::Point> > contours;
				cv::findContours(mat_mask_rect_remap_l, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
				cv::drawContours(mat_l_remap, contours, 0, CV_RGB(255, 0, 0), 1);
				cv::drawContours(mat_dis, contours, 0, CV_RGB(255, 255, 255), 1);

				cv::findContours(mat_mask_rect_remap_r, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
				cv::drawContours(mat_r_remap, contours, 0, CV_RGB(255, 0, 0), 1);
				cv::drawContours(mat_dis, contours, 0, CV_RGB(128, 128, 128), 1);
			}
		}
		if(is_coi_detected)
		{
			cv::rectangle(mat_l_raw, rect_target, CV_RGB(255, 0, 0), 1, 8);
		}
		save_depth_result(ofs_res, str_file, millimeter_quant, x_y_disp_quant.z, rect_target);
		cv::imshow("mat_l_raw", mat_l_raw);
		cv::imshow("mat_r_raw", mat_r_raw);
		cv::imshow("mat_l_remap", mat_l_remap);
		cv::imshow("mat_r_remap", mat_r_remap);
		cv::imshow("mat_dis", mat_dis);
		cv::waitKey(30 * 1000);
		//cv::destroyAllWindows();
		free(namelist[iF]);
	}
	ofs_res.close();
	free(namelist);
}

int init_camera_or_video(int& w, int& h, int& frames, const char *filename, bool is_ocam, int cam_index)
{  
    printf("filename in demo : %s\n", filename); 
    printf("requested  width : %d, height : %d, fps : %d\n", is_ocam ? int(w / 2) : w, h, frames); 
    int image_size = -1; 
    if(filename)
    {
        printf("video file: %s\n", filename);
        cap_ocv = cvCaptureFromFile(filename);
        if(!cap_ocv) 
        { 
            error("Couldn't open the video file.\n");
            exit(0);
        }
        image_size = 1;
        w = (int)cvGetCaptureProperty(cap_ocv, CV_CAP_PROP_FRAME_WIDTH);
        h = (int)cvGetCaptureProperty(cap_ocv, CV_CAP_PROP_FRAME_HEIGHT);
        frames = (int)cvGetCaptureProperty(cap_ocv, CV_CAP_PROP_FPS);
    }
    else
    {
        if (is_ocam)
        {
            char path_cam[50];
            sprintf(path_cam, "/dev/video%d", cam_index);
            cap_ocam = new Withrobot::Camera(path_cam);
            cap_ocam->set_format(w > 0 ? int(w / 2) : 640, h > 0 ? h : 360, Withrobot::fourcc_to_pixformat('Y', 'U', 'Y', 'V'), 1, frames > 0 ? frames : 30);
            Withrobot::camera_format camFormat;
            cap_ocam->get_current_format(camFormat);
            std::string camName = cap_ocam->get_dev_name();
            std::string camSerialNumber = cap_ocam->get_serial_number();
            printf("dev: %s, serial number: %s\n", camName.c_str(), camSerialNumber.c_str());
            printf("----------------- Current format informations -----------------\n");
            camFormat.print();
            printf("---------------------------------------------------------------\n");
            //int brightness = cap_ocam->get_control("Gain");
            //int exposure = cap_ocam->get_control("Exposure (Absolute)");//
            cap_ocam->set_control("Gain", g_brightness);//
            //cap_ocam->set_control("Exposure (Absolute)", exposure);//
            cap_ocam->set_control("Exposure (Absolute)", g_exposure);//
            g_brightness = cap_ocam->get_control("Gain");
            g_exposure = cap_ocam->get_control("Exposure (Absolute)");//
            printf("Default brightness : %d, Default exposure (Absolute) : %d\n", DEFAULT_BRIGHTNESS, DEFAULT_EXPOSURE);
            printf("Brightness : %d, Exposure (Absolute) : %d\n", g_brightness, g_exposure);
            if (!cap_ocam->start()) {
                printf("Failed to start. \n");
                exit(0);
            }
            image_size = camFormat.image_size;
            w = camFormat.width;
            h = camFormat.height;
            frames = (int)camFormat.frame_rate;
        }
        else
        {
            cap_ocv = cvCaptureFromCAM(cam_index);
            if(!cap_ocv) 
            { 
                error("Couldn't connect to webcam.\n");
                exit(0);
            }
            if(w) cvSetCaptureProperty(cap_ocv, CV_CAP_PROP_FRAME_WIDTH, w);
            if(h) cvSetCaptureProperty(cap_ocv, CV_CAP_PROP_FRAME_HEIGHT, h);
            if(frames) cvSetCaptureProperty(cap_ocv, CV_CAP_PROP_FPS, frames);
            image_size = 1;
            w = (int)cvGetCaptureProperty(cap_ocv, CV_CAP_PROP_FRAME_WIDTH);
            h = (int)cvGetCaptureProperty(cap_ocv, CV_CAP_PROP_FRAME_HEIGHT);
            frames = (int)cvGetCaptureProperty(cap_ocv, CV_CAP_PROP_FPS);
        }
    }
    printf("image_size : %d\n", image_size);
    return image_size;
}

void demo_measure (char *cfgfile, char *weightfile, float thresh, int cam_index, const char *dir_img, char **names, int classes, char *prefix, int avg_frames, float hier, int w_cam, int h_cam, int frames, int fullscreen, int left_right_or_full, int comp_depth, int both_disparity, int path_agg, int percent_closest, double alfa, bool is_ocam, bool gray_4_stereo, bool is_disp_2_milli, bool show_all_detection, float ratio_shrink, char *class_of_interest, char *fn_poly, int s, int inter)
{
	
#ifdef LOG
	std::ofstream ofs(g_fn_log.c_str(), std::ios::out | std::ios::trunc); ofs.close();
#endif  //  LOG
	if(!mkdir_if_not_exist(dir_img))
	{
		exit(0);
	}
	starting = s;
	interval = inter;
	std::cout << "start : " << starting <<"mm, interval : "<< interval << " mm" << std::endl;
	std::cout << "demo_measure_left_right_or_full : " << left_right_or_full << std::endl;
	bool isVerticalStereo;
	//demo_frame = avg_frame;
	image **alphabet = load_alphabet();
	demo_names = names;
	demo_alphabet = alphabet;
	demo_classes = classes;
	demo_thresh = thresh;
	demo_hier = hier;
	printf("Demo\n");
	net = load_network(cfgfile, weightfile, 0);
	set_batch_network(net, 1);
	pthread_t detect_thread;
	pthread_t depth_thread;
	pthread_t fetch_thread;
	pthread_t display_thread;
	//printf("names: %s\n", demo_names);

	srand(2222222);

	int i, image_size;
	cv::Mat srcImg/*, dstImg[2]*/;
	demo_total = size_network(net);
	predictions = (float **)calloc(demo_frame, sizeof(float*));
	for (i = 0; i < demo_frame; ++i){
		predictions[i] = (float *)calloc(demo_total, sizeof(float));
	}
	avg = (float *)calloc(demo_total, sizeof(float));

	image_size = init_camera_or_video(w_cam, h_cam, frames, NULL, is_ocam, cam_index);

	//buff[0] = get_image_from_stream(cap);
	std::cout << "left_right_or_full : " << left_right_or_full << std::endl;
	std::cout << "comp_depth : " << comp_depth << std::endl;

	if (is_ocam)
	{
        printf("OCAM CAMERA VERSION\n");
		printf("srcImg w_cam : %d, h_cam : %d\n", w_cam, h_cam);
		srcImg.create(cv::Size(w_cam, h_cam), CV_8UC2);
		//buff[0] = get_image_from_stream_ocam(cap_ocam, left_right_or_full, comp_depth, gray_4_stereo, &(ipl_l_buff[0]), &(ipl_r_buff[0]));
		im_raw = get_image_from_stream_ocam(cap_ocam, srcImg, /*dstImg,*/ image_size, left_right_or_full, comp_depth, gray_4_stereo, &(ipl_l), &(ipl_r));
		// im_detect = get_image_from_stream_ocam(cap_ocam, srcImg, /*dstImg,*/ image_size, left_right_or_full, comp_depth, gray_4_stereo, &(ipl_l_buff), &(ipl_r_buff));
	}
	else
	{
        printf("ZED CAMERA VERSION\n");
		im_raw = get_image_from_stream_ocv(cap_ocv, left_right_or_full, comp_depth, gray_4_stereo, &(ipl_l), &(ipl_r));
		//im_detect = get_image_from_stream_ocv(cap_ocv, left_right_or_full, comp_depth, gray_4_stereo, &(ipl_l_buff), &(ipl_r_buff));

	}

	//std::cout << "ipl_l_buff[0] width, height : " << ipl_l_buff[0]->width << " " << ipl_l_buff[0]->height << std::endl;
	//std::cout << "ipl_r_buff[0] width, height : " << ipl_r_buff[0]->width << " " << ipl_r_buff[0]->height << std::endl;
	printf("net->w : %d, net->h : %d\n", net->w, net->h);

	im_letter = letterbox_image(im_raw, net->w, net->h);
	ipl = cvCreateImage(cvSize(im_raw.w, im_raw.h), IPL_DEPTH_8U, im_raw.c);
	//ipl_l_raw = cvCloneImage(ipl_l_buff[0]);        ipl_r_raw = cvCloneImage(ipl_r_buff[0]);     
	im_object = copy_image(im_raw);
	im_letter = letterbox_image(im_raw, net->w, net->h);
	if(left_right_or_full && comp_depth) 
	{ 
		ipl_l = cvCloneImage(ipl_l);    
		ipl_r = cvCloneImage(ipl_r);     
		//ipl_l_buff = cvCloneImage(ipl_l_buff);
		//ipl_r_buff = cvCloneImage(ipl_r_buff);
	}


	int count = 0;

	struct_display param_display;
	param_display.class_of_interest = class_of_interest;
	param_display.dir_img = dir_img;
	//demo_time = what_time_is_it_now();
	struct_detect param_detect;
	param_detect.class_of_interest = class_of_interest;
	param_detect.ratio_shrink = ratio_shrink;
	//struct_depth param_depth;
	//param_depth.parameter_poly = NULL;
	//if(left_right_or_full && comp_depth)
	//{
	//  compute_depth = true;
	//param_display.show_all_detection = show_all_detection;
	//param_detect.show_all_detection = show_all_detection;
	/* if(fn_poly)
	   {
	   param_depth.parameter_poly = load_polynomial_coefficient(fn_poly);
	   param_depth.parameter_poly->is_from_disparity_to_millimeter = is_disp_2_milli;
	   }
	   param_depth.percent_closest = percent_closest;
	   param_depth.wsize = 13; param_depth.max_disp = 160; param_depth.lambda = 10000.0;   param_depth.sigma = 1.0;
	   param_depth.both_disparity = both_disparity;
	   param_depth.path_agg = path_agg;
	   if (8 == path_agg)
	   {
	   param_depth.p1 = 6; param_depth.p2 = 96;
	   }
	   else if(4 == path_agg)
	   {
	   param_depth.p1 = 7; param_depth.p2 = 86;
	   }
	   else
	   {
	   std::cout << "path_agg is " << path_agg << ", which is wrong !!" << std::endl;
	   exit(0);
	   }*/
	//}
	//std::cout << "demo aaa" << std::endl;
	struct_fetch param_fetch;
	param_fetch.left_right_or_full = left_right_or_full;
	param_fetch.is_ocams = is_ocam;
	param_fetch.gray_4_stereo = gray_4_stereo;
	param_fetch.srcImg = is_ocam ? &srcImg : NULL;
	param_fetch.image_size = is_ocam ? image_size : -1; 

	//std::cout << "demo bbb" << std::endl;
	/*if (left_right_or_full && comp_depth) 
	  {
	  param_depth.wls_filter = initDisparityWLSFilter(param_depth.wsize, param_depth.max_disp, param_depth.lambda, param_depth.sigma);
	  */  
	struct_stereo param_stereo;    
	param_stereo.img_size = cv::Size(im_raw.w, im_raw.h);
	//param_stereo.fn_ext = yml_extrinsic;
	//    param_stereo.fn_int = yml_intrinsic;
	param_stereo.alfa = alfa;
	//  isVerticalStereo = load_stereo_calibration_info(&param_stereo);
	param_fetch.parameter_stereo = &param_stereo;
	//param_depth.parameter_stereo = &param_stereo;
	//mask_rect_remap.create(param_stereo.img_size, CV_8UC1); 
	mask_rect_remap = cv::Mat::zeros(param_stereo.img_size, CV_8UC1); 
	mat_disp = cv::Mat::zeros(param_stereo.img_size, CV_8UC1);
	// }
	/* else 
	   {
	   param_fetch.parameter_stereo = NULL;
	   param_depth.parameter_stereo = NULL;
	   }*/

	pthread_mutex_init(&mutex_lock_contour, NULL);
	pthread_mutex_init(&mutex_lock_disp, NULL);
	pthread_mutex_init(&mutex_lock_rect, NULL);
	pthread_mutex_init(&mutex_lock_mask_remap, NULL);
	pthread_mutex_init(&mutex_lock_millimeter, NULL);
	//pthread_mutex_init(&mutex_lock_log, NULL);
	pthread_mutex_init(&mutex_lock_image, NULL);



	std::cout << "demo ccc" << std::endl;
	//while(!demo_done){
	//buff_index = (buff_index + 1) %3;
	if(pthread_create(&fetch_thread, 0, fetch_in_thread, &param_fetch)) error("Thread creation failed");
	if(pthread_create(&detect_thread, 0, detect_in_thread, &param_detect)) error("Thread creation failed");
	/*if(left_right_or_full && comp_depth)
	  { 
	  if(pthread_create(&depth_thread, 0, depth_in_thread, &param_depth)) error("Thread creation failed");
	  }*/
	if(pthread_create(&display_thread, 0, display_in_thread, &param_display)) error("Thread creation failed");
	std::cout << "demo bbb" << std::endl;

	printf("prefix : %d\n", prefix);
	if(!prefix){
		printf("ppp\n");
		//fps = 1./(what_time_is_it_now() - demo_time);
		//demo_time = what_time_is_it_now();
		//display_in_thread(0);
	}
	else
	{
		printf("eee");
		char name[256];
		sprintf(name, "%s_%08d", prefix, count);
		save_image(im_raw, name);
	}

	int rc, status;
	//pthread_join(fetch_thread, 0);
	printf(" rc: %d \n", rc);
	//printf(" ipl_l.w : %d h : %d", ipl_l->width, ipl_l->height);
	//printf(" ipl_r.w : %d h : %d", ipl_r->width, ipl_r->height);
	//printf(" ipl_l_c.w : %d h : %d", ipl_l_copy->width, ipl_l_copy->height);
	//printf(" ipl_r_c.w : %d h : %d", ipl_r_copy->width, ipl_r_copy->height);

	rc = pthread_join(fetch_thread, (void **)&status);
	printf("  pthread\n");
	if (rc == 0)
	{
		printf("in\n");
		printf("Completed join with fetch_thread. status = %d\n", status);
	}
	else
	{
		printf("ERROR; return code from pthread_join() is %d, fetch_thread\n", rc);
		return;
	}

	//pthread_join(detect_thread, 0);

	rc = pthread_join(detect_thread, (void **)&status);
	if (rc == 0)
	{
		printf("Completed join with detect_thread. status = %d\n", status);
	}
	else
	{
		printf("ERROR; return code from pthread_join() is %d, detect_thread\n", rc);
		return;
	}


	rc = pthread_join(display_thread, (void **)&status);
	if (rc == 0)
	{
		printf("Completed join with display_thread. status = %d\n", status);
	}
	else
	{
		printf("ERROR; return code from pthread_join() is %d, display_thread\n", rc);
		return;
	}




	/*
	   if(comp_depth) 
	   {
	//pthread_join(depth_thread, 0);
	rc = pthread_join(depth_thread, (void **)&status);
	if (rc == 0)
	{
	printf("Completed join with depth_thread. status = %d\n", status);
	}
	else
	{
	printf("ERROR; return code from pthread_join() is %d, depth_thread\n", rc);
	return;
	}
	}*/
	++count;
	//}
}

void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, char *prefix, int avg_frames, float hier, int w_cam, int h_cam, int frames, int fullscreen, int left_right_or_full, int comp_depth, int both_disparity, int path_agg, int percent_closest, double alfa, bool is_ocam, bool gray_4_stereo, bool is_disp_2_milli, bool show_all_detection, float ratio_shrink, char *class_of_interest, char *fn_poly, char *yml_intrinsic, char *yml_extrinsic, bool is_measure)
{

#ifdef LOG
	std::ofstream ofs(g_fn_log.c_str(), std::ios::out | std::ios::trunc); ofs.close();
#endif  //  LOG
    is_measure = is_measure;
	std::cout << "left_right_or_full : " << left_right_or_full << std::endl;
	bool isVerticalStereo;
	//demo_frame = avg_frames;
	image **alphabet = load_alphabet();
	demo_names = names;
	demo_alphabet = alphabet;
	demo_classes = classes;
	demo_thresh = thresh;
	demo_hier = hier;
    g_idx_person = get_index_of_class("person", demo_names, demo_classes);
	printf("Demo\n");
	net = load_network(cfgfile, weightfile, 0);
	set_batch_network(net, 1);
	pthread_t detect_thread;
	pthread_t depth_thread;
	pthread_t fetch_thread;
	pthread_t display_thread;

	srand(2222222);

	int i, image_size;
	cv::Mat srcImg/*, dstImg[2]*/;
	demo_total = size_network(net);
	predictions = (float **)calloc(demo_frame, sizeof(float*));
	for (i = 0; i < demo_frame; ++i){
		predictions[i] = (float *)calloc(demo_total, sizeof(float));
	}
	avg = (float *)calloc(demo_total, sizeof(float));

	image_size = init_camera_or_video(w_cam, h_cam, frames, filename, is_ocam, cam_index);

	std::cout << "left_right_or_full : " << left_right_or_full << std::endl;
	std::cout << "comp_depth : " << comp_depth << std::endl;

	if (is_ocam)
	{
		printf("srcImg w_cam : %d, h_cam : %d\n", w_cam, h_cam);
		srcImg.create(cv::Size(w_cam, h_cam), CV_8UC2);
		im_raw = get_image_from_stream_ocam(cap_ocam, srcImg, /*dstImg,*/ image_size, left_right_or_full, comp_depth, gray_4_stereo, &(ipl_l), &(ipl_r));
	}
	else
	{
		im_raw = get_image_from_stream_ocv(cap_ocv, left_right_or_full, comp_depth, gray_4_stereo, &(ipl_l), &(ipl_r));

	}

	printf("net->w : %d, net->h : %d\n", net->w, net->h);

	im_letter = letterbox_image(im_raw, net->w, net->h);
	ipl = cvCreateImage(cvSize(im_raw.w, im_raw.h), IPL_DEPTH_8U, im_raw.c);
	im_object = copy_image(im_raw);
	im_letter = letterbox_image(im_raw, net->w, net->h);
	if(left_right_or_full && comp_depth) 
	{ 
		ipl_l = cvCloneImage(ipl_l);    
		ipl_r = cvCloneImage(ipl_r);     
	}
	int count = 0;
	struct_display param_display;
	param_display.class_of_interest = class_of_interest;
	//demo_time = what_time_is_it_now();
	struct_detect param_detect;
	param_detect.class_of_interest = class_of_interest;
	param_detect.ratio_shrink = ratio_shrink;
	struct_depth param_depth;
	param_depth.parameter_poly = NULL;
	if(left_right_or_full && comp_depth)
	{
		compute_depth = true;
		if(fn_poly)
		{
			param_depth.parameter_poly = load_polynomial_coefficient(fn_poly);
			param_depth.parameter_poly->is_from_disparity_to_millimeter = is_disp_2_milli;
		}
		param_depth.percent_closest = percent_closest;
		param_depth.wsize = 13; param_depth.max_disp = 160; param_depth.lambda = 10000.0;   param_depth.sigma = 1.0;
		param_depth.both_disparity = both_disparity;
		param_depth.path_agg = path_agg;
		if (8 == path_agg)
		{
			param_depth.p1 = 6; param_depth.p2 = 96;
		}
		else if(4 == path_agg)
		{
			param_depth.p1 = 7; param_depth.p2 = 86;
		}
		else
		{
			std::cout << "path_agg is " << path_agg << ", which is wrong !!" << std::endl;
			exit(0);
		}
	}
	//std::cout << "demo aaa" << std::endl;
	struct_fetch param_fetch;
	param_fetch.left_right_or_full = left_right_or_full;
	param_fetch.is_ocams = is_ocam;
	param_fetch.gray_4_stereo = gray_4_stereo;
	param_fetch.srcImg = is_ocam ? &srcImg : NULL;
	param_fetch.image_size = is_ocam ? image_size : -1; 

	//std::cout << "demo bbb" << std::endl;
	if (left_right_or_full && comp_depth) 
	{
		param_depth.wls_filter = initDisparityWLSFilter(param_depth.wsize, param_depth.max_disp, param_depth.lambda, param_depth.sigma);

		struct_stereo param_stereo;    
		param_stereo.img_size = cv::Size(im_raw.w, im_raw.h);
		param_stereo.fn_ext = yml_extrinsic;
		param_stereo.fn_int = yml_intrinsic;
		param_stereo.alfa = alfa;
		isVerticalStereo = load_stereo_calibration_info(&param_stereo);
		param_fetch.parameter_stereo = &param_stereo;
		param_depth.parameter_stereo = &param_stereo;
		//mask_rect_remap.create(param_stereo.img_size, CV_8UC1); 
		mask_rect_remap = cv::Mat::zeros(param_stereo.img_size, CV_8UC1); 
		mat_disp = cv::Mat::zeros(param_stereo.img_size, CV_8UC1);
	}
	else 
	{
		param_fetch.parameter_stereo = NULL;
		param_depth.parameter_stereo = NULL;
	}

	pthread_mutex_init(&mutex_lock_contour, NULL);
	pthread_mutex_init(&mutex_lock_disp, NULL);
	pthread_mutex_init(&mutex_lock_rect, NULL);
	pthread_mutex_init(&mutex_lock_mask_remap, NULL);
	pthread_mutex_init(&mutex_lock_millimeter, NULL);
	//pthread_mutex_init(&mutex_lock_log, NULL);
	pthread_mutex_init(&mutex_lock_image, NULL);
	std::cout << "demo ccc" << std::endl;
	if(pthread_create(&fetch_thread, 0, fetch_in_thread, &param_fetch)) error("Thread creation failed");
	if(pthread_create(&detect_thread, 0, detect_in_thread, &param_detect)) error("Thread creation failed");
	if(left_right_or_full && comp_depth)
	{ 
		if(pthread_create(&depth_thread, 0, depth_in_thread, &param_depth)) error("Thread creation failed");
	}
	if(pthread_create(&display_thread, 0, display_in_thread, &param_display)) error("Thread creation failed");
	std::cout << "demo bbb" << std::endl;

	printf("prefix : %d\n", prefix);
	if(!prefix){
	}
	else
	{
		printf("eee");
		char name[256];
		sprintf(name, "%s_%08d", prefix, count);
		save_image(im_raw, name);
	}

	int rc, status;

	rc = pthread_join(fetch_thread, (void **)&status);
	//printf("  pthread\n");
	if (rc == 0)
	{
		//printf("in\n");
		printf("Completed join with fetch_thread. status = %d\n", status);
	}
	else
	{
		printf("ERROR; return code from pthread_join() is %d, fetch_thread\n", rc);
		return;
	}

	rc = pthread_join(detect_thread, (void **)&status);
	if (rc == 0)
	{
		printf("Completed join with detect_thread. status = %d\n", status);
	}
	else
	{
		printf("ERROR; return code from pthread_join() is %d, detect_thread\n", rc);
		return;
	}


	rc = pthread_join(display_thread, (void **)&status);
	if (rc == 0)
	{
		printf("Completed join with display_thread. status = %d\n", status);
	}
	else
	{
		printf("ERROR; return code from pthread_join() is %d, display_thread\n", rc);
		return;
	}

	if(comp_depth) 
	{
		rc = pthread_join(depth_thread, (void **)&status);
		if (rc == 0)
		{
			printf("Completed join with depth_thread. status = %d\n", status);
		}
		else
		{
			printf("ERROR; return code from pthread_join() is %d, depth_thread\n", rc);
			return;
		}
	}
	++count;
}

/*
   void demo_compare(char *cfg1, char *weight1, char *cfg2, char *weight2, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
   {
   demo_frame = avg_frames;
   predictions = calloc(demo_frame, sizeof(float*));
   image **alphabet = load_alphabet();
   demo_names = names;
   demo_alphabet = alphabet;
   demo_classes = classes;
   demo_thresh = thresh;
   demo_hier = hier;
   printf("Demo\n");
   net = load_network(cfg1, weight1, 0);
   set_batch_network(net, 1);
   pthread_t detect_thread;
   pthread_t fetch_thread;

   srand(2222222);

   if(filename){
   printf("video file: %s\n", filename);
   cap = cvCaptureFromFile();
   }else{
   cap = cvCaptureFromCAM(cam_index);

   if(w){
   cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
   }
   if(h){
   cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
   }
   if(frames){
   cvSetCaptureProperty(cap, CV_CAP_PROP_FPS, frames);
   }
   }

   if(!cap) error("Couldn't connect to webcam.\n");

   layer l = net->layers[net->n-1];
   demo_detections = l.n*l.w*l.h;
   int j;

   avg = (float *) calloc(l.outputs, sizeof(float));
   for(j = 0; j < demo_frame; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));

   boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
   probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
   for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes+1, sizeof(float));

   buff[0] = get_image_from_stream(cap);
   buff[1] = copy_image(buff[0]);
   buff[2] = copy_image(buff[0]);
   buff_letter[0] = letterbox_image(buff[0], net->w, net->h);
   buff_letter[1] = letterbox_image(buff[0], net->w, net->h);
   buff_letter[2] = letterbox_image(buff[0], net->w, net->h);
   ipl = cvCreateImage(cvSize(buff[0].w,buff[0].h), IPL_DEPTH_8U, buff[0].c);

   int count = 0;
   if(!prefix){
   cvNamedWindow("Demo", CV_WINDOW_NORMAL); 
   if(fullscreen){
   cvSetWindowProperty("Demo", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
   } else {
   cvMoveWindow("Demo", 0, 0);
   cvResizeWindow("Demo", 1352, 1013);
   }
   }

   demo_time = what_time_is_it_now();

   while(!demo_done){
buff_index = (buff_index + 1) %3;
if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");
if(!prefix){
	fps = 1./(what_time_is_it_now() - demo_time);
	demo_time = what_time_is_it_now();
	display_in_thread(0);
}else{
	char name[256];
	sprintf(name, "%s_%08d", prefix, count);
	save_image(buff[(buff_index + 1)%3], name);
}
pthread_join(fetch_thread, 0);
pthread_join(detect_thread, 0);
++count;
}
}
*/
#else   //  OPENCV
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg, float hier, int w, int h, int frames, int fullscreen)
{
	fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif  //  OPENCV

