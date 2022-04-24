#ifdef OPENCV

#include "stdio.h"
#include "stdlib.h"
#include "opencv2/opencv.hpp"
#include "image.h"

/*
################################
借助OpenCV库实现 图像格式转换（像素矩阵格式与image格式转换）
################################
*/

using namespace cv;



//   ***************
//   kalman_filter_tracking卡尔曼滤波跟踪

/*class kalman{

public:
	float dt=0.005;
        float A[2][2]={1,0,
		      	0,1};
        cvMat va=cvMat(2,2,CV_64FC1,A);
	cvMat vat=cvMat(2,2,CV_64FC1,A);
	float U[2][1]={0,
			0};
        cvMat vu=cvMat(2,1,CV_64FC1,u);
	float B[2][1]={0,
			255};
	cvMat vb=cvMat(2,1,CV_64FC1,B);
	float P[2][2]={3.0,0,
			0,3.0};
        cvMat vp=cvMat(2,2,CV_64FC1,P);
	cvMat vk=cvMat(2,2,CV_64FC1,P);
	cvMat vks=cvMat(2,2,CV_64FC1,P);
	float F[2][2]={1.0,0.005,
			0,1.0};
        cvMat vf=cvMat(2,2,CV_64FC1,F);
        cvMat vft=cvMat(2,2,CV_64FC1,F);
	float Q[2][2]={1,0,
			0,1};
	cvMat vq=cvMat(2,2,CV_64FC1,Q);
	float R[2][2]={1,0,
			0.1};
        cvMat vr=cvMat(2,2,CV_64FC1,R);
	cvMat vc=cvMat(2,2,CV_64FC1,R);
	cvMat vcinv=cvMat(2,2,CV_64FC1,R);
	float lastresult[2][1]={0,
				255};
	cvMat vlastresult=cvMat(2,2,CV_64FC1,lastresult);


	cvMat predict()
	{
		cvMul(&vf,&vu,&vu);
		cvTranspose(&vf,&vft);
		cvMul(&vp,&vft,&vft);
		cvMul(&vf,&vft,&vft);
                vp=vft+vq;
                vlastresult=vu

        return vu;
        
	cvMat correct(cvMat vbs,bool flag)
	{
		if (!flag)
			vb=vlastresult;
		else
  			vb=vbs
		cvTranspose(&va,&vat);
		cvMul(&vp,&vat,&vat);
		cvMul(&va,&vat,&vat);
  		vc=vat+vr;

		cvTranspose(&va,&vat);
       		cvInvert(&vc,&vcinv);
		cvMul(&vat,&vcinv,&vcinv);
		cvMul(&vp,&vcinv,&vk);
  		
		cvMul(&va,&vu,&vu);
                cvMul(&vk,&(b-vu),&vks);
		vu=vu+vks;
		cvTranspose(&vk,&vks);
		cvMul(&vc,&vks,&vc);
		cvMul(&vk,&vc,&vc);
		vp=vp+vc;
	return vu;
	}


};*/


extern "C" {

IplImage *image_to_ipl(image im)
{
    int x,y,c;
    IplImage *disp = cvCreateImage(cvSize(im.w,im.h), IPL_DEPTH_8U, im.c);
    int step = disp->widthStep;
    for(y = 0; y < im.h; ++y){
        for(x = 0; x < im.w; ++x){
            for(c= 0; c < im.c; ++c){
                float val = im.data[c*im.h*im.w + y*im.w + x];
                disp->imageData[y*step + x*im.c + c] = (unsigned char)(val*255);
            }
        }
    }
    return disp;
}

image ipl_to_image(IplImage* src)
{
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    image im = make_image(w, h, c);
    unsigned char *data = (unsigned char *)src->imageData;
    int step = src->widthStep;
    int i, j, k;

    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){
                im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
            }
        }
    }
    return im;
}

Mat image_to_mat(image im)
{
    image copy = copy_image(im);
    constrain_image(copy);
    if(im.c == 3) rgbgr_image(copy);

    IplImage *ipl = image_to_ipl(copy);
    Mat m = cvarrToMat(ipl, true);
    cvReleaseImage(&ipl);
    free_image(copy);
    return m;
}

image mat_to_image(Mat m)
{
    IplImage ipl = m;
    image im = ipl_to_image(&ipl);
    rgbgr_image(im);
    return im;
}

void *open_video_stream(const char *f, int c, int w, int h, int fps)
{
    VideoCapture *cap;
    if(f) cap = new VideoCapture(f);
    else cap = new VideoCapture(c);
    if(!cap->isOpened()) return 0;
    if(w) cap->set(CV_CAP_PROP_FRAME_WIDTH, w);
    if(h) cap->set(CV_CAP_PROP_FRAME_HEIGHT, w);
    if(fps) cap->set(CV_CAP_PROP_FPS, w);
    return (void *) cap;
}

image get_image_from_stream(void *p)
{
    VideoCapture *cap = (VideoCapture *)p;
    Mat m;
    *cap >> m;
    if(m.empty()) return make_empty_image(0,0,0);
    return mat_to_image(m);
}

image load_image_cv(char *filename, int channels)
{
    int flag = -1;
    if (channels == 0) flag = -1;
    else if (channels == 1) flag = 0;
    else if (channels == 3) flag = 1;
    else {
        fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);
    }
    Mat m;
    m = imread(filename, flag);
    if(!m.data){
        fprintf(stderr, "Cannot load image \"%s\"\n", filename);
        char buff[256];
        sprintf(buff, "echo %s >> bad.list", filename);
        system(buff);
        return make_image(10,10,3);
        //exit(0);
    }
    image im = mat_to_image(m);
    return im;
}

int show_image_cv(image im, const char* name, int ms)
{
    Mat m = image_to_mat(im);
    imshow(name, m);
    int c = waitKey(ms);
    if (c != -1) c = c%256;
    return c;
}

void make_window(char *name, int w, int h, int fullscreen)
{
    namedWindow(name, WINDOW_NORMAL); 
    if (fullscreen) {
        setWindowProperty(name, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    } else {
        resizeWindow(name, w, h);
        if(strcmp(name, "Demo") == 0) moveWindow(name, 0, 0);
    }
}

}

#endif
