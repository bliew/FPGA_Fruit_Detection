#include <hls_opencv.h>
#include <stdint.h>
#include <stdio.h>
#include <malloc.h>

using namespace cv;
void conv(uint8_t * image_in, uint8_t * image_out);
int main(){
   Mat im = imread("basket.jpg",CV_LOAD_IMAGE_GRAYSCALE);
   uint8_t *image_in;
   uint8_t *image_out;
//	image_in = (uint8_t *)malloc(1080*1920* sizeof(uint8_t));
//	image_out = (uint8_t *)malloc(1080*1920* sizeof(uint8_t));
//   memcpy(image_in,im.data,sizeof(uint8_t)*1080*1920);
	image_in = (uint8_t *)malloc(100*100* sizeof(uint8_t));
	image_out = (uint8_t *)malloc(100*100* sizeof(uint8_t));
    memcpy(image_in,im.data,sizeof(uint8_t)*100*100);
   conv(image_in,image_out);
//   Mat out = Mat(1080,1920,CV_8UC1,image_out);
   Mat out = Mat(100,100,CV_8UC1,image_out);
   namedWindow("output");
   imshow("output",out);
   waitKey(0);

 return 0;
}
#include <hls_opencv.h>
#include <stdint.h>
#include <stdio.h>
#include <malloc.h>

using namespace cv;
void conv(uint8_t * image_in, uint8_t * image_out);
int main(){
   Mat im = imread("basket.jpg",CV_LOAD_IMAGE_GRAYSCALE);
   uint8_t *image_in;
   uint8_t *image_out;
//	image_in = (uint8_t *)malloc(1080*1920* sizeof(uint8_t));
//	image_out = (uint8_t *)malloc(1080*1920* sizeof(uint8_t));
//   memcpy(image_in,im.data,sizeof(uint8_t)*1080*1920);
	image_in = (uint8_t *)malloc(100*100* sizeof(uint8_t));
	image_out = (uint8_t *)malloc(100*100* sizeof(uint8_t));
    memcpy(image_in,im.data,sizeof(uint8_t)*100*100);
   conv(image_in,image_out);
//   Mat out = Mat(1080,1920,CV_8UC1,image_out);
   Mat out = Mat(100,100,CV_8UC1,image_out);
   namedWindow("output");
   imshow("output",out);
   waitKey(0);

 return 0;
}
