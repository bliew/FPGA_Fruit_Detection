#include "hls_opencv.h"
#include <ap_fixed.h>
#include "hls_video.h"

using namespace cv;

typedef hls::stream< ap_axiu<24,1,1,1> > AXI_STREAM;
typedef hls::Mat<100,100, HLS_8UC3> RGB_IMAGE;
typedef ap_fixed<10,2, AP_RND, AP_SAT> coeff_type;

typedef hls::Scalar<1, unsigned char> GRAY_PIX;
typedef hls::Scalar<3, unsigned char> RGB_PIX;

#define INPUT_IMAGE		"banana.png"
#define OUTPUT_IMAGE 	"test_output.png"

void ced(AXI_STREAM& input, AXI_STREAM& output);
int main(int argc, char** argv){

	IplImage* src = cvLoadImage("banana.png");
	IplImage* dst = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);

	AXI_STREAM src_axi, dst_axi;
	IplImage2AXIvideo(src, src_axi);

	ced(src_axi, dst_axi);

	AXIvideo2IplImage(dst_axi, dst);
	cvSaveImage("test_output3.jpg", dst);

}
