#ifndef _CED_H_
#define _CED_H_

#include <ap_fixed.h>
#include "hls_video.h"

typedef hls::stream< ap_axiu<24,1,1,1> > AXI_STREAM;
typedef hls::Mat<100,100, HLS_8UC3> RGB_IMAGE;
typedef ap_fixed<10,2, AP_RND, AP_SAT> coeff_type;

typedef hls::Scalar<1, unsigned char> GRAY_PIX;
typedef hls::Scalar<3, unsigned char> RGB_PIX;

#define INPUT_IMAGE		"banana.png"
#define OUTPUT_IMAGE 	"test_output.png"

void ced(AXI_STREAM& input, AXI_STREAM& output);

#endif
