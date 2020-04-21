#include <hls_video.h>
#include <stdint.h>
#include <stdio.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include "ced.h"


coeff_type const1 = 0.114; //B
coeff_type const2 = 0.587; //G
coeff_type const3 = 0.2989; //R


void RGB2Gray(RGB_IMAGE& img_in,RGB_IMAGE& img_out) {

	RGB_PIX pin;
	RGB_PIX pout;
	char gray;

L_row: for(int row = 0; row < 100; row++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=100
	L_col: for(int col = 0; col < 100; col++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=100
#pragma HLS loop_flatten off
#pragma HLS PIPELINE II = 1
//#pragma HLS unroll
           img_in >> pin;

		   gray =  const1 * pin.val[0] + const2 * pin.val[1] + const3 * pin.val[2];
		   pout.val[0] = gray;
		   pout.val[1] = gray;
		   pout.val[2] = gray;

           img_out << pout;
        }
    }
}

template<int KH,int KW,typename K_T>
void GaussianKernel(hls::Window<KH,KW,K_T> &kernel)
{
#pragma HLS INLINE
const int k_val[3*3] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
	for (int i = 0; i < 3; i++){
#pragma HLS unroll
		for (int j = 0; j < 3; j++){
#pragma HLS unroll
			kernel.val[i][j] = k_val[i*3+j]*0.0625;
		}
	}
}

void duplicate(RGB_IMAGE& img_in, RGB_IMAGE& img_outa, RGB_IMAGE& img_outb) {

	RGB_PIX pin;
	RGB_PIX pout;

L_row: for(int row = 0; row < 100; row++) {
#pragma HLS LOOP_TRIPCOUNT min=720 max=1080

L_col: for(int col = 0; col < 100; col++) {
#pragma HLS LOOP_TRIPCOUNT min=100 max=100
#pragma HLS loop_flatten off
#pragma HLS PIPELINE II = 1

//#pragma HLS unroll
           img_in >> pin;

		   pout = pin;

           img_outa << pout;
           img_outb << pout;
        }
    }
}

template<int KH,int KW,typename K_T>
void ySobelKernel(hls::Window<KH,KW,K_T> &kernel)
{

	const char ycoefficients[3][3] = { {1,2,1},
	                                  {0,0,0},
	                                  {-1,-2,-1} };
	for (int i=0;i<3;i++){
		for (int j=0;j<3;j++){
//#pragma HLS unroll
		kernel.val[i][j]=ycoefficients[i][j];
		}
	}
}

template<int KH,int KW,typename K_T>
void xSobelKernel(hls::Window<KH,KW,K_T> &kernel)
{
	const char xcoefficients[3][3] =  { {-1,0,1},
										{-2,0,2},
										{-1,0,1} };

	for (int i=0;i<3;i++){
			for (int j=0;j<3;j++){
//#pragma HLS unroll
			kernel.val[i][j]=xcoefficients[i][j];
			}
	}
}

void addSobel(RGB_IMAGE& img_ina, RGB_IMAGE& img_inb, RGB_IMAGE& img_out) {

	RGB_PIX pin0, pin1;
	RGB_PIX pout;

L_row: for(int row = 0; row < 100; row++) {
#pragma HLS LOOP_TRIPCOUNT min=100 max=100

L_col: for(int col = 0; col < 100; col++) {
#pragma HLS LOOP_TRIPCOUNT min=100 max=100
#pragma HLS loop_flatten off
#pragma HLS PIPELINE II = 1
//#pragma HLS unroll

           img_ina >> pin0;
           img_inb >> pin1;

		   pout = (pin0 + pin1);

           img_out << pout;
        }
    }
}



void ced(AXI_STREAM& image_in, AXI_STREAM& image_out){


#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis register both port=image_in
#pragma HLS INTERFACE axis register both port=image_out


int ximage = 100;
int yimage = 100;


#pragma HLS DATAFLOW

	RGB_IMAGE src(ximage, yimage);
	RGB_IMAGE dup_1(ximage, yimage);
	RGB_IMAGE dup_2(ximage, yimage);
	RGB_IMAGE dsta(ximage, yimage);
	RGB_IMAGE dstb(ximage, yimage);
	RGB_IMAGE dstc(ximage, yimage);
	RGB_IMAGE dstd(ximage, yimage);
	RGB_IMAGE dste(ximage, yimage);
	RGB_IMAGE dstf(ximage, yimage);


	hls::AXIvideo2Mat(image_in, src);

	hls::Point_<int> anchor = hls::Point_<int>(-1,-1);

	RGB2Gray(src,dsta);

	hls::Window<3, 3, ap_fixed<16,2,AP_RND> 	> gkernel;
	GaussianKernel(gkernel);
	//hls::GaussianBlur<3,3>(src, dsta);

	hls::Filter2D(dsta,dstb,gkernel,anchor);

	duplicate(dstb,dup_1,dup_2);

	hls::Window<3, 3, char> ykernel;
	ySobelKernel(ykernel);
	hls::Filter2D(dup_1,dstc,ykernel,anchor);

	hls::Window<3, 3, char> xkernel;
	xSobelKernel(xkernel);
	hls::Filter2D(dup_2,dstd,xkernel,anchor);

	addSobel(dstc, dstd, dste);
	//hls::Point_<int> anchor = hls::Point_<int>(-1,-1);
	//hls::Duplicate(dsta,dstb,dstc);
	//hls::Filter2D(dstb,dstd,kernelx,anchor);
	//hls::AddWeighted(dstd,0.5,dste, 0.5, 0.0,dstf);

	hls::Mat2AXIvideo(dste, image_out);

}
