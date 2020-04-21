// Copyright 1986-2018 Xilinx, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2018.3 (win64) Build 2405991 Thu Dec  6 23:38:27 MST 2018
// Date        : Tue Apr 21 16:50:38 2020
// Host        : DESKTOP-GSUMVJ7 running 64-bit major release  (build 9200)
// Command     : write_verilog -force -mode synth_stub
//               d:/cedRTL/cedRTL.srcs/sources_1/bd/design_1/ip/design_1_ced_0_0/design_1_ced_0_0_stub.v
// Design      : design_1_ced_0_0
// Purpose     : Stub declaration of top-level module interface
// Device      : xc7z020clg400-1
// --------------------------------------------------------------------------------

// This empty module with port declaration file causes synthesis tools to infer a black box for IP.
// The synthesis directives are for Synopsys Synplify support to prevent IO buffer insertion.
// Please paste the declaration into a Verilog source file or add the file as an additional source.
(* X_CORE_INFO = "ced,Vivado 2018.3" *)
module design_1_ced_0_0(image_in_TVALID, image_in_TREADY, 
  image_in_TDATA, image_in_TKEEP, image_in_TSTRB, image_in_TUSER, image_in_TLAST, 
  image_in_TID, image_in_TDEST, image_out_TVALID, image_out_TREADY, image_out_TDATA, 
  image_out_TKEEP, image_out_TSTRB, image_out_TUSER, image_out_TLAST, image_out_TID, 
  image_out_TDEST, ap_clk, ap_rst_n)
/* synthesis syn_black_box black_box_pad_pin="image_in_TVALID,image_in_TREADY,image_in_TDATA[23:0],image_in_TKEEP[2:0],image_in_TSTRB[2:0],image_in_TUSER[0:0],image_in_TLAST[0:0],image_in_TID[0:0],image_in_TDEST[0:0],image_out_TVALID,image_out_TREADY,image_out_TDATA[23:0],image_out_TKEEP[2:0],image_out_TSTRB[2:0],image_out_TUSER[0:0],image_out_TLAST[0:0],image_out_TID[0:0],image_out_TDEST[0:0],ap_clk,ap_rst_n" */;
  input image_in_TVALID;
  output image_in_TREADY;
  input [23:0]image_in_TDATA;
  input [2:0]image_in_TKEEP;
  input [2:0]image_in_TSTRB;
  input [0:0]image_in_TUSER;
  input [0:0]image_in_TLAST;
  input [0:0]image_in_TID;
  input [0:0]image_in_TDEST;
  output image_out_TVALID;
  input image_out_TREADY;
  output [23:0]image_out_TDATA;
  output [2:0]image_out_TKEEP;
  output [2:0]image_out_TSTRB;
  output [0:0]image_out_TUSER;
  output [0:0]image_out_TLAST;
  output [0:0]image_out_TID;
  output [0:0]image_out_TDEST;
  input ap_clk;
  input ap_rst_n;
endmodule
