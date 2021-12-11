
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// Program to calculate PSNR value of original video frames and reconstructed frames

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <conio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <dos.h>
#include <float.h>

#define TRUE 1
#define FALSE 0

#pragma pack(push,1)
 typedef struct bmpheader
 {   
	unsigned char bfType[2];     //BM
	unsigned long bfSize;        //file size in bytes
	unsigned short bfReserved1;  //reserved, has 0 value
	unsigned short bfReserved2;  //reserved; has 0 value
	unsigned long bfOffBits;     //offset in bytes from bitmap header to the bitmap bits
 }bmpfileheader;

 typedef struct bmpinfoheader
 {
	unsigned long biSize;         //number of bytes required by the struct
	unsigned long biWidth;        //width in pixels
	unsigned long biHeight;       //height in pixels
	unsigned short biPlanes;      //number of color planes, must be 1
	unsigned short biBitCount;    //number of bit per pixel
	unsigned long biCompression;  //type of compression
	unsigned long biSizeImage;    //size of image in bytes
	unsigned long biXPelsPerMeter;//number of pixels per meter in x axis
	unsigned long biYPelsPerMeter;//number of pixels per meter in y axis
	unsigned long biClrUsed;      //number of colors used by th ebitmap
	unsigned long biClrImportant; //number of colors that are important
	}bmpfileinfoheader;
 #pragma pack(pop)
 
 bmpfileheader hhp;
 bmpfileinfoheader hp;
 double total_psnr = 0;
 char initial_path[100],output_path[100],fixed_path[100],fixed_output_path[100];
 unsigned char *frame1_data=NULL,*frame2_data=NULL;

 void psnr2();
 void create_path(int);
 void read_frame(int);
 double calculatePSNR(unsigned char *, unsigned char *);

 __global__ void psnr(unsigned char* data1, unsigned char* data2, unsigned char* diff)
 {
	 int UniqueBlockIndex = blockIdx.y * gridDim.x + blockIdx.x;

     int UniqueThreadIndex = UniqueBlockIndex * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

     int iindex1 = 3*UniqueThreadIndex;        // compute byte offset
     int iindex2 = 3*UniqueThreadIndex;
	 int oindex = 3*UniqueThreadIndex;

	 int diff_y, diff_u, diff_v;
	 unsigned char data_y1, data_u1, data_v1, data_y2, data_u2, data_v2;

	 data_y1 = data1[iindex1];
	 data_u1 = data1[iindex1+1];
	 data_v1 = data1[iindex1+2];

	 data_y2 = data2[iindex2];
	 data_u2 = data2[iindex2+1];
	 data_v2 = data2[iindex2+2];

	 if(data_y1 > data_y2)
		diff_y = data_y1 - data_y2;
	 else
		diff_y = data_y2 - data_y1;

	 if(data_u1 > data_u2)
		diff_u = data_u1 - data_u2;
	 else
		diff_u = data_u2 - data_u1;

	 if(data_v1 > data_v2)
		diff_v = data_v1 - data_v2;
	 else
		diff_v = data_v2 - data_v1;

	 diff[oindex] = diff_y;
	 diff[oindex+1] = diff_u;
	 diff[oindex+2] = diff_v;
 }

 int main ()
 {
	 psnr2();
	 return 0;
 }

 void psnr2()
 {
	 int frame_count;
	 FILE *fptr=NULL,*fp=NULL;		
	 fptr=fopen("I:\\Input_Frames\\frame_count.txt", "rb");
	 if(fptr == (FILE *)0) 
	 {  
		 printf("File opening error.\n"); 
	 }
	 
	 fscanf(fptr,"%d",&frame_count);
	 fclose(fptr);
	 fptr=NULL;

	 strcpy(initial_path,"I:\\Input_Frames\\Frames\\");
	 strcpy(output_path,"I:\\Input_Frames\\Output_frames\\");
	 strcpy(fixed_path,initial_path);
	 strcpy(fixed_output_path,output_path);

	 create_path(frame_count);

	 printf("\n Total PSNR : %lf",total_psnr);

	 double avg_psnr;

	 avg_psnr = total_psnr / frame_count;

	 printf("\n Average PSNR : %lf",avg_psnr);
	 getch();
	 
	 fp = fopen("I:\\Input_Frames\\psnr.txt", "wb");
	 if(fp == (FILE *)0) 
	 {  
		 printf("File opening error.\n"); 
	 }
	 
	 fprintf(fp,"%lf",avg_psnr);
	 fclose(fp);
	 fp = NULL;
 }

 void create_path(int frame_count)
 {
	 int i=0, no_of_frames_processed=0;
	 char c[5],extension_in[5],extension_op[5],number_creation[5];
	 strcpy(extension_in,".BMP");			
	 strcpy(extension_op,".BMP");

	 for(i=0;i<frame_count;i++)
	 {
		 itoa(i,c,10);
		 strcpy(number_creation,c);
		
		 strcat(initial_path,number_creation);
		 strcat(initial_path,extension_in);

		 strcat(output_path,number_creation);
		 strcat(output_path,extension_op);

		 read_frame(i);

		 no_of_frames_processed++;
		
		 strcpy(initial_path,fixed_path);
		 strcpy(output_path,fixed_output_path);
	 }
 }

 void read_frame(int frame_no)
 {
	 FILE *fp_in=NULL, *fp_op=NULL;

     //Open input file:
     fp_in = fopen(initial_path, "rb");
	 if(fp_in == (FILE *)0) 
	 {  
		 printf("File opening error.\n"); 
	 }

	 fread(&hhp, sizeof(bmpfileheader),1,fp_in);

	 if(hhp.bfType!=(unsigned char *)0x4D42)
	 {	
		//printf("\nIs a bmp file");
	 }
	 
	 // Reading content of bmp Info Header
	 fread(&hp, sizeof(bmpfileinfoheader),1,fp_in);

	 frame1_data = (unsigned char*)malloc(sizeof(char)*hp.biSizeImage);

	 fseek(fp_in,sizeof(char)*hhp.bfOffBits,SEEK_SET);
	 fread(frame1_data,sizeof(char),hp.biSizeImage, fp_in);

	 fclose(fp_in);
	 fp_in = NULL;

	 //Open output file:
     fp_op = fopen(output_path, "rb");
	 if(fp_op == (FILE *)0) 
	 {  
		 printf("File opening error.\n"); 
	 }
	 
	 fread(&hhp, sizeof(bmpfileheader),1,fp_op);

	 if(hhp.bfType!=(unsigned char *)0x4D42)
	 {	
		//printf("\nIs a bmp file");
	 }
	 
	 // Reading content of bmp Info Header
	 fread(&hp, sizeof(bmpfileinfoheader),1,fp_op);

	 frame2_data = (unsigned char*)malloc(sizeof(char)*hp.biSizeImage);

	 fseek(fp_op,sizeof(char)*hhp.bfOffBits,SEEK_SET);
	 fread(frame2_data,sizeof(char),hp.biSizeImage, fp_op);

	 fclose(fp_op);
	 fp_op = NULL;

	 total_psnr += calculatePSNR(frame1_data,frame2_data); 

	 free(frame1_data);
	 free(frame2_data);
 }

 double calculatePSNR(unsigned char *frame1_data, unsigned char *frame2_data)
 {
	unsigned char *d_frame1=NULL;
	unsigned char *d_frame2=NULL;
	unsigned char *diff=NULL;
	unsigned char *d_diff=NULL;

	unsigned char *r = NULL;
	unsigned char *g = NULL;
	unsigned char *b = NULL;

	float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

	int size = 0, j = 0, size2 = 0;
	size = hp.biSizeImage/3;

	int sum_r = 0, sum_g = 0, sum_b = 0, val_r = 0, val_g = 0, val_b = 0;
	int i = 0;
	double calpsnr = 0;
	double mse_r = 0, mse_g = 0, mse_b = 0, mse = 0;
	double val = 0;

	r = (unsigned char*)malloc(sizeof(char)*size);
	g = (unsigned char*)malloc(sizeof(char)*size);
	b = (unsigned char*)malloc(sizeof(char)*size);

	diff = (unsigned char*)malloc(sizeof(char)*hp.biSizeImage);
	cudaFree(diff);
	
	size2 = hp.biSizeImage;

	for(i=0;i<size2;i++)
	{
		diff[i] = 0;
	}

	cudaMalloc(&d_frame1, sizeof(char)*hp.biSizeImage);
	cudaMalloc(&d_frame2, sizeof(char)*hp.biSizeImage);
	cudaMalloc(&d_diff, sizeof(char)*hp.biSizeImage);

	cudaMemcpy(d_frame1, frame1_data, sizeof(char) * hp.biSizeImage, cudaMemcpyHostToDevice);
	cudaMemcpy(d_frame2, frame2_data, sizeof(char) * hp.biSizeImage, cudaMemcpyHostToDevice);
	cudaMemcpy(d_diff, diff, sizeof(char) * hp.biSizeImage, cudaMemcpyHostToDevice);

	dim3 block(16,16);									// 16x16 = 256 threads  // 32x32 = 1024      // No of threads per block
	dim3 grid(hp.biHeight/16, hp.biWidth/16);		    // 40x30 = 1200 blocks  // 20x15 = 300       // No of blocks per grid
														// 1200x256 = 307200	// 1024x300 = 307200
	cudaEventRecord(start, 0);

	psnr<<<grid,block>>>(d_frame1,d_frame2,d_diff);

	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
	printf("\nKernel elapsed time for difference:  %3.3f ms \n", time);

	cudaMemcpy(frame2_data, d_frame2, sizeof(char)*hp.biSizeImage, cudaMemcpyDeviceToHost);
	cudaMemcpy(frame1_data, d_frame1, sizeof(char)*hp.biSizeImage, cudaMemcpyDeviceToHost);
	cudaMemcpy(diff, d_diff, sizeof(char)*hp.biSizeImage, cudaMemcpyDeviceToHost);

	cudaFree(d_frame1);
	cudaFree(d_frame2);
	cudaFree(d_diff);

	for(i=0,j=0;i<size2;i++)
	{
		r[j] = diff[i];
		g[j] = diff[++i];
		b[j] = diff[++i];
		j++;
	}

	for(i=0;i<size;i++)
	{
		val_r = r[i];
		val_r = val_r * val_r;
		sum_r = sum_r + val_r;

		val_g = g[i];
		val_g = val_g * val_g;
		sum_g = sum_g + val_g;

		val_b = b[i];
		val_b = val_b * val_b;
		sum_b = sum_b + val_b;
	}

	mse_r = sum_r;

	mse_g = sum_g;

	mse_b = sum_b;

	mse = (mse_r + mse_g + mse_b)/921600;

	double temp = 0;
	temp = 255*255;
	temp = temp/mse;

	calpsnr = 10 * log10(double(temp));

	printf("\n PSNR : %lf",calpsnr);
	
	free(diff);
	free(r);
	free(g);
	free(b);

	return calpsnr;
 }