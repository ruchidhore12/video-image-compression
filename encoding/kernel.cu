
// This is Encoding Program implemented in CUDA (Common Unified Device Architecture).
// Input : BMP uncompressed images
// BMP images consist 54 bytes header - bmpheader & bmpinfoheader
// 5 modules : RGB to YUV, Motion Estimation, Simple Difference & RLE(byte-level)/Packbit(byte-level)
// BMP images act as input to first module : RGB to YUV
// Motion Estimation to find out Key-Frames & Sub-Frames using PSNR formula with MSE(Mean Square Error)
// Threshold value assumed is 40dB
// Simple Difference : Sub-Frame - Key-Frame
// RLE : Structure - Value & Count
// Packbit : Structure - Value & Count
// Output Stream stored in .bin file
// All Rights Reserved (c)

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <conio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <dos.h>
#include <float.h>
#include <time.h>

/* BMP Header:
   BM
   File size in bytes
   Reserved, has 0 value
   Reserved, has 0 value
   Offset in bytes from bitmap header to the bitmap bits
   Number of bytes required by the struct
   Width in pixels
   Height in pixels
   Number of color planes, must be 1
   Number of bit per pixel
   Type of compression
   Size of image in bytes
   Number of pixels per meter in x axis
   Number of pixels per meter in y axis
   Number of colors used by th ebitmap
   Number of colors that are important
*/

#pragma pack(push,1)
 typedef struct bmpheader
 {   
	unsigned char bfType[2];     
	unsigned long bfSize;        
	unsigned short bfReserved1;  
	unsigned short bfReserved2;  
	unsigned long bfOffBits;     
 }bmpfileheader;

 typedef struct bmpinfoheader
 {
	unsigned long biSize;         
	unsigned long biWidth;        
	unsigned long biHeight;       
	unsigned short biPlanes;      
	unsigned short biBitCount;    
	unsigned long biCompression;  
	unsigned long biSizeImage;    
	unsigned long biXPelsPerMeter;
	unsigned long biYPelsPerMeter;
	unsigned long biClrUsed;      
	unsigned long biClrImportant; 
 }bmpfileinfoheader;
 #pragma pack(pop)

 __global__ void rgbToyuv(unsigned char* data)
{
     /* Calculating ID of each thread */ 

	 int UniqueBlockIndex = blockIdx.y * gridDim.x + blockIdx.x;

     int UniqueThreadIndex = UniqueBlockIndex * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

	 /* Compute byte offset for accessing components of each pixel */

	 int iindex = 3*UniqueThreadIndex;        
     int oindex = 3*UniqueThreadIndex;

	 unsigned char temp, temp1, temp2;
	 unsigned char data_r, data_g, data_b;

	 data_b = data[iindex];
	 data_g = data[iindex+1];
	 data_r = data[iindex+2];

	 /* 
	    RGB to YUV
	    Y = 0.299R + 0.587G + 0.114B
	    U = -0.147R - 0.289G + 0.436B
	    V = 0.615R - 0.515G - 0.100B
	 */

	 temp = floor(data_r * 0.2990f + data_g * 0.5870f + data_b * 0.1140f);				// temp for Y
	 temp1 = floor(data_r * -0.1470f + data_g * -0.2890f + data_b * 0.4360f) + 128;		// temp1 for U
	 temp2 = floor(data_r * 0.6150f + data_g * -0.5150f + data_b * -0.1000f) + 128;		// temp2 for V

     data[oindex] = temp;																// Storing Y 
     data[oindex+1] = temp1;															// Storing U
     data[oindex+2] = temp2;															// Storing V
 }

 __global__ void simpleDiff(unsigned char* data1, unsigned char* data2, unsigned char* diff)
 {
	 /* Calculating ID of each thread */

	 int UniqueBlockIndex = blockIdx.y * gridDim.x + blockIdx.x;

     int UniqueThreadIndex = UniqueBlockIndex * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

	 /* Compute byte offset for accessing components of each pixel */

     int iindex1 = 3*UniqueThreadIndex;
     int iindex2 = 3*UniqueThreadIndex;
	 int oindex = 3*UniqueThreadIndex;

	 unsigned char diff_y, diff_u, diff_v;
	 unsigned char data_y1, data_u1, data_v1, data_y2, data_u2, data_v2;

	 data_y1 = data1[iindex1];			// Y component of Key-frame
	 data_u1 = data1[iindex1+1];        // U component of Key-frame
	 data_v1 = data1[iindex1+2];        // V component of Key-frame

	 data_y2 = data2[iindex2];			// Y component of Sub-frame
	 data_u2 = data2[iindex2+1];		// U component of Sub-frame
	 data_v2 = data2[iindex2+2];		// V component of Sub-frame

	 /* Taking Difference between Sub-Frame and Key-Frame */

     diff_y = data_y2 - data_y1;
     diff_u = data_u2 - data_u1;
	 diff_v = data_v2 - data_v1;

	 diff[oindex] = diff_y;			   // Storing difference of Y-component
	 diff[oindex+1] = diff_u;		   // Storing difference of U-component
	 diff[oindex+2] = diff_v;		   // Storing difference of V-component
 }

 __global__ void psnr(unsigned char* data1, unsigned char* data2, unsigned char* diff)
 {
	 /* Calculating ID of each thread */

	 int UniqueBlockIndex = blockIdx.y * gridDim.x + blockIdx.x;

     int UniqueThreadIndex = UniqueBlockIndex * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

	 /* Compute byte offset for accessing components of each pixel */

     int iindex1 = 3*UniqueThreadIndex;
     int iindex2 = 3*UniqueThreadIndex;
	 int oindex = 3*UniqueThreadIndex;

	 int diff_y, diff_u, diff_v;
	 unsigned char data_y1, data_u1, data_v1, data_y2, data_u2, data_v2;

	 data_y1 = data1[iindex1];			// Y component of First-frame
	 data_u1 = data1[iindex1+1];		// U component of First-frame
	 data_v1 = data1[iindex1+2];		// V component of First-frame

	 data_y2 = data2[iindex2];			// Y component of Second-frame
	 data_u2 = data2[iindex2+1];		// U component of Second-frame
	 data_v2 = data2[iindex2+2];		// V component of Second-frame

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

 bmpfileheader hhp;
 bmpfileinfoheader hp;
 double rgbToyuv_time = 0, simplediff_time = 0, rle_time = 0, psnr_time = 0, packbit_time = 0;
 char append_char;
 double difference = 0;
 double original_filesize = 0, compressed_filesize = 0;
 int frame_count=0;
 clock_t launch, done;
 char initial_path[100],output_path[100],fixed_path[100],fixed_output_path[100];
 unsigned char *frame1_data=NULL,*frame2_data=NULL;
 unsigned char *frame1_data2=NULL, *frame2_data2=NULL;

 void create_path(int);
 void read_frame(int);
 void Transform_RgbToYuv(FILE *,int);
 void Simple_Difference(unsigned char *,unsigned char *,int);
 void Transform_YuvToRgb(unsigned char *);
 void yuvSeparation(unsigned char *);
 void packbits(unsigned char *, FILE *);
 int calculatePSNR(unsigned char *,unsigned char *,int);
 void run_length_encoding(unsigned char *, FILE *);

 int main()
 {
	 FILE *fptr=NULL;		
	 fptr=fopen("I:\\Input_Frames\\frame_count.txt", "rb");
	 if(fptr == (FILE *)0) 
	 {  
		 printf("File opening error.\n"); 
	 }
	
	 fscanf(fptr,"%d",&frame_count);
	 fclose(fptr);
	 fptr=NULL;

	strcpy(initial_path,"I:\\Input_Frames\\Frames\\");
	strcpy(output_path,"I:\\Input_Frames\\compressed_files\\");
	strcpy(fixed_path,initial_path);
	strcpy(fixed_output_path,output_path);			 
    create_path(frame_count);
	
	printf("\nTime taken by RGB to YUV module : %lf ms",rgbToyuv_time);
	printf("\nTime taken by PSNR module : %lf ms",psnr_time);
	printf("\nTime taken by Simple Difference module : %lf ms",simplediff_time);
	printf("\nTime taken by Packbits module : %lf seconds",packbit_time);

	FILE *fp1=NULL,*fp2=NULL;

    //Open input file:
    fp1 = fopen("I:\\Input_Frames\\op1.txt", "w");
	if(fp1 == (FILE *)0) 
	{  
		printf("File opening error.\n"); 
	}
	
	fp2 = fopen("I:\\Input_Frames\\op2.txt", "w");
	if(fp2 == (FILE *)0) 
    {  
		printf("File opening error.\n"); 
	}
	
	original_filesize = original_filesize/1048576;
	fprintf(fp1,"%lf",original_filesize);
    compressed_filesize = compressed_filesize/1048576;
	fprintf(fp2,"%lf",compressed_filesize);

	fclose(fp1);
	fp1 = NULL;
	fclose(fp2);
	fp1 = NULL;
	return 0;
 }

 void create_path(int frame_count)
 {
	 int i=0, no_of_frames_processed=0;
	 char c[5],extension[5],extension_txt[5],number_creation[5];
	 strcpy(extension,".BMP");			
	 strcpy(extension_txt,".bin");

	 for(i=0;i<frame_count;i++)
	 {
		 itoa(i,c,10);
		 strcpy(number_creation,c);

		 strcat(initial_path,number_creation);
		 strcat(initial_path,extension);

		 strcat(output_path,number_creation);
		 strcat(output_path,extension_txt);

		 read_frame(i);

		 no_of_frames_processed++;
		
		 strcpy(initial_path,fixed_path);
		 strcpy(output_path,fixed_output_path);
	 }
 }

 void read_frame(int frame_no)
 {
	 FILE *fp=NULL;

     //Open input file:
     fp = fopen(initial_path, "rb");
	 if(fp == (FILE *)0) 
	 {  
		 printf("File opening error.\n"); 
	 }
	 
	 /* For calculating File Size */

	 fseek(fp, 0, SEEK_END);				// seek to end of file
	 original_filesize += ftell(fp);		// get current file pointer
	 fseek(fp, 0, SEEK_SET);				// seek back to beginning of file

	 fread(&hhp, sizeof(bmpfileheader),1,fp);

	 if(hhp.bfType!=(unsigned char *)0x4D42)
	 {	
		printf("\nIs a bmp file");
	 }
	 
	 
	 fread(&hp, sizeof(bmpfileinfoheader),1,fp);

	 Transform_RgbToYuv(fp,frame_no);
 }

 /* FIRST MODULE:
    Converting 24-bit color depth RGB frames to YUV color model */

 void Transform_RgbToYuv(FILE *fp,int frame_no)
 {
	 unsigned char *d_frame1 = NULL;

	 int Key_frame_detected = 0;

	 if(frame_no==0)
		 Key_frame_detected = 1;
	 
	 float time;
	 cudaEvent_t start, stop;
     cudaEventCreate(&start);
     cudaEventCreate(&stop);
     
	 if(frame_no==0)
	 {
		frame1_data = (unsigned char*)malloc(sizeof(char)*hp.biSizeImage); 

		cudaMalloc(&d_frame1, sizeof(char)*hp.biSizeImage);

		fseek(fp,sizeof(char)*hhp.bfOffBits,SEEK_SET);
		fread(frame1_data,sizeof(char),hp.biSizeImage, fp); 

		fclose(fp);
		fp = NULL;

		cudaMemcpy(d_frame1, frame1_data, sizeof(char) * hp.biSizeImage, cudaMemcpyHostToDevice);
		
		/* Calculating No of Blocks and Threads */

		dim3 block(16,16);									// 16x16 = 256 threads  // 32x32 = 1024      // No of threads per block
		dim3 grid(hp.biHeight/16, hp.biWidth/16);		    // 40x30 = 1200 blocks  // 20x15 = 300       // No of blocks per grid
															// 1200x256 = 307200	// 1024x300 = 307200
		cudaEventRecord(start, 0);

		rgbToyuv<<<grid,block>>>(d_frame1);

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		printf("\nKernel elapsed time for RgbToYuv conversion:  %3.3f ms \n", time);
		rgbToyuv_time = rgbToyuv_time + time;

		cudaMemcpy(frame1_data, d_frame1, sizeof(char) * hp.biSizeImage, cudaMemcpyDeviceToHost);

		append_char = 'K';

		yuvSeparation(frame1_data);

		cudaFree(d_frame1);

		Key_frame_detected=0;
	 }
	 else
	 {
		 frame2_data = (unsigned char*)malloc(sizeof(char)*hp.biSizeImage);

		 frame1_data2 = (unsigned char*)malloc(sizeof(char)*hp.biSizeImage);

		 frame2_data2 = (unsigned char*)malloc(sizeof(char)*hp.biSizeImage); 

		 cudaMalloc(&d_frame1, sizeof(char)*hp.biSizeImage);

		 fseek(fp,sizeof(char)*hhp.bfOffBits,SEEK_SET);
		 fread(frame2_data,sizeof(char),hp.biSizeImage, fp); 

		 fclose(fp);
		 fp = NULL;

		 cudaMemcpy(d_frame1, frame2_data, sizeof(char) * hp.biSizeImage, cudaMemcpyHostToDevice);
		
		 /* Calculating No of Blocks and Threads */

		 dim3 block(16,16);									// 16x16 = 256 threads  // 32x32 = 1024      // No of threads per block
		 dim3 grid(hp.biHeight/16, hp.biWidth/16);		    // 40x30 = 1200 blocks  // 20x15 = 300       // No of blocks per grid
														    // 1200x256 = 307200	// 1024x300 = 307200
		 cudaEventRecord(start, 0);

		 rgbToyuv<<<grid,block>>>(d_frame1);

		 cudaEventRecord(stop, 0);
		 cudaEventSynchronize(stop);
		 cudaEventElapsedTime(&time, start, stop);
		 printf("\nKernel elapsed time for RgbToYuv conversion:  %3.3f ms \n", time);
		 rgbToyuv_time = rgbToyuv_time + time;

		 cudaMemcpy(frame2_data, d_frame1, sizeof(char) * hp.biSizeImage, cudaMemcpyDeviceToHost);

		 cudaMemcpy(frame1_data2, frame1_data, sizeof(char) * hp.biSizeImage, cudaMemcpyHostToHost);
		 
		 cudaMemcpy(frame2_data2, frame2_data, sizeof(char) * hp.biSizeImage, cudaMemcpyHostToHost);

		 cudaFree(d_frame1);
		 
		 Key_frame_detected = calculatePSNR(frame1_data,frame2_data,frame_no);
	 }

	 if(frame_no>0)
	 {
		if(Key_frame_detected==1)
		{
			 append_char = 'K';
			 yuvSeparation(frame2_data);
			 free(frame1_data);

			 frame1_data = (unsigned char*)malloc(sizeof(char)*hp.biSizeImage);
			 cudaMemcpy(frame1_data, frame2_data, sizeof(char) * hp.biSizeImage, cudaMemcpyHostToHost);
			 free(frame2_data);
		}
		else
		{
			append_char = 'S';
			Simple_Difference(frame1_data2,frame2_data2,frame_no);
			free(frame1_data2);
			free(frame2_data2);
			free(frame2_data);	  
		}
	 } 
}		
 
 /* SECOND MODULE:
    This module is used for MOTION ESTIMATION.
	It identifies dynamically Key-frames and Sub-frames using PSNR */

 int calculatePSNR(unsigned char *frame1_data , unsigned char *frame2_data, int frame_no)
 {
	unsigned char *d_frame1=NULL;
	unsigned char *d_frame2=NULL;
	unsigned char *diff=NULL;
	unsigned char *d_diff=NULL;

	float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

	int sum_y = 0, sum_u = 0, sum_v = 0, val_y = 0, val_u = 0, val_v = 0;
	int i = 0, j = 0, size = 0, size2 = 0;
	double calpsnr = 0, temp = 0;
	double mse_y = 0, mse_u = 0, mse_v = 0, mse = 0;
	
	size2 = hp.biSizeImage/3;

	size = hp.biSizeImage;

	diff = (unsigned char*)malloc(sizeof(char)*hp.biSizeImage);
	cudaFree(diff);
	
	for(i=0;i<size;i++)
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
	printf("\nKernel elapsed time for PSNR:  %3.3f ms \n", time);
	psnr_time = psnr_time + time;

	cudaMemcpy(frame2_data, d_frame2, sizeof(char)*hp.biSizeImage, cudaMemcpyDeviceToHost);
	cudaMemcpy(frame1_data, d_frame1, sizeof(char)*hp.biSizeImage, cudaMemcpyDeviceToHost);
	cudaMemcpy(diff, d_diff, sizeof(char)*hp.biSizeImage, cudaMemcpyDeviceToHost);

	cudaFree(d_frame1);
	cudaFree(d_frame2);
	cudaFree(d_diff);

	unsigned char *y = NULL;
	unsigned char *u = NULL;
	unsigned char *v = NULL;

	y = (unsigned char*)malloc(sizeof(char)*size2);
	u = (unsigned char*)malloc(sizeof(char)*size2);
	v = (unsigned char*)malloc(sizeof(char)*size2);

	for(i=0,j=0;i<size;i++)
	{
		y[j] = diff[i];
		u[j] = diff[++i];
		v[j] = diff[++i];
		j++;
	}
	free(diff);

	for(i=0;i<size2;i++)
	{
		val_y = y[i];
		val_y = val_y * val_y;
		sum_y = sum_y + val_y;

		val_u = u[i];
		val_u = val_u * val_u;
		sum_u = sum_u + val_u;

		val_v = v[i];
		val_v = val_v * val_v;
		sum_v = sum_v + val_v;
	}
	
	mse_y = sum_y;

	mse_u = sum_u;

	mse_v = sum_v;

	mse = (mse_y + mse_u + mse_v)/921600;

	if(mse<=0)
		calpsnr = 100;
	else
	{
		temp = 255*255;
		temp = temp/mse;
		calpsnr = 10 * log10(double(temp));
	}

	printf("\n PSNR : %lf",calpsnr);

	free(y);
	free(u);
	free(v);

	/* THRESHOLD VALUE is considered 40dB */

	if(calpsnr<40)
		return 1;
	else
		return 0;
 }

 /* THIRD MODULE:
    This module takes difference between Sub-frame and Key-frame
	It stores only difference between the frames */

 void Simple_Difference(unsigned char *frame1_data , unsigned char *frame2_data,int frame_no)
{
	unsigned char *d_frame1=NULL;
	unsigned char *d_frame2=NULL;
	unsigned char *diff=NULL;
	unsigned char *d_diff=NULL;

	int i=0,size=hp.biSizeImage;

	float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

	diff = (unsigned char*)malloc(sizeof(char)*hp.biSizeImage);
	cudaFree(diff);
	
	for(i=0;i<size;i++)
	{
		diff[i] = 0;
	}

	cudaMalloc(&d_frame1, sizeof(char)*hp.biSizeImage);
	cudaMalloc(&d_frame2, sizeof(char)*hp.biSizeImage);
	cudaMalloc(&d_diff, sizeof(char)*hp.biSizeImage);

	cudaMemcpy(d_frame1, frame1_data, sizeof(char) * hp.biSizeImage, cudaMemcpyHostToDevice);
	cudaMemcpy(d_frame2, frame2_data, sizeof(char) * hp.biSizeImage, cudaMemcpyHostToDevice);
	cudaMemcpy(d_diff, diff, sizeof(char) * hp.biSizeImage, cudaMemcpyHostToDevice);

	/* Calculating No of Blocks and Threads */

	dim3 block(16,16);									// 16x16 = 256 threads  // 32x32 = 1024      // No of threads per block
	dim3 grid(hp.biHeight/16, hp.biWidth/16);		    // 40x30 = 1200 blocks  // 20x15 = 300       // No of blocks per grid
														// 1200x256 = 307200	// 1024x300 = 307200
	cudaEventRecord(start, 0);

	simpleDiff<<<grid,block>>>(d_frame1,d_frame2,d_diff);

	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
	printf("\nKernel elapsed time for difference:  %3.3f ms \n", time);
	simplediff_time = simplediff_time + time;

	cudaMemcpy(frame2_data, d_frame2, sizeof(char)*hp.biSizeImage, cudaMemcpyDeviceToHost);
	cudaMemcpy(frame1_data, d_frame1, sizeof(char)*hp.biSizeImage, cudaMemcpyDeviceToHost);
	cudaMemcpy(diff, d_diff, sizeof(char)*hp.biSizeImage, cudaMemcpyDeviceToHost);
	
	yuvSeparation(diff);

	free(diff);

	cudaFree(frame2_data);
	cudaFree(d_diff);

	cudaFree(frame1_data);
	cudaFree(d_frame1);
	cudaFree(d_frame2);
 }

 /* This MODULE separates YUV streams and each stream 
    in turn is applied Packbits module  */

void yuvSeparation(unsigned char *diff)
{
	FILE *fptr_rle=NULL;

	unsigned char *rle_data_y=NULL;
	unsigned char *rle_data_u=NULL;
	unsigned char *rle_data_v=NULL;

	int i=0,j=0,size1=0,size2=0;
	
	fptr_rle = fopen(output_path,"wb");
	if(fptr_rle == (FILE *)0) 
	{  
		 printf("File opening error.\n"); 
	}
	else
	{  
		 printf("\n No error in opening"); 
	}

	size1 = hp.biSizeImage/3;
	size2 = hp.biSizeImage;

	rle_data_y = (unsigned char*)malloc(sizeof(char)*size1);
	rle_data_u = (unsigned char*)malloc(sizeof(char)*size1);
	rle_data_v = (unsigned char*)malloc(sizeof(char)*size1);

	for(i=0,j=0;i<size2;i++)
	{
		rle_data_y[j] = diff[i];
		rle_data_u[j] = diff[++i];
		rle_data_v[j] = diff[++i];
		j++;
	}

	fprintf(fptr_rle,"%c",append_char);
	
	launch = clock();
	packbits(rle_data_y,fptr_rle);
	done = clock();
	difference = (double)(done - launch) / CLOCKS_PER_SEC;
	packbit_time = packbit_time + difference;
	
	free(rle_data_y);
	
	launch = clock();
	packbits(rle_data_u,fptr_rle);
	done = clock();
	difference = (double)(done - launch) / CLOCKS_PER_SEC;
	packbit_time = packbit_time + difference;
	
	free(rle_data_u);
	
	launch = clock();
	packbits(rle_data_v,fptr_rle);
	done = clock();
	difference = (double)(done - launch) / CLOCKS_PER_SEC;
	packbit_time = packbit_time + difference;
	
	free(rle_data_v);

	/* For calculating compressed file size */

	fseek(fptr_rle, 0, SEEK_SET);				// seek back to beginning of file
	fseek(fptr_rle, 0, SEEK_END);				// seek to end of file
	compressed_filesize += ftell(fptr_rle);		// get current file pointer

	fclose(fptr_rle);
	fptr_rle = NULL;
}

/* FOURTH MODULE : PACKBITS
   Packbit is revised version of RLE(Run Length Encoding)
   Basic Structure is for consecutive repeated data : Value & Count
   Whenever any value occurs just once, the structure is : Value
   This Stream is written to a binary file with space & dollor($, ) as separators
*/

void packbits(unsigned char *rle, FILE *fptr_rle)
{
	unsigned char value;
	signed int count = 0; 
	int i = 0, size = 0;
	size = hp.biSizeImage/3;

	value = rle[0];

	for(i=0;i<size;i++)
	{
		value = rle[i];

		while(rle[i]==value)
		{
			++count;
			++i;
		}

		if(count==1)
		{
			fprintf(fptr_rle,"%c",value);
			fprintf(fptr_rle," ");
			count = 0;
			i--;
		}
		else
		{
			fprintf(fptr_rle,"%c",value);
			fprintf(fptr_rle,"$");
			fprintf(fptr_rle,"%d",count);
			fprintf(fptr_rle," ");
			count = 0;
			i--;
		}
	}
}

/* FOURTH MODULE : Byte-Level RLE
   This module is used for storing repeated consecutive data in less space
   Basic Structure is : Value & Count	*/

void run_length_encoding(unsigned char *rle, FILE *fptr_rle)
{
	unsigned char value;
	int count = 0, size1 = 0, i = 0;
	size1 = hp.biSizeImage/3;

	value = rle[0];
	count = 0;

	for(i=0;i<size1;i++)
	{
		while(value==rle[i])
		{
			++count;
			++i;
		}
		fprintf(fptr_rle,"%c",value);
		fprintf(fptr_rle," ");
		fprintf(fptr_rle,"%d",count);
		fprintf(fptr_rle," ");

		if(count==307200)
		{	}
		else if(i==size1)
		{	}
		else
		{
			if(value!=rle[i])
			{
				value = rle[i];
				++i;
				count=1;
				if(value==rle[i])
				{
					while(value==rle[i])
					{
						++count;
						++i;
					}
					fprintf(fptr_rle,"%c",value);
					fprintf(fptr_rle," ");
					fprintf(fptr_rle,"%d",count);
					fprintf(fptr_rle," ");
				}
				else
				{
					count=1;
					fprintf(fptr_rle,"%c",value);
					fprintf(fptr_rle," ");
					fprintf(fptr_rle,"%d",count);
					fprintf(fptr_rle," ");
				}
			}
			value = rle[i];
			count = 1;
		}
	}
}