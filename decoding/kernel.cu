
// This is Decoding Program implemented in CUDA (Common Unified Device Architecture).
// Input : Compressed frame stored in Binary file
// BMP images consist 54 bytes header - bmpheader & bmpinfoheader
// 5 modules : YUV to RGB, Motion Estimation, Simple Difference & RLE(byte-level)/Packbit(byte-level)
// Binary file act as input to RLE/Packbit module and the file is expanded
// All frames are reconstructed back from expanded stream using Simple Difference module
// Frames are converted back from YUV to RGB color model
// BMP images are reconstructed back
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

void create_path(int);
void read_frame(int);
void decompressRle(int);
void Transform_YuvToRgb(unsigned char *);
int createBMPimage(unsigned char *);
void createframes_packbits(int);
void createframes_rle(int);
void simpleDifference(int);
void decompressPackbits(int);
void createFile(unsigned char *);

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

 __global__ void simpleDiff(unsigned char* data1, unsigned char* data2)
 {
	 int UniqueBlockIndex = blockIdx.y * gridDim.x + blockIdx.x;

     int UniqueThreadIndex = UniqueBlockIndex * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

     int iindex1 = 3*UniqueThreadIndex;        
     int iindex2 = 3*UniqueThreadIndex;
	 int oindex = 3*UniqueThreadIndex;

	 unsigned char diff_y, diff_u, diff_v;
	 unsigned char data_y1, data_u1, data_v1, data_y2, data_u2, data_v2;

	 data_y1 = data1[iindex1];
	 data_u1 = data1[iindex1+1];
	 data_v1 = data1[iindex1+2];

	 data_y2 = data2[iindex2];
	 data_u2 = data2[iindex2+1];
	 data_v2 = data2[iindex2+2];

	 diff_y = data_y1 + data_y2;

	 diff_u = data_u1 + data_u2;

	 diff_v = data_v1 + data_v2;

	 data2[oindex] = diff_y;
	 data2[oindex+1] = diff_u;
	 data2[oindex+2] = diff_v;
 }

 __global__ void yuvTorgb(unsigned char* image_data)
 {
	 int UniqueBlockIndex = blockIdx.y * gridDim.x + blockIdx.x;

     int UniqueThreadIndex = UniqueBlockIndex * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

     int iindex = 3*UniqueThreadIndex;        
     int oindex = 3*UniqueThreadIndex;

	 unsigned char temp, temp1, temp2;
	 unsigned char data_y, data_u, data_v;

	 data_y = image_data[iindex];
	 data_u = image_data[iindex+1];
	 data_v = image_data[iindex+2];

	 /*
		R = Y + 1.140V
		G = Y - 0.395U - 0.581V
		B = Y + 2.032U
	 */

	 temp = floor(data_y + 1.1400f * (data_v - 128));								  //R
	 temp1 = floor(data_y - 0.3950f * (data_u - 128) - (0.5810f * (data_v - 128)));   //G
	 temp2 = floor(data_y + 2.0320f * (data_u - 128));								  //B

	 //temp = floor(data_y + 1.4075f * (data_v - 128));								  //R
	 //temp1 = floor(data_y - 0.3455f * (data_u - 128) - (0.7169f * (data_v - 128))); //G
	 //temp2 = floor(data_y + 1.7790f * (data_u - 128));							  //B

     image_data[oindex] = temp2;    
     image_data[oindex+1] = temp1;  
     image_data[oindex+2] = temp;   
 } 

 bmpfileheader hhp;
 bmpfileinfoheader hp;
 char appended_char;
 int frame_count=0;
 unsigned char *rle_data=NULL,*pack_bit=NULL,*frame1_data=NULL,*frame2_data=NULL;
 int sizeOfimage=640*480*3;
 char initial_path[100],output_path[100],fixed_path[100],fixed_output_path[100];

int main()
{	
     return 0;
}
	
void decompress(char *input_path)
{
	FILE *fptr=NULL;		
	fptr=fopen("I:\\Input_Frames\\frame_count.txt", "rb");
	if(fptr == (FILE *)0) 
	{  
		printf("File opening error.\n"); 
	}
	else
	{  
		printf("\n No error in opening");
	}	
	 
	fscanf(fptr,"%d",&frame_count);
	fclose(fptr);
	fptr=NULL;

	strcpy(initial_path,input_path);
	strcpy(output_path,"I:\\Input_Frames\\Output_frames\\");
	strcpy(fixed_path,initial_path);
	strcpy(fixed_output_path,output_path);
	create_path(frame_count);
}
void create_path(int frame_count)
{
	 int i=0, no_of_frames_processed=0;
	 char c[5],iextension[5],oextension[5],number_creation[5];
	 strcpy(iextension,".bin");			
	 strcpy(oextension,".BMP");

	 for(i=0;i<frame_count;i++)
	 {
		 itoa(i,c,10);
		 strcpy(number_creation,c);

		 strcat(initial_path,"\\");
		 strcat(initial_path,number_creation);
		 strcat(initial_path,iextension);
		 printf("\n%s",initial_path);
		 getch();
		 strcat(output_path,number_creation);
		 strcat(output_path,oextension);

		 decompressPackbits(i);
		 no_of_frames_processed++;
		
		 strcpy(initial_path,fixed_path);
		 strcpy(output_path,fixed_output_path);
	 }
}

void decompressPackbits(int frame_no)
{
	FILE *fp1 = NULL;

	unsigned char *pk_y=NULL,*pk_u=NULL,*pk_v=NULL;

	int size = 0, i = 0, j = 0, k = 0, size1 = 0;

	unsigned char value=NULL,value2=NULL,value3=NULL;
	int count = 0;

	size = 921600;
	size1 =307200;

	fp1 = fopen(initial_path,"rb");
	if(fp1 == (FILE *)0) 
	{  
		 printf("File opening error.\n"); 
	}

	pack_bit = (unsigned char*)malloc(sizeof(char)*size);

	i = 0;

	fscanf(fp1,"%c",&appended_char);

	if(frame_no>0)
	{
		if(appended_char=='K')
		{
			free(frame1_data);
		}
	}

	while(!feof(fp1))
	{
		fscanf(fp1,"%c",&value);
		fscanf(fp1,"%c",&value2);

		if(value2=='$')
		{
			fscanf(fp1,"%d",&count);
			fscanf(fp1,"%c",&value3);

			while(count!=0)
			{
				pack_bit[i]=value;
				i++;
				count--;
			}
		}
		else
		{	
			pack_bit[i]=value;
			i++;
		}
	}

	fclose(fp1);
	fp1 = NULL;
	pk_y = (unsigned char*)malloc(sizeof(char)*size1);
	pk_u = (unsigned char*)malloc(sizeof(char)*size1);
	pk_v = (unsigned char*)malloc(sizeof(char)*size1);

    for(i=0;i<size1;i++)
	{
		pk_y[i] = pack_bit[i];
	}

	for(j=0,i=size1;i<(size1*2);i++,j++)
	{
		pk_u[j] = pack_bit[i];
	}

	size1 = size1*2;
	for(k=0,i=size1;i<size;i++,k++)
	{
		pk_v[k] = pack_bit[i];
	}

	size1 =307200;
	
	free(pack_bit);
	
	pack_bit = (unsigned char*)malloc(sizeof(char)*size);

	for(i=0,j=0;i<size;i++,j++)
	{
		pack_bit[i] = pk_y[j];
		pack_bit[++i] = pk_u[j];
		pack_bit[++i] = pk_v[j];
	}

	free(pk_y);
	free(pk_u);
	free(pk_v);

	createframes_packbits(frame_no);
}

void createframes_packbits(int frame_no)
{
	 int size=921600;
	 
	 if(appended_char=='K')
	 {
		frame1_data = (unsigned char*)malloc(sizeof(char)*size);
		cudaMemcpy(frame1_data, pack_bit,sizeof(char) * size, cudaMemcpyHostToHost);
		Transform_YuvToRgb(pack_bit);
		free(pack_bit);
	 }
	 else
	 {
		 frame2_data = (unsigned char*)malloc(sizeof(char)*size);
		 cudaMemcpy(frame2_data, pack_bit, sizeof(char) * size, cudaMemcpyHostToHost);
		 free(pack_bit);
		 simpleDifference(frame_no);
	 }
}

void simpleDifference(int frame_no)
{
	unsigned char *d_frame1=NULL,*d_frame2=NULL;
	
	int size = 0;

	size = 921600;

	float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	
	cudaMalloc(&d_frame1, sizeof(char)*size);
	cudaMalloc(&d_frame2, sizeof(char)*size);

	cudaMemcpy(d_frame1, frame1_data, sizeof(char) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_frame2, frame2_data, sizeof(char) * size, cudaMemcpyHostToDevice);

	dim3 block(16,16);									// 16x16 = 256 threads  // 32x32 = 1024      // No of threads per block
	dim3 grid(480/16, 640/16);							// 40x30 = 1200 blocks  // 20x15 = 300       // No of blocks per grid
														// 1200x256 = 307200	// 1024x300 = 307200
	cudaEventRecord(start, 0);

	simpleDiff<<<grid,block>>>(d_frame1,d_frame2);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("\nKernel elapsed time for difference:  %3.3f ms \n", time);

	cudaMemcpy(frame1_data, d_frame1, sizeof(char) * size, cudaMemcpyDeviceToHost);
	cudaMemcpy(frame2_data, d_frame2, sizeof(char) * size, cudaMemcpyDeviceToHost);

	cudaFree(d_frame1);
	cudaFree(d_frame2);

	Transform_YuvToRgb(frame2_data);
	free(frame2_data);
}

void createFile(unsigned char *data)
{
	FILE *fptr_yuv;

	 fptr_yuv = fopen(output_path,"wb+");
	 if(fptr_yuv == (FILE *)0) 
	 {  
		 printf("File opening error.\n"); 
	 }
	 
	 int i=0,j=0;
	 for(i=0,j=0;i<921600;i++)
	 {
		 fprintf(fptr_yuv,"%u ",data[i]);
		 fprintf(fptr_yuv,"%u ",data[++i]);
		 fprintf(fptr_yuv,"%u ",data[++i]);
		 j++;
	 }

	 fclose(fptr_yuv);
	 fptr_yuv = NULL;
}

void Transform_YuvToRgb(unsigned char *imagedata)
 {
	 unsigned char *d_imagedata;
	 float time;

	 cudaEvent_t start, stop;
     cudaEventCreate(&start);
     cudaEventCreate(&stop);

	 cudaMalloc(&d_imagedata, sizeof(char)*sizeOfimage);
	 cudaMemcpy(d_imagedata, imagedata, sizeof(char) * sizeOfimage, cudaMemcpyHostToDevice);
		
	 dim3 block(16,16);									// 16x16 = 256 threads  // 32x32 = 1024      // No of threads per block
	 dim3 grid(480/16, 640/16);							// 40x30 = 1200 blocks  // 20x15 = 300       // No of blocks per grid
														// 1200x256 = 307200	// 1024x300 = 307200
	 cudaEventRecord(start, 0);

	 yuvTorgb<<<grid,block>>>(d_imagedata);

	 cudaEventRecord(stop, 0);
	 cudaEventSynchronize(stop);
	 cudaEventElapsedTime(&time, start, stop);
	 printf("\nKernel elapsed time for YuvToRgb conversion:  %3.3f ms \n", time);

	 cudaMemcpy(imagedata, d_imagedata, sizeof(char) * sizeOfimage, cudaMemcpyDeviceToHost);
	 cudaFree(d_imagedata);

	 createBMPimage(imagedata);
}
		
int createBMPimage(unsigned char *imageData)
{
	FILE* filePtr=NULL;

	// Open file for writing binary mode
	filePtr = fopen(output_path,"wb");
	if (!filePtr)
	{
		return 0;
	}

	// Define the bitmap file header
	hhp.bfSize = sizeOfimage + 54;
	hhp.bfType[0] ='B';
	hhp.bfType[1] ='M';
	hhp.bfReserved1 = 0;
	hhp.bfReserved2 = 0;
	hhp.bfOffBits = 54;

	// Define the bitmap information header
	hp.biSize = 40;
	hp.biWidth = 640;                        
	hp.biHeight = 480;                       
	hp.biPlanes = 1;
	hp.biBitCount = 24;                      
	hp.biCompression = 0;                    
	hp.biSizeImage = sizeOfimage;            
	hp.biXPelsPerMeter = 3780;
	hp.biYPelsPerMeter = 3780;
	hp.biClrUsed = 0;
	hp.biClrImportant = 0;

	// Write the bitmap file header
	fwrite(&hhp, 1, sizeof(hhp), filePtr);

	// Write the bitmap info header
	fwrite(&hp, 1, sizeof(hp), filePtr);

	// Write the image data
	fwrite(imageData, 1, hp.biSizeImage, filePtr);

	// Close our file
	fclose(filePtr);
	filePtr=NULL;

	// Success
	return 1;
}

