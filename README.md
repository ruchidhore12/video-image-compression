# Video-Image-Compression-Using-CUDA-and-NVIDIA-s-GPU
This project aimed to compress the video for following reasons:
1. To reduce the space required for the purpose of storage.
2. To achieve the compression in the range of 10-30% to save transmission bandwidth.
3. To implement modules in parallel wherever possible to minimize the execution time.
4. To maintain the quality of the video after compression.

This repository consists of compression, decompression and PSNR calculation algorithms

## In compression system the following steps are applied on pixel data:
1. RGB to YUV conversion
2. Motion estimation
3. Simple difference
4. Run length encoding / pack bits.
After this compression system will generate compressed data of video as output.

## This compressed data will act as the input for decompression system and again following steps are performed on the data.
1. Reverse of pack bits
2. Simple difference
3. YUV to RGB
And output of this will be the video frames and again video is reconstructed.
