//Kernel file for applying histogram equalisation on an RGB image
//both 8 and 16 bit images have been used

//8 bit image histogram with specified bins
kernel void get_hist_8(global uchar* image, global uint* H)
{
	uint global_id = get_global_id(0);
	atomic_inc(&H[image[global_id]]); //input image as bin index
}

//16bit image histogram
//sum of elements should equal pixels
kernel void get_hist_16(global const ushort* image, global uint* H)
{
	uint global_id = get_global_id(0);
	atomic_inc(&H[image[global_id]]); //input image as bin index for 16bit
}

//8bit histogram using local memory
kernel void get_hist_8LC(global const uchar* image, global uint* H, local uint* H_local, const uint image_elements)
{
	uint global_id = get_global_id(0); 
	int local_id = get_local_id(0); //set local hist to 0
	
	if (local_id < 256) H_local[local_id] = 0;
	
	barrier(CLK_LOCAL_MEM_FENCE); //wait for local threads to finish
	
	//local histogram computation
	//bin index taken from input image
	if (global_id < image_elements) atomic_inc(&H_local[image[global_id]]);
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	//local to global histogram
	if (local_id < 256) atomic_add(&H[local_id], H_local[local_id]);
}

//cumulative histogram
//last element = total numb of pixels
kernel void get_c_hist(global const uint* H, global uint* CH, const int bin_count)
{
	int global_id = get_global_id(0);
	
	for (int i = global_id + 1; i < bin_count && global_id < bin_count; i++)
	{
		atomic_add(&CH[i], H[global_id] / 3);
	}
}

//cumulative histogram using hillis and steele scan and local memory
//8 bit image, local elements should equal 256
//16 bit needs helper kernels
//last element in the cumulative histogram should equal the total num of pixels
kernel void get_chist_HS(global const uint* H, global uint* CH, local uint* H_local, local uint* CH_local)
{
	int global_id = get_global_id(0);
	int local_id = get_local_id(0);
	local uint* swap_value;//enables buffer swap 
	
	H_local[local_id] = H[global_id]; //cache histogram values from global to local
	
	barrier(CLK_LOCAL_MEM_FENCE); //waiting for local threads to finish
	
	for (int i = 1; i < get_local_size(0); i *= 2){
		if (local_id >= i) CH_local[local_id] = H_local[local_id] + H_local[local_id - i];
		else
			CH_local[local_id] = H_local[local_id];
			
		barrier(CLK_LOCAL_MEM_FENCE);
		
		//buffer swap
		swap_value = CH_local;
		CH_local = H_local;
		H_local = swap_value;
		}
		
	
	
	CH[global_id] = H_local[local_id] / 3;
	}

//helper kernel with scanned block sums
kernel void get_B_S(global const uint* CH, global uint* BS, const uint local_elements)
{
	int global_id = get_global_id(0);
	
	
	BS[global_id] = CH[(global_id + 1) * local_elements - 1];
	}
	
//performing an exclusive scan
kernel void get_scanned_BS_1(global const uint* BS, global uint* BS_scanned)
{
	int global_id = get_global_id(0);
	
	
	int size = get_global_size(0);
	
	for (int i = global_id + 1; i < size && global_id < size; i++)
	{
		atomic_add(&BS_scanned[i], BS[global_id]);
	}
}

//exclusive scan using Blelloch method
kernel void get_scanned_BS_2(global uint* BS)
{
	int global_id = get_global_id(0);
	int size = get_global_size(0);
	int temp_value; //used as a temp value
	
	//up-sweep
	for (int i = 1; i < size; i *= 2)
	{
		if (((global_id + 1) % (i * 2)) == 0) BS[global_id] += BS[global_id - i];

		
		
	barrier(CLK_GLOBAL_MEM_FENCE); }
	
	//down sweep
	if (global_id == 0) BS[size - 1] = 0;

	
	
	barrier(CLK_GLOBAL_MEM_FENCE);
	
	
	
	for (int i = size / 2; i > 0; i /= 2)
	{
		if (((global_id + 1) % (i * 2)) == 0)
		{
			temp_value = BS[global_id];
			
			
			BS[global_id] += BS[global_id - i];
			
			
			BS[global_id - i] = temp_value;
		}
		
		
		barrier(CLK_GLOBAL_MEM_FENCE);	}	
}


//complete c_hist (adding block sums to blocks)
kernel void get_complete_chist(global const uint* BS_scanned, global uint* CH)
{
	CH[get_global_id(0)] += BS_scanned[get_group_id(0)];
}

//normalised c-hist as an LUT
kernel void get_LUT(global uint* CH, global uint* LUT, const int bin_count, const int pixel_count)
{
	int global_id = get_global_id(0);
	
	//ulong is needed so it doesnt overflow past the int
	if (global_id < bin_count) LUT[global_id] = ((ulong)CH[global_id] * (bin_count - 1)) / pixel_count;
}

//getting the 8bit image output
kernel void get_Output8(global const uchar* input_image, global const uint* LUT, global uchar* output_image)
{
	uint global_id = get_global_id(0);
	output_image[global_id] = LUT[input_image[global_id]]; //getting the output image from the LUT value from the altered input image
}

//getting the 16bit image output
kernel void get_Output16(global const ushort* input_image, global const uint* LUT, global ushort* output_image)
{
	uint global_id = get_global_id(0);
	output_image[global_id] = LUT[input_image[global_id]]; //getting the output image from the 16bit LUT values from the altered input image
}