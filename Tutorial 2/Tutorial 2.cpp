
/*
The project was developed using Tutorial 2 as a foundation and was appropriately edited and built upon to carry out the task.
Once the solution has been built it can then run.  The implementation can be run on colour, greyscale and monchrome images and on both 8bit and 16 bit
images. Histogram based on local memory has been implemented as has the Hillis and Steele and the Blelloch scans that were used in the workshops.
Images in the folder that have been tested are ‘test.ppm’, ‘test_large.ppm’, ‘monochrome1.ppm’ and ‘colour1.ppm’. The kernels that have been implemented
are the basic histogram, Cumulative histogram, histogram using local memory, C-hist using HS scan, a helper kernel to obtain block sums,
an exclusive scan, an exclusive scan using Blelloch, a complete histogram, obtaining the look up tables and kernels to output the images.
*/

#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;

int main(int argc, char** argv)
{
	// Part 1 - handle command line options such as device selection
	int platform_id = 0;
	int device_id = 0;
	int mode_id = 0;
	string image_filename = "test.ppm";

	for (int i = 1; i < argc; i++)
	{
		// run the program according to the command line options
		if (strcmp(argv[i], "-l") == 0)
			std::cout << ListPlatformsDevices();
		else if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1)))
			platform_id = atoi(argv[++i]);
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1)))
			device_id = atoi(argv[++i]);
		else if ((strcmp(argv[i], "-m") == 0) && (i < (argc - 1)))
			mode_id = atoi(argv[++i]);
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1)))
			image_filename = argv[++i];
		else if (strcmp(argv[i], "-h") == 0)
		{
			// print help info to the console
			std::cerr << "Application usage:" << std::endl;
			std::cerr << "  __ : (no option specified) run with default input image file in default run mode on 1st device of 1st platform" << std::endl;
			std::cerr << "  -l : list all platforms, devices, and run modes, and then run as no options specified if no other options" << std::endl;
			std::cerr << "  -p : select platform" << std::endl;
			std::cerr << "  -d : select device" << std::endl;
			std::cerr << "  -m : select run mode" << std::endl;
			std::cerr << "  -f : specify input image file" << std::endl;
			std::cerr << "       ATTENTION: 1. \"test.ppm\" is default" << std::endl;
			std::cerr << "                  2. Please select a PPM image file (8-bit/16-bit RGB)" << std::endl;
			std::cerr << "                  3. The specified image should be put under the folder \"images\"" << std::endl;
			std::cerr << "  -h : print this message" << std::endl;
			return 0;
		}
	}

	string image_path = "images/" + image_filename;
	//the try from the exception handling
	try
	{
		// loading image
		CImg<unsigned short> input_image(image_path.c_str()); // reads data from the image file
		CImg<unsigned char> input_image_8;

		size_t input_image_elements = input_image.size(); // number of elements
		size_t input_image_size = input_image_elements * sizeof(unsigned short); // size in bytes
		int input_image_width = input_image.width(), input_image_height = input_image.height();

		// image bin numbers
		int bin_count = input_image.max() <= 255 ? 256 : 65536;


		float scale = 1.0f; // image output scale


		CImgDisplay input_image_display;

		// detects image using bin count - either 8bit in the if statement or 16 bit outside of it
		if (bin_count == 256)
		{
			input_image_8.load(image_path.c_str());
			input_image_size = input_image_elements * sizeof(unsigned char);

			//displays image
			input_image_display.assign(CImg<unsigned char>(input_image_8), "Input image 8bit");
		}
		else

			//displays the image but for 16bit instead of 8bit
			input_image_display.assign(CImg<unsigned short>(input_image), "Input image 16bit");

		// Part 3 - host operations
		// 3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		std::cout << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands to the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		// 3.2 Load & build the device code
		cl::Program::Sources sources;
		AddSources(sources, "kernels/my_kernels.cl");
		cl::Program program(context, sources);


		// build and debug the kernel code
		try
		{
			program.build();
		}
		catch (const cl::Error& err)
		{
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		// Part 4 - memory allocation
		typedef unsigned int standard; //use unsigned int to avoid overflow
		std::vector<standard> H(bin_count, 0); //vector to store hist
		size_t H_elements = H.size();
		size_t H_size = H_elements * sizeof(standard);

		std::vector<standard> CH(H_elements, 0); //vector to store c-hist
		size_t CH_elements = CH.size();
		size_t CH_size = CH_elements * sizeof(standard);


		//number of local elements is taken from the num of bins (8bit)
		//8bit only needs one workgroup
		size_t local_elements_8 = 256;
		size_t local_size_8 = local_elements_8 * sizeof(standard);

		//adjusts the length of global elements of the histogram kernel for an 8-bit image;
		//this is to try and ensure that the global size is a multiple of the local size for the padding 
		size_t kernel1_global_elements_8 = input_image_elements;

		size_t kernel1_global_elements_8_padding = kernel1_global_elements_8 % local_elements_8;
		if (kernel1_global_elements_8_padding)
			kernel1_global_elements_8 += (local_elements_8 - kernel1_global_elements_8_padding);

		//16bit image size segment
		size_t local_elements_16 = cl::Kernel(program, "get_chist_HS").getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(context.getInfo<CL_CONTEXT_DEVICES>()[0]);

		//obtain max workgroup size
		size_t local_size_16 = local_elements_16 * sizeof(standard);
		size_t group_count = bin_count == 256 ? 1 : CH_elements / local_elements_16;




		mode_id = (mode_id == 1 && bin_count == 65536 && (group_count & (group_count - 1))) ? 0 : mode_id;//obtaining mode id as to either use a basic version or a more optimised version


		//adjusting the length of global elements for 16bit
		// trying to get the global size to be a multiple of the local size for the padding
		size_t kernel2_global_elements_16 = H_elements;

		size_t kernel2_global_elements_16_padding = kernel2_global_elements_16 % local_elements_16;
		if (kernel2_global_elements_16_padding)
			kernel2_global_elements_16 += (local_elements_16 - kernel2_global_elements_16_padding);

		//using a vector to store the block sums
		std::vector<standard> BS(group_count, 0);
		size_t BS_size = BS.size() * sizeof(standard);

		//vector is equal to the num of workgroups to scan the block sums
		std::vector<standard> BS_scanned(group_count, 0);

		// size in bytes
		size_t BS_scanned_size = BS_scanned.size() * sizeof(standard);
		std::vector<standard> LUT(CH_elements, 0);//using a vector to store the LUT

		//a LUT for a c-hist
		size_t LUT_size = LUT.size() * sizeof(standard);

		// Part 5 - device operations
		// device - buffers
		cl::Buffer buffer_input_image(context, CL_MEM_READ_ONLY, input_image_size);

		//histogram buffer
		cl::Buffer buffer_H(context, CL_MEM_READ_WRITE, H_size);

		//c-hist buffer
		cl::Buffer buffer_CH(context, CL_MEM_READ_WRITE, CH_size);

		//block sum buffer
		cl::Buffer buffer_BS(context, CL_MEM_READ_WRITE, BS_size);

		//scanned block sum buffer
		cl::Buffer buffer_BS_scanned(context, CL_MEM_READ_WRITE, BS_scanned_size);

		// LUT buffer
		cl::Buffer buffer_LUT(context, CL_MEM_READ_WRITE, LUT_size);

		// LUT buffer
		cl::Buffer buffer_output_image(context, CL_MEM_READ_WRITE, input_image_size);

		// 5.1 Copy the image to and initialise other arrays on device memory
		cl::Event input_image_event, H_input_event, CH_input_event, BS_input_event, BS_scanned_input_event, LUT_input_event;

		if (bin_count == 256)
			queue.enqueueWriteBuffer(buffer_input_image, CL_TRUE, 0, input_image_size, &input_image_8.data()[0], NULL, &input_image_event);
		else
			queue.enqueueWriteBuffer(buffer_input_image, CL_TRUE, 0, input_image_size, &input_image.data()[0], NULL, &input_image_event);

		//histogram buffer 0
		queue.enqueueFillBuffer(buffer_H, 0, 0, H_size, NULL, &H_input_event);

		//c-hist buffer 0
		queue.enqueueFillBuffer(buffer_CH, 0, 0, CH_size, NULL, &CH_input_event);

		//LUT buffer 0
		queue.enqueueFillBuffer(buffer_LUT, 0, 0, LUT_size, NULL, &LUT_input_event);

		if (bin_count == 65536)
		{
			queue.enqueueFillBuffer(buffer_BS, 0, 0, BS_size, NULL, &BS_input_event); //0 block sum buffer

			if (mode_id == 0)
				queue.enqueueFillBuffer(buffer_BS_scanned, 0, 0, BS_scanned_size, NULL, &BS_scanned_input_event); //0 scanned block sum buffer
		}

		// 5.2 Setup and execute the kernel
		cl::Kernel kernel1, kernel2, kernel2_helper1, kernel2_helper2, kernel2_helper3;

		//use am optimised version if any are available
		if (mode_id == 0 || mode_id == 1)
		{
			if (bin_count == 256)
			{
				std::cout << "Using optimised histogram and cumulative histogram kernels" << std::endl;

				kernel1 = cl::Kernel(program, "get_hist_8LC");
				//get a hist with a specified number of bins


				kernel2 = cl::Kernel(program, "get_chist_HS");
				//get a c-hist


				kernel1.setArg(2, cl::Local(local_size_8));
				//local memory size for a local histogram


				kernel1.setArg(3, (standard)input_image_elements);


				kernel2.setArg(2, cl::Local(local_size_8));


				kernel2.setArg(3, cl::Local(local_size_8));
				//local memory size for a c-hist
			}

			else
			{
				std::cout << "Using optimised cumulative histogram kernel";

				//get a histogram with a specified number of bins;
				//only for 8bit

				kernel1 = cl::Kernel(program, "get_hist_16");

				kernel2 = cl::Kernel(program, "get_chist_HS"); //get a starting c-hist
				kernel2_helper1 = cl::Kernel(program, "get_B_S"); //get block sums of a starting c-hist

				if (mode_id == 0 || mode_id == 2)
				{
					std::cout << std::endl;

					kernel2_helper2 = cl::Kernel(program, "get_scanned_BS_1"); //get scanned block sums

					kernel2_helper2.setArg(1, buffer_BS_scanned);
				}
				else
				{
					std::cout << " including a helper kernel different from Fast Mode 1" << std::endl;

					kernel2_helper2 = cl::Kernel(program, "get_scanned_BS_2"); //get scanned block sums
				}

				kernel2_helper3 = cl::Kernel(program, "get_complete_chist"); //get a complete c-hist


				kernel2.setArg(2, cl::Local(local_size_16)); //set local memory for local hist
				kernel2.setArg(3, cl::Local(local_size_16)); //set local memory for a c-hist

				kernel2_helper1.setArg(0, buffer_CH);

				kernel2_helper1.setArg(1, buffer_BS);

				kernel2_helper1.setArg(2, (int)local_elements_16);

				kernel2_helper2.setArg(0, buffer_BS);

				if (mode_id == 0 || mode_id == 2)
					kernel2_helper3.setArg(0, buffer_BS_scanned);
				else
					kernel2_helper3.setArg(0, buffer_BS);

				kernel2_helper3.setArg(1, buffer_CH);
			}
		}

		//use basic version
		else
		{
			std::cout << "Using basic kernels" << std::endl;

			//get a histogram with a specified number of bins
			if (bin_count == 256)
				kernel1 = cl::Kernel(program, "get_hist_8");
			else
				kernel1 = cl::Kernel(program, "get_hist_16");

			kernel2 = cl::Kernel(program, "get_chist"); //get a c-hist

			kernel2.setArg(2, bin_count);
		}

		std::cout << "----------------------------------" << std::endl;

		cl::Kernel kernel3 = cl::Kernel(program, "get_LUT"); //get a LUT froma normalised c-hist
		cl::Kernel kernel4;

		//get the output image using the lut
		if (bin_count == 256)
			kernel4 = cl::Kernel(program, "get_Output8");
		else
			kernel4 = cl::Kernel(program, "get_Output16");

		kernel1.setArg(0, buffer_input_image);

		kernel1.setArg(1, buffer_H);

		kernel2.setArg(0, buffer_H);

		kernel2.setArg(1, buffer_CH);

		kernel3.setArg(0, buffer_CH);

		kernel3.setArg(1, buffer_LUT);

		kernel3.setArg(2, bin_count);

		kernel3.setArg(3, input_image_width * input_image_height);


		kernel4.setArg(0, buffer_input_image);

		kernel4.setArg(1, buffer_LUT);

		kernel4.setArg(2, buffer_output_image);

		cl::Event kernel1_event, kernel2_event, kernel2_helper1_event, kernel2_helper2_event, kernel2_helper3_event, kernel3_event, kernel4_event;

		if ((mode_id == 0 || mode_id == 1) && bin_count == 256)
			queue.enqueueNDRangeKernel(kernel1, cl::NullRange, cl::NDRange(kernel1_global_elements_8), cl::NDRange(local_elements_8), NULL, &kernel1_event);
		else
			queue.enqueueNDRangeKernel(kernel1, cl::NullRange, cl::NDRange(input_image_elements), cl::NullRange, NULL, &kernel1_event);

		if ((mode_id == 0 || mode_id == 1) && bin_count == 65536)
		{
			queue.enqueueNDRangeKernel(kernel2, cl::NullRange, cl::NDRange(kernel2_global_elements_16), cl::NDRange(local_elements_16), NULL, &kernel2_event);
			queue.enqueueNDRangeKernel(kernel2_helper1, cl::NullRange, cl::NDRange(group_count), cl::NullRange, NULL, &kernel2_helper1_event);

			if (mode_id == 0)
				queue.enqueueNDRangeKernel(kernel2_helper2, cl::NullRange, cl::NDRange(group_count), cl::NullRange, NULL, &kernel2_helper2_event);
			else
				queue.enqueueNDRangeKernel(kernel2_helper2, cl::NullRange, cl::NDRange(group_count), cl::NDRange(group_count), NULL, &kernel2_helper2_event);

			queue.enqueueNDRangeKernel(kernel2_helper3, cl::NullRange, cl::NDRange(kernel2_global_elements_16), cl::NDRange(local_elements_16), NULL, &kernel2_helper3_event);
		}
		else if ((mode_id == 0 || mode_id == 1) && bin_count == 256)
			queue.enqueueNDRangeKernel(kernel2, cl::NullRange, cl::NDRange(H_elements), cl::NDRange(local_elements_8), NULL, &kernel2_event);
		else
			queue.enqueueNDRangeKernel(kernel2, cl::NullRange, cl::NDRange(H_elements), cl::NullRange, NULL, &kernel2_event);

		queue.enqueueNDRangeKernel(kernel3, cl::NullRange, cl::NDRange(CH_elements), cl::NullRange, NULL, &kernel3_event);

		queue.enqueueNDRangeKernel(kernel4, cl::NullRange, cl::NDRange(input_image_elements), cl::NullRange, NULL, &kernel4_event);

		//print info to the console and display the output image
		queue.enqueueReadBuffer(buffer_H, CL_TRUE, 0, H_size, &H[0]);
		queue.enqueueReadBuffer(buffer_CH, CL_TRUE, 0, CH_size, &CH[0]);

		queue.enqueueReadBuffer(buffer_LUT, CL_TRUE, 0, LUT_size, &LUT[0]);
		std::cout << "H = " << H << std::endl;
		std::cout << "----------------------------------" << std::endl;
		std::cout << "CH = " << CH << std::endl;
		std::cout << "----------------------------" << std::endl;
		if ((mode_id == 0 || mode_id == 1) && bin_count == 65536)
		{
			queue.enqueueReadBuffer(buffer_BS, CL_TRUE, 0, BS_size, &BS[0]);
			std::cout << "BS = " << BS << std::endl;
			std::cout << "--------------------------------------" << std::endl;
			if (mode_id == 0)
			{
				queue.enqueueReadBuffer(buffer_BS_scanned, CL_TRUE, 0, BS_scanned_size, &BS_scanned[0]);
				std::cout << "BS_scanned = " << BS_scanned << std::endl;
				std::cout << "--------------------------------" << std::endl;
			}
		}
		std::cout << "LUT = " << LUT << std::endl;
		std::cout << "-------------------------" << std::endl;

		cl::Event output_image_event;

		CImgDisplay output_image_display;

		if (bin_count == 256)
		{
			vector<unsigned char> output_buffer_8(input_image_elements); //unsigned char can be used for data from 8bit buffer
			queue.enqueueReadBuffer(buffer_output_image, CL_TRUE, 0, output_buffer_8.size() * sizeof(unsigned char), &output_buffer_8.data()[0], NULL, &output_image_event);
			CImg<unsigned char> output_image_8(output_buffer_8.data(), input_image_width, input_image_height, input_image.depth(), input_image.spectrum());


			//output the 8bit image and resize if needed
			output_image_display.assign(output_image_8.resize((int)(input_image_width * scale), (int)(input_image_height * scale)), "Output image (8-bit)");
		}
		else
		{
			vector<unsigned short> output_buffer_16(input_image_elements); //unsigned short is required for 16bit buffer
			queue.enqueueReadBuffer(buffer_output_image, CL_TRUE, 0, output_buffer_16.size() * sizeof(unsigned short), &output_buffer_16.data()[0], NULL, &output_image_event);
			CImg<unsigned short> output_image_16(output_buffer_16.data(), input_image_width, input_image_height, input_image.depth(), input_image.spectrum());


			//output 16bit image and resize if needed
			output_image_display.assign(output_image_16.resize((int)(input_image_width * scale), (int)(input_image_height * scale)), "Output image (16-bit)");
		}
		cl_ulong total_upload_time = input_image_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - input_image_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()
			+ H_input_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - H_input_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()
			+ CH_input_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - CH_input_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()
			+ LUT_input_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - LUT_input_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();


		//total upload time of input vectors
		cl_ulong kernel1_time = kernel1_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernel1_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();


		//histogram kernel execution time
		cl_ulong kernel2_time = kernel2_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernel2_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();


		//c-hist kernel execution time
		cl_ulong total_kernel_time = kernel1_time + kernel2_time
			+ kernel3_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernel3_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()
			+ kernel4_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernel4_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();


		//total execution time of kernels
		cl_ulong output_image_download_time = output_image_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - output_image_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

		if ((mode_id == 0 || mode_id == 1) && bin_count == 65536)
		{
			cl_ulong kernel2_helper_time = kernel2_helper1_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernel2_helper1_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()
				+ kernel2_helper2_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernel2_helper2_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()
				+ kernel2_helper3_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernel2_helper3_event.getProfilingInfo<CL_PROFILING_COMMAND_START>(); //c-hist extra kernel execution time


			total_upload_time += (BS_input_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - BS_input_event.getProfilingInfo<CL_PROFILING_COMMAND_START>());

			if (mode_id == 0)
				total_upload_time += (BS_scanned_input_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - BS_scanned_input_event.getProfilingInfo<CL_PROFILING_COMMAND_START>());

			kernel2_time += kernel2_helper_time;

			//adds the helper kernel execution time so that the entire execution time is taken into account
			total_kernel_time += kernel2_helper_time;
		}

		//execution times in microseconds, so total time is divided by 1000
		std::cout << " Memory transfer time: " << total_upload_time / 1000 << "ms" << std::endl;
		std::cout << " ---------------------------------------------------------" << std::endl;
		std::cout << " Kernel execution time: " << total_kernel_time / 1000 << "ms" << std::endl;
		std::cout << " ---------------------------------------------------------" << std::endl;
		std::cout << " Histogram kernel execution time: " << kernel1_time / 1000 << "ms" << std::endl;
		std::cout << " ---------------------------------------------------------" << std::endl;
		std::cout << " Cumulative histogram kernel execution time: " << kernel2_time / 1000 << "ms" << std::endl;
		std::cout << " ---------------------------------------------------------" << std::endl;
		std::cout << " Program execution time: " << (total_upload_time + total_kernel_time + output_image_download_time) / 1000 << "ms" << std::endl;

		//keeps the input and output images open while they are not closed and the escape key hasnt been pressed
		while (!input_image_display.is_closed() && !output_image_display.is_closed()
			&& !input_image_display.is_keyESC() && !output_image_display.is_keyESC())
		{
			input_image_display.wait(1);
			output_image_display.wait(1);
		}
	}
	//catches errors and throws exceptions
	catch (const cl::Error& e)
	{
		std::cerr << "OpenCL - ERROR: " << e.what() << ", " << getErrorString(e.err()) << std::endl;
	}
	catch (CImgException& e)
	{
		std::cerr << "CImg - ERROR: " << e.what() << std::endl;
	}

	return 0;
}