#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>

void vectorAddition() {

    const int dataSize = 10000;
    int* h_a, *h_b, *h_c;
    h_a = (int*)malloc(sizeof(int) * dataSize);
    h_b = (int*)malloc(sizeof(int) * dataSize);
    h_c = (int*)malloc(sizeof(int) * dataSize);

    for (int i = 0; i < dataSize; i++) {
        h_a[i] = i;
        h_b[i] = i;
    }

    auto start = std::chrono::high_resolution_clock::now();

    #pragma acc parallel num_gangs(10) num_workers(4) vector_length(32) copyin(h_a[0:dataSize], h_b[0:dataSize]) copyout(h_c[0:dataSize])
    {
        #pragma acc loop gang
        for (int i = 0; i < dataSize; i++) {
            #pragma acc loop worker
            for (int j = 0; j < 4; j++) {
                h_c[i] = h_a[i] + h_b[i];
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    printf("Processing Time by event: %.1f us\n", elapsedTime);

    //check the result
    for (int i = 0; i < dataSize; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            std::cout << "Error at position " << i << std::endl;
            break;
        }
    }
}

void bgr2gray_blocked() {

    printf("BGR2GRAY with blocking...\n");

    int rows = 1080;
    int cols = 1920;
    int bufferSizeRGB = rows * cols * 3 * sizeof(unsigned char);
    int bufferSizeGray = rows * cols * sizeof(unsigned char);
    // buffers
    unsigned char *input, *output;
    // Allocate host memory
    input = (unsigned char *)malloc(bufferSizeRGB);
    output = (unsigned char *)malloc(bufferSizeGray);
    // Fill input buffer with random values
    for (int i = 0; i < rows * cols * 3; i++)
        input[i] = rand() % 256;

    auto start = std::chrono::high_resolution_clock::now();

    #pragma acc enter data create(output[:rows * cols])
    #pragma acc enter data copyin(input[:rows * cols * 3])
    const int numBlocks = 8;
    const int rowsPerBlock = (rows+(numBlocks-1))/numBlocks;
    for(int block = 0; block < numBlocks; block++) {
        int lower = block*rowsPerBlock;
        int upper = rows < lower+rowsPerBlock ? rows : lower+rowsPerBlock;
        #pragma acc parallel loop gang present(input, output)
        for(int y = lower; y < upper; y++) {
            #pragma acc loop vector
            for(int x = 0; x < cols; x++) {
                int idx = y * cols + x;
                unsigned char b = input[idx * 3];
                unsigned char g = input[idx * 3 + 1];
                unsigned char r = input[idx * 3 + 2];
                output[idx] = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
            }
        }
    }

    #pragma acc exit data delete(input) copyout(output[:rows*cols])

    auto end = std::chrono::high_resolution_clock::now();
    double elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    printf("Processing Time: %.1f us\n", elapsedTime);

    // OpenCV
    printf("OpenCV...\n");
    cv::Mat cv_input(rows, cols, CV_8UC3, input);
    cv::Mat output_opencv;
    cv::cvtColor(cv_input, output_opencv, cv::COLOR_BGR2GRAY);

    // get data from output_opencv
    unsigned char *output_opencv_data = output_opencv.data;

    // Compare results
    printf(" ...comparing the results\n");

    // Check if the OpenCV and CUDA results are the same
    int sum = 0;
    int delta = 0;

    for (int i = 0; i < 5; i++) {
        printf("output_opencv_data[%d]: %d, output[%d]: %d\n", i, output_opencv_data[i], i, output[i]);
        sum += output_opencv_data[i];
        delta += abs(output_opencv_data[i] - output[i]);
    }

    double L2norm = sqrt((double)delta / sum);
    printf(" ...Relative L2 norm: %E\n", L2norm);

    // Check if the L2 norm is within the tolerance
    assert(L2norm < 1e-1);
    printf("Test PASSED\n");

	  // Free memory on host and device
	  free(input); 
    free(output); 
}

void bgr2gray_async() {

    printf("BGR2GRAY with Async...\n");

    int rows = 1080;
    int cols = 1920;
    int bufferSizeRGB = rows * cols * 3 * sizeof(unsigned char);
    int bufferSizeGray = rows * cols * sizeof(unsigned char);
    // buffers
    unsigned char *input, *output;
    // Allocate host memory
    input = (unsigned char *)malloc(bufferSizeRGB);
    output = (unsigned char *)malloc(bufferSizeGray);
    // Fill input buffer with random values
    for (int i = 0; i < rows * cols * 3; i++)
        input[i] = rand() % 256;

    auto start = std::chrono::high_resolution_clock::now();

    #pragma acc enter data create(input[:rows * cols * 3], output[:rows * cols])
    const int numBlocks = 8;
    const int rowsPerBlock = (rows+(numBlocks-1))/numBlocks;
    for(int block = 0; block < numBlocks; block++) {
        int lower = block*rowsPerBlock;
        int upper = rows < lower+rowsPerBlock ? rows : lower+rowsPerBlock;
        
        #pragma acc update device(input[lower*cols:(upper-lower)*cols]) async(block%2)
        #pragma acc parallel loop present(input,output) async(block%2)
        for(int y = lower; y < upper; y++) {
            #pragma acc loop 
            for(int x = 0; x < cols; x++) {
                int idx = y * cols + x;
                unsigned char b = input[idx * 3];
                unsigned char g = input[idx * 3 + 1];
                unsigned char r = input[idx * 3 + 2];
                output[idx] = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
            }
        }
        #pragma acc update self(output[lower*cols:(upper-lower)*cols]) async(block%2)
    }

    #pragma acc wait
    #pragma acc exit data delete(input, output)

    auto end = std::chrono::high_resolution_clock::now();
    double elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    printf("Processing Time: %.1f us\n", elapsedTime);

    // OpenCV
    printf("OpenCV...\n");
    cv::Mat cv_input(rows, cols, CV_8UC3, input);
    cv::Mat output_opencv;
    cv::cvtColor(cv_input, output_opencv, cv::COLOR_BGR2GRAY);

    // get data from output_opencv
    unsigned char *output_opencv_data = output_opencv.data;

    // Compare results
    printf(" ...comparing the results\n");

    // Check if the OpenCV and CUDA results are the same
    int sum = 0;
    int delta = 0;

    for (int i = 0; i < 5; i++) {
        printf("output_opencv_data[%d]: %d, output[%d]: %d\n", i, output_opencv_data[i], i, output[i]);
        sum += output_opencv_data[i];
        delta += abs(output_opencv_data[i] - output[i]);
    }

    double L2norm = sqrt((double)delta / sum);
    printf(" ...Relative L2 norm: %E\n", L2norm);

    // Check if the L2 norm is within the tolerance
    assert(L2norm < 1e-1);
    printf("Test PASSED\n");

	  // Free memory on host and device
    free(input); 
    free(output); 
}

int main() {
    // vectorAddition();
    bgr2gray_blocked();
    bgr2gray_async();
}
