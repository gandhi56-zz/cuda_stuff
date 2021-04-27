
#include <iostream>
#include <cstdlib>
#include <ios>
#include <stdio.h>
#include <cuda_runtime.h>
#include "ImageBlur.h"
#include <stdexcept>

#include <boost/gil.hpp>
#include <boost/gil/io/io.hpp>
#include <boost/gil/extension/io/jpeg.hpp>
#include <fstream>

int ImageBlur::run(void){
    // read image
    const char* filePath = "assets/sample.jpg";
    printf("Reading image %s...\n", filePath);
    
    try{
        std::fstream fileStream;
        namespace bg = boost::boost::gil;
        bg::image_read_settings<bg::jpeg_tag> readSettings;
        // bg::image_t image;

    } catch (std::exception const& e){
        fprintf(stderr, "%s\n", e.what());
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}










