// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include <mcrt_denoise/denoiser/Denoiser.h>

#include <scene_rdl2/common/except/exceptions.h>
#include <scene_rdl2/common/fb_util/FbTypes.h>
#include <scene_rdl2/render/logging/logging.h>
#include <scene_rdl2/render/util/Args.h>
#include <scene_rdl2/render/util/Files.h>

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>


//---------------------------------------------------------------------------

void usage(char *argv0)
{
    std::cerr << "Denoises an image" << std::endl;
    std::cerr << "Usage: " << argv0 << " [options]" << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -in input.exr              input image file to denoise" << std::endl;
    std::cerr << "  -albedo albedo.exr         optional input albedo image to aid denoising" << std::endl;
    std::cerr << "  -normals normals.exr       optional input normals image to aid denoising" << std::endl;
    std::cerr << "  -mode optix OR -mode oidn OR -mode oidn_cpu OR -mode oidn_cuda" << std::endl;
    std::cerr << "                             use Optix or Open Image Denoise denoiser" << std::endl;
    std::cerr << "  -out output.exr            denoised output image file (default = \"denoised.exr\")" << std::endl;
}

//---------------------------------------------------------------------------

void
readImage(const std::string &filename, scene_rdl2::fb_util::RenderBuffer *buffer, OIIO::ImageSpec *origImageSpec)
{
    std::unique_ptr<OIIO::ImageInput> in(OIIO::ImageInput::create(filename));
    if (!in) {
        throw scene_rdl2::except::IoError("Cannot find image file: \"" + filename +
                  "\" (file not found)");
    }

    OIIO::ImageSpec imageSpec;
    if (in->open(filename, imageSpec) == false) {
        throw scene_rdl2::except::IoError("Cannot open image file: \"" + filename +
                  "\" (" + in->geterror() + ")");
    }

    if (origImageSpec) {
        *origImageSpec = imageSpec;
    }

    buffer->init(imageSpec.width, imageSpec.height);

    std::vector<float> tmpBuffer(imageSpec.width * imageSpec.height * 4);

    if (!in->read_image(OIIO::TypeDesc::FLOAT, tmpBuffer.data())) {
        throw scene_rdl2::except::IoError("Cannot read image file: \"" + filename +
                  "\" (" + in->geterror() + ")");
    }

    // Flip the buffer: our frame buffers are upside down compared to what
    // OIIO is giving us.
    imageSpec.format = OIIO::TypeDesc::FLOAT;
    OIIO::ImageSpec finalmageSpec = imageSpec;
    finalmageSpec.nchannels = 4;
    OIIO::ImageBuf source("source", imageSpec, tmpBuffer.data());
    OIIO::ImageBuf flipped("flipped", finalmageSpec, buffer->getData());
    if (!OIIO::ImageBufAlgo::flip(flipped, source)) {
        throw scene_rdl2::except::IoError("Cannot flip image");
    }
}

void
writeImage(const std::string &filename, scene_rdl2::fb_util::RenderBuffer *buffer, const OIIO::ImageSpec &outputImageSpec)
{
    // Create an OIIO output buffer.
    std::unique_ptr<OIIO::ImageOutput> out(OIIO::ImageOutput::create(filename.c_str()));
    if (!out) {
        std::stringstream errMsg;
        errMsg << "Failed to open " << filename << " for writing.";
        throw scene_rdl2::except::IoError(errMsg.str());
    }

    // Define the format of the input buffer.
    OIIO::ImageSpec inputSpec = outputImageSpec;
    inputSpec.width = buffer->getWidth();
    inputSpec.height = buffer->getHeight();
    inputSpec.depth = 1;
    inputSpec.full_depth = 1;
    inputSpec.tile_width = 0;
    inputSpec.tile_height = 0;
    inputSpec.tile_depth = 1;
    inputSpec.nchannels = 4;
    inputSpec.format = OIIO::TypeDesc::FLOAT;
    inputSpec.channelformats.clear();
    inputSpec.channelnames.clear();
    inputSpec.deep = false;

    // Flip the buffer: our frame buffers are upside down compared to what
    // OIIO is expecting.
    OIIO::ImageBuf srcBuffer(filename, inputSpec, buffer->getData());
    OIIO::ImageBuf flippedBuffer(filename, outputImageSpec);
    OIIO::ImageBufAlgo::flip(flippedBuffer, srcBuffer);

    // Open the output file, write the image buffer into it, and close it.
    if (!out->open(filename.c_str(), outputImageSpec)) {
        throw scene_rdl2::except::IoError("Cannot open image file: \"" + filename +
                  "\" (" + out->geterror() + ")");
    }
    if (!flippedBuffer.write(out.get())) {
        throw scene_rdl2::except::IoError("Cannot write image to file: \"" + filename +
                  "\" (" + out->geterror() + ")");
    }
    out->close();
}

int
main(int argc, char* argv[])
{
    try {
        //------------------------------------

        // Initialize our logging
        scene_rdl2::logging::Logger::init();

        // Check for no flags or help flag.
        if (argc == 1 || std::string(argv[1]) == "-h") {
            usage(argv[0]);
            std::exit(EXIT_FAILURE);
        }

        //------------------------------------

        // Args parsing
        scene_rdl2::util::Args args(argc, argv);
        scene_rdl2::util::Args::StringArray values;

        if (args.getFlagValues("-in", 1, values) < 0) {
            usage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
        std::string inFilename = values[0];

        std::string inAlbedo;
        if (args.getFlagValues("-albedo", 1, values) >= 0) {
            inAlbedo = values[0];
        }

        std::string inNormals;
        if (args.getFlagValues("-normals", 1, values) >= 0) {
            inNormals = values[0];
        }

        std::string outFilename = "denoised.exr";
        if (args.getFlagValues("-out", 1, values) >= 0) {
            outFilename = values[0];
        }

        moonray::denoiser::DenoiserMode denoiserMode = moonray::denoiser::OPTIX;
        std::string inDenoiserMode;
        if (args.getFlagValues("-mode", 1, values) >= 0) {
            inDenoiserMode = values[0];
        }
        if (inDenoiserMode == "" || inDenoiserMode == "optix") {
            denoiserMode = moonray::denoiser::OPTIX;
            std::cout << "Denoising with Optix" << std::endl;
        } else if (inDenoiserMode == "oidn") {
            denoiserMode = moonray::denoiser::OPEN_IMAGE_DENOISE;
            std::cout << "Denoising with Open Image Denoise (default/best device)" << std::endl;
        } else if (inDenoiserMode == "oidn_cpu") {
            denoiserMode = moonray::denoiser::OPEN_IMAGE_DENOISE_CPU;
            std::cout << "Denoising with Open Image Denoise (CPU device)" << std::endl;
        } else if (inDenoiserMode == "oidn_cuda") {
            denoiserMode = moonray::denoiser::OPEN_IMAGE_DENOISE_CUDA;
            std::cout << "Denoising with Open Image Denoise (CUDA device)" << std::endl;
        } else {
            std::cerr << "Unrecognized denoiser mode." << std::endl;
            usage(argv[0]);
            std::exit(EXIT_FAILURE);
        }

        // Load input image
        std::cout << "Reading image: \"" << inFilename << "\"\n";
        scene_rdl2::fb_util::RenderBuffer inputBuffer;
        OIIO::ImageSpec inputImageSpec;
        readImage(inFilename, &inputBuffer, &inputImageSpec);
        unsigned width = inputBuffer.getWidth();
        unsigned height = inputBuffer.getHeight();
        std::cout << "Image width: " << width << "\n";
        std::cout << "Image height: " << height << "\n";

        // Load optional albedo
        scene_rdl2::fb_util::RenderBuffer albedoBuffer;
        unsigned albedoWidth = 0;
        unsigned albedoHeight = 0;
        bool useAlbedo = false;
        if (!inAlbedo.empty()) {
            std::cout << "Reading albedo image: \"" << inAlbedo << "\"\n";
            readImage(inAlbedo, &albedoBuffer, nullptr);
            albedoWidth = albedoBuffer.getWidth();
            albedoHeight = albedoBuffer.getHeight();
            std::cout << "Albedo image width: " << albedoWidth << "\n";
            std::cout << "Albedo image height: " << albedoHeight << "\n";

            if (width != albedoWidth || height != albedoHeight) {
                std::cerr << "Error: Input and albedo dimensions must match." << std::endl;
                std::exit(EXIT_FAILURE);
            }
            useAlbedo = true;
        }

        // Load optional normals
        scene_rdl2::fb_util::RenderBuffer normalsBuffer;
        unsigned normalsWidth = 0;
        unsigned normalsHeight = 0;
        bool useNormals = false;
        if (!inNormals.empty()) {
            std::cout << "Reading normals image: \"" << inNormals << "\"\n";
            readImage(inNormals, &normalsBuffer, nullptr);
            normalsWidth = normalsBuffer.getWidth();
            normalsHeight = normalsBuffer.getHeight();
            std::cout << "Normals image width: " << normalsWidth << "\n";
            std::cout << "Normals image height: " << normalsHeight << "\n";

            if (width != normalsWidth || height != normalsHeight) {
                std::cerr << "Error: Input and normals dimensions must match." << std::endl;
                std::exit(EXIT_FAILURE);
            }
            useNormals = true;
        }

        // Do denoising.
        std::cout << "Denoising..." << std::endl;

        std::string errorMsg;
        std::unique_ptr<moonray::denoiser::Denoiser> denoiser = 
            std::make_unique<moonray::denoiser::Denoiser>(denoiserMode, width, height, 
                                                          useAlbedo, useNormals, &errorMsg);
        if (!errorMsg.empty()) {
            std::cerr << "Error creating denoiser: " << errorMsg << std::endl;
            std::exit(EXIT_FAILURE);
        }

        scene_rdl2::fb_util::RenderBuffer denoisedBuffer;
        denoisedBuffer.init(width, height);

        const scene_rdl2::fb_util::RenderColor *inputBeautyPixels = inputBuffer.getData();
        const scene_rdl2::fb_util::RenderColor *inputAlbedoPixels = useAlbedo ? albedoBuffer.getData() : nullptr;
        const scene_rdl2::fb_util::RenderColor *inputNormalPixels = useNormals ? normalsBuffer.getData() : nullptr;
        scene_rdl2::fb_util::RenderColor *denoisedPixels = denoisedBuffer.getData();

        denoiser->denoise(reinterpret_cast<const float*>(inputBeautyPixels),
                          reinterpret_cast<const float*>(inputAlbedoPixels),
                          reinterpret_cast<const float*>(inputNormalPixels),
                          reinterpret_cast<float*>(denoisedPixels),
                          &errorMsg);

        if (!errorMsg.empty()) {
            std::cerr << "Error denoising: " << errorMsg << std::endl;
        }

        // Output results.
        std::cout << "Writing output image: \"" << outFilename << "\"\n";
        writeImage(outFilename, &denoisedBuffer, inputImageSpec);
    }

    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    return EXIT_SUCCESS;
}

