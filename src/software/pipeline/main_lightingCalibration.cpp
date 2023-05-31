#include <aliceVision/image/all.hpp>
#include <aliceVision/image/io.hpp>

// SFMData
#include <aliceVision/sfmData/SfMData.hpp>
#include <aliceVision/sfmDataIO/sfmDataIO.hpp>

// Lighting calibration
#include <aliceVision/lightingEstimation/lightingCalibration.hpp>

// Command line parameters
#include <aliceVision/cmdline/cmdline.hpp>
#include <aliceVision/system/main.hpp>
#include <aliceVision/system/Logger.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <array>
#include <sstream>
#include <fstream>

// These constants define the current software version.
// They must be updated when the command line is changed.
#define ALICEVISION_SOFTWARE_VERSION_MAJOR 1
#define ALICEVISION_SOFTWARE_VERSION_MINOR 0

// Namespaces
namespace po = boost::program_options;
namespace fs = boost::filesystem;

using namespace aliceVision;

int aliceVision_main(int argc, char **argv)
{
    std::string inputPath;
    std::string inputJSON;
    std::string ouputJSON;
    std::string method;
    bool saveAsModel;

    po::options_description requiredParams("Required parameters");
    requiredParams.add_options()
        ("inputPath,i", po::value<std::string>(&inputPath)->required(), "Path to input. Could be SfMData file or folder with pictures.")
        ("inputJSON, j", po::value<std::string>(&inputJSON)->required(), "Path to JSON which describes sphere positions and radius.")
        ("outputFile, o", po::value<std::string>(&ouputJSON)->required(), "Path to JSON output file.");

    po::options_description optionalParams("Optional parameters");
    optionalParams.add_options()
        ("saveAsModel, s", po::value<bool>(&saveAsModel)->default_value(false), "Calibration used for several datasets.")
        ("method, m", po::value<std::string>(&method)->default_value("brightestPoint"), "Method for light estimation.");

    CmdLine cmdline("AliceVision lightingCalibration");
    cmdline.add(requiredParams);
    cmdline.add(optionalParams);

    if (!cmdline.execute(argc, argv))
    {
        return EXIT_FAILURE;
    }

    if (fs::is_directory(inputPath))
    {
        std::cout << "Directory input : WIP" << std::endl;
    }
    else
    {
        sfmData::SfMData sfmData;
        if (!sfmDataIO::Load(sfmData, inputPath, sfmDataIO::ESfMData(sfmDataIO::VIEWS | sfmDataIO::INTRINSICS)))
        {
            ALICEVISION_LOG_ERROR("The input file '" + inputPath + "' cannot be read.");
            return EXIT_FAILURE;
        }
        lightingEstimation::lightCalibration(sfmData, inputJSON, ouputJSON, method, saveAsModel);
    }

    return 0;
}
