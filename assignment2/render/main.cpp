#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>

#include "refRenderer.h"
#include "cudaRenderer.h"
#include "platformgl.h"


void startRendererWithDisplay(CircleRenderer* renderer);
void startBenchmark(CircleRenderer* renderer, int startFrame, int totalFrames, const std::string& frameFilename);
void CheckBenchmark(CircleRenderer* ref_renderer, CircleRenderer* cuda_renderer,
                        int benchmarkFrameStart, int totalFrames, const std::string& frameFilename);


void usage(const char* progname) {
    printf("Usage: %s [options] scenename\n", progname);
    printf("Valid scenenames are: rgb, rgby, rand10k, rand100k, biglittle, littlebig, pattern, bouncingballs, fireworks, hypnosis, snow, snowsingle\n");
    printf("Program Options:\n");
    printf("  -b  --bench <START:END>    Benchmark mode, do not create display. Time frames [START,END)\n");
    printf("  -c  --check                Check correctness of output\n");
    printf("  -f  --file  <FILENAME>     Dump frames in benchmark mode (FILENAME_xxxx.ppm)\n");
    printf("  -r  --renderer <ref/cuda>  Select renderer: ref or cuda\n");
    printf("  -s  --size  <INT>          Make rendered image <INT>x<INT> pixels\n");
    printf("  -?  --help                 This message\n");
}


int main(int argc, char** argv)
{
    int benchmarkFrameStart = -1;   // the start frame number in benchmark version.
    int benchmarkFrameEnd = -1;
    int imageSize = 1024;

    std::string sceneNameStr;   // used to accept the command option.
    std::string frameFilename;
    SceneName sceneName;
    bool useRefRenderer = true;

    bool checkCorrectness = false;

    // parse commandline options ////////////////////////////////////////////
    /*
        https://man7.org/linux/man-pages/man3/getopt.3.html
        long_options is in the type of: 
        struct option {
               const char *name;
               int         has_arg; // 0, 1, 2
               int        *flag;
               int         val;     // the value to return
           };
    */
    int opt;
    static struct option long_options[] = {
        {"help",     0, 0,  '?'},
        {"check",    0, 0,  'c'},
        {"bench",    1, 0,  'b'},
        {"file",     1, 0,  'f'},
        {"renderer", 1, 0,  'r'},
        {"size",     1, 0,  's'},
        // the last element of the array has to be filled with zeros.
        {0 ,0, 0, 0}
    };

    // return the option character, If there are no more option characters, getopt() returns -1. then why the condition is EOF here?
    while ((opt = getopt_long(argc, argv, "b:f:r:s:c?", long_options, NULL)) != EOF) {

        switch (opt) {
        case 'b':
            // sscanf: read data from s, fill them into additional arguments, just like scanf but reading from s instead of from standard ipnut.
            // if there is text in the current argv-element, then it's returned in optarg, otherwise optarg is set to be 0, optarg is auto-filled with the subsequent text( value ).
            if (sscanf(optarg, "%d:%d", &benchmarkFrameStart, &benchmarkFrameEnd) != 2) {
                fprintf(stderr, "Invalid argument to - b option\n");
                usage(argv[0]);
                exit(1);
            }
            break;
        case 'c':
            checkCorrectness = true;
            break;
        case 'f':
            frameFilename = optarg;
            break;
        case 'r':
            if (std::string(optarg).compare("cuda") == 0) {
                useRefRenderer = false;
            }
            break;
        case 's':
            imageSize = atoi(optarg);
            break;
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }
    // end parsing of commandline options //////////////////////////////////////

    // the last command param is not option param, so we can't use getopt to get it instead of getting it using optind(option param index)
    if (optind + 1 > argc) {
        fprintf(stderr, "Error: missing scene name\n");
        usage(argv[0]);
        return 1;
    }

    sceneNameStr = argv[optind];

    if (sceneNameStr.compare("snow") == 0) {
        sceneName = SNOWFLAKES;
    } else if (sceneNameStr.compare("snowsingle") == 0) {
        sceneName = SNOWFLAKES_SINGLE_FRAME;
    } else if (sceneNameStr.compare("rgb") == 0) {
        sceneName = CIRCLE_RGB;
    } else if (sceneNameStr.compare("rgby") == 0) {
        sceneName = CIRCLE_RGBY;
    } else if (sceneNameStr.compare("rand10k") == 0) {
        // render #10k bubbles.
        sceneName = CIRCLE_TEST_10K;
    } else if (sceneNameStr.compare("rand100k") == 0) {
        sceneName = CIRCLE_TEST_100K;
    } else if (sceneNameStr.compare("pattern") == 0) {
        sceneName = PATTERN;
    } else if (sceneNameStr.compare("biglittle") == 0) {
        sceneName = BIG_LITTLE;
    } else if (sceneNameStr.compare("littlebig") == 0) {
        sceneName = LITTLE_BIG;
    } else if (sceneNameStr.compare("bouncingballs") == 0) {
        // dynamic
        sceneName = BOUNCING_BALLS;  
    } else if (sceneNameStr.compare("hypnosis") == 0) { 
        // dymamic
        // hypnosis: hɪpnoʊsɪs, Hypnosis is a state in which a person seems to be asleep but can still see, hear, or respond to things said to them. so the picture it produces is just like the rippling water.
        sceneName = HYPNOSIS;           
    } else if (sceneNameStr.compare("fireworks") == 0) { 
        // em... all right, the shape of fireworks also like rippling water but disperesed in different position.
        sceneName = FIREWORKS;    
    }else {
        fprintf(stderr, "Unknown scene name (%s)\n", sceneNameStr.c_str());
        usage(argv[0]);
        return 1;
    }

    printf("Rendering to %dx%d image\n", imageSize, imageSize);

    // abstract hper circle renderer.
    CircleRenderer* renderer;

    if (checkCorrectness) {

        // Need both the renderers
        CircleRenderer* ref_renderer;
        CircleRenderer* cuda_renderer;

        ref_renderer = new RefRenderer();
        cuda_renderer = new CudaRenderer();

        ref_renderer->allocOutputImage(imageSize, imageSize);
        ref_renderer->loadScene(sceneName);
        ref_renderer->setup();

        cuda_renderer->allocOutputImage(imageSize, imageSize);
        cuda_renderer->loadScene(sceneName);
        cuda_renderer->setup();

        // Check the correctness
        CheckBenchmark(ref_renderer, cuda_renderer, 0, 1, frameFilename);
    }
    else {
        if (useRefRenderer)
            renderer = new RefRenderer();
        else
            renderer = new CudaRenderer();

        renderer->allocOutputImage(imageSize, imageSize);
        renderer->loadScene(sceneName);
        renderer->setup();

        // showing window is not opened in benchmark model
        if (benchmarkFrameStart >= 0)
            startBenchmark(renderer, benchmarkFrameStart, benchmarkFrameEnd - benchmarkFrameStart, frameFilename);
        else {
            glutInit(&argc, argv);
            startRendererWithDisplay(renderer);
        }
    }

    return 0;
}
