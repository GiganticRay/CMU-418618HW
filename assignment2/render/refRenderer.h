#ifndef __REF_RENDERER_H__
#define __REF_RENDERER_H__

#include "circleRenderer.h"


class RefRenderer : public CircleRenderer {

private:

    Image* image;           // every render has its own Image object, which is the container for every frame.
    SceneName sceneName;    // enum type, indicate scene's name

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

public:

    RefRenderer();
    virtual ~RefRenderer();

    const Image* getImage();

    // the function is called prior to rendering the first frame
    void setup();

    // ?
    void loadScene(SceneName name);

    void allocOutputImage(int width, int height);

    void clearImage();

    // the function is invoked once per frame. it updates circle positions and velocities.
    void advanceAnimation();

    // the function is called for each frame and is responsible for drawing all circles in the oupt image.
    void render();

    // ?
    void dumpParticles(const char* filename);

    // ?
    void shadePixel(
        int circleIndex,
        float pixelCenterX, float pixelCenterY,
        float px, float py, float pz,
        float* pixelData);
};


#endif
