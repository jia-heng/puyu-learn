// -*- coding: utf-8 -*-
/* *******************************************************************************
Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved
Func：exr data loader api
Creator：s00827220
********************************************************************************** */
#include "dataloader.h"
#include <cmath>
#include <stdlib.h>
#include <sstream>
#include <iomanip>

/* V1版本 输入为
    'color': shape(1, 3, 720, 1280),
    'depth': shape(1, 1, 720, 1280),
    'normal': shape(1, 3, 720, 1280),
    'diffuse': shape(1, 3, 720, 1280),
    'motion': shape(1, 2, 720, 1280),*/

// {"color", "depth", "normal", "diffuse", "motion", "temporal", "output1", "output2" }
uint16_t tensorChannel[] = { 3, 1, 3, 3, 2, 38, 3, 38 };

std::string tensorNames[] = {
    "color",
    "depth",
    "normal",
    "albedo",
    "motionVector",
    //    "refraction",
    //    "reflection"
};

InferDataLoader::InferDataLoader(const std::string& dataPath, const std::string& savePath, int length, int width, int height) :
    mDataPath(dataPath), mSavePath(savePath), sequenceLength(length), mWidth(width), mHeight(height)
{
    setpad();
}

void InferDataLoader::setpad() {
    int offset = 0;
    if (std::fmod(mHeight, 32) != 0) { offset = (mHeight / 32 + 1) * 32 - mHeight; }
    padstru.bottom = offset;
    offset = 0;
    if (std::fmod(mWidth, 32) != 0) { offset = (mWidth / 32 + 1) * 32 - mWidth; }
    padstru.right = offset;
}

int InferDataLoader::getExHeight()
{
    return mHeight + padstru.bottom;
}

void InferDataLoader::saveExrFile(float* data, int index)
{
    std::ostringstream oss;
    oss << mSavePath
        << "/predict_"
        << std::setfill('0') << std::setw(6) << index
        << ".exr";
    std::string savePath = oss.str();

    int height = mHeight;
    int width = mWidth;
    int exHeight = getExHeight();
    Imf::Array2D<Imf::Rgba> pixels(height, width);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = y * width + x;
            pixels[y][x] = Imf::Rgba(data[index], data[index + width * exHeight], data[index + 2 * width * exHeight]);
        }
    }

    try
    {
        Imf::RgbaOutputFile file(savePath.c_str(), width, height, Imf::WRITE_RGBA);
        file.setFrameBuffer(&pixels[0][0], 1, width);
        file.writePixels(height);
    }
    catch (const std::exception& e)
    {
        std::cerr << "error writing image file hello.exr:" << e.what() << std::endl;
        return;
    }
}

void InferDataLoader::loadExrFile(Imf::Array2D<Imf::Rgba>& pixels, LAYERNAMES feature, int frame_idx)
{
    std::ostringstream oss;
    oss << mDataPath
        << "/"
        << tensorNames[feature]
        << "/"
        << tensorNames[feature]
        << "_"
        << std::setfill('0') << std::setw(6) << frame_idx
        << "_0.exr";
    std::string featurePath = oss.str();

    try {
        Imf::RgbaInputFile file(featurePath.c_str());
        Imath::Box2i       dw = file.dataWindow();
        int                width = dw.max.x - dw.min.x + 1;
        int                height = dw.max.y - dw.min.y + 1;

        file.setFrameBuffer(&pixels[0][0], 1, width);
        file.readPixels(dw.min.y, dw.max.y);
    }
    catch (const std::exception& e) {
        std::cerr << "error reading image file hello.exr:" << e.what() << std::endl;
        return;
    }
}

void InferDataLoader::setTensorLayer(float* buffer, LAYERNAMES feature, int frame_idx)
{
    int height = mHeight;
    int width = mWidth;
    int exheight = height + padstru.bottom;
    int exwidth = width + padstru.right;
    int offset = exheight * exwidth;
    Imf::Array2D<Imf::Rgba> pixels(height, width);
    loadExrFile(pixels, feature, frame_idx);
    float maxVal = 5000.0;
    if (tensorChannel[feature] == 3) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = y * width + x;
                buffer[index] = pixels[y][x].r;
                buffer[index + offset] = pixels[y][x].g;
                buffer[index + 2 * offset] = pixels[y][x].b;
            }
        }
        for (int y = height, pad = height - 1; y < exheight; y++, pad--) {
            for (int x = 0; x < width; x++) {
                int index = y * width + x;
                int reuse = pad * width + x;
                buffer[index] = buffer[reuse];
                buffer[index + offset] = buffer[reuse + offset];
                buffer[index + 2 * offset] = buffer[reuse + 2 * offset];
            }
        }
    }
    if (tensorChannel[feature] == 2) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = y * width + x;
                buffer[index] = pixels[y][x].r > maxVal ? maxVal : pixels[y][x].r;
                buffer[index + offset] = pixels[y][x].g > maxVal ? maxVal : pixels[y][x].g;
            }
        }
        for (int y = height, pad = height - 1; y < exheight; y++, pad--) {
            for (int x = 0; x < width; x++) {
                int index = y * width + x;
                int reuse = pad * width + x;
                buffer[index] = buffer[reuse];
                buffer[index + offset] = buffer[reuse + offset];
            }
        }
    }
    if (tensorChannel[feature] == 1) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = y * width + x;
                buffer[index] = std::log1p(1.0/(pixels[y][x].r > maxVal ? maxVal : pixels[y][x].r));
            }
        }
        for (int y = height, pad = height - 1; y < exheight; y++, pad--) {
            for (int x = 0; x < width; x++) {
                int index = y * width + x;
                int reuse = pad * width + x;
                buffer[index] = buffer[reuse];
            }
        }
    }
}

