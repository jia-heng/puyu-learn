// -*- coding: utf-8 -*-
/* *******************************************************************************
Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved
Func：exr data loader api
Creator：s00827220
********************************************************************************** */
#ifndef DATALOADER_H
#define DATALOADER_H

#include <iostream>
#include <string>
#include <OpenEXR/ImfRgbaFile.h>
#include <OpenEXR/ImfArray.h>

#pragma once

typedef enum {
    COLOR = 0,
    DEPTH,
    NORMAL,
    DIFFUSE,
    MOTION = 4,
    // REFRACTION,
    // REFLECTION,
    TEMPORAL,
    PREDICT,
    OUTPUT_TEMP
} LAYERNAMES;

extern uint16_t tensorChannel[];
extern std::string tensorNames[];

typedef struct {
    int bottom;
    int right;
} PadStru;

class InferDataLoader {
public:
    InferDataLoader(const std::string& dataPath, const std::string& savePath, int length, int width, int height);
    void setpad();
    void loadExrFile(Imf::Array2D<Imf::Rgba>& pixels, LAYERNAMES feature, int frame_idx);
    void setTensorLayer(float* buffer, LAYERNAMES feature, int frame_idx);
    void saveExrFile(float* data, int index);
    int InferDataLoader::getExHeight();

private:
    std::string mDataPath;
    std::string mSavePath;
    int sequenceLength;
    int mWidth;
    int mHeight;
    PadStru padstru;
};


#endif
