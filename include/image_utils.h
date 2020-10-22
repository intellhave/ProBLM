// Common Utilities for images
#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include<iostream>
#include<fstream>
#include<random>
#include<ctime>
#include<chrono>
#include<vector>
#include<Eigen/Core>
#include<Eigen/Dense>
#include<string>

#include "common_utils.h"

#include "Base/v3d_image.h"
#include "Math/v3d_nonlinlsq.h"

using namespace V3D;


template <typename T>
    inline void
    convolveImageHorizontal(V3D::Image<T> const &src, int const kernelSize, int const center, T const *kernel, V3D::Image<T> &dst)
    {
        int const w = src.width();
        int const h = src.height();
        int const nChannels = src.numChannels();

        dst.resize(w, h, nChannels);

        for (int ch = 0; ch < nChannels; ++ch)
        {
            for (int y = 0; y < h; ++y)
            {
                for (int x = 0; x < w; ++x)
                {
                    T sum(0);
                    for (int dx = 0; dx < kernelSize; ++dx)
                    {
                        int xx = x - center + dx;
                        //xx = abs(xx); // mirror at 0, if necessary
                        xx = (xx < 0) ? (-xx - 1) : xx;         // mirror at 0, if necessary
                        xx = (xx >= w) ? (2 * w - xx - 1) : xx; // mirror at w-1, xx = w - (xx - w + 1) = 2*w - xx - 1
                        sum += kernel[dx] * src(xx, y, ch);
                    } // end for (dx)
                    dst(x, y, ch) = sum;
                } // end for (dx)
            }     // end for (y)
        }         // end for (ch)
    }             // end convolveImage()

    template <typename T>
    inline void
    convolveImageVertical(V3D::Image<T> const &src, int const kernelSize, int const center, T const *kernel, V3D::Image<T> &dst)
    {
        int const w = src.width();
        int const h = src.height();
        int const nChannels = src.numChannels();

        dst.resize(w, h, nChannels);

        for (int ch = 0; ch < nChannels; ++ch)
        {
            for (int y = 0; y < h; ++y)
            {
                for (int x = 0; x < w; ++x)
                {
                    T sum(0);
                    for (int dy = 0; dy < kernelSize; ++dy)
                    {
                        int yy = y - center + dy;
                        //yy = abs(yy); // mirror at 0, if necessary
                        yy = (yy < 0) ? (-yy - 1) : yy;         // mirror at 0, if necessary
                        yy = (yy >= h) ? (2 * h - yy - 1) : yy; // mirror at h-1
                        sum += kernel[dy] * src(x, yy, ch);
                    } // end for (dx)
                    dst(x, y, ch) = sum;
                } // end for (dx)
            }     // end for (y)
        }         // end for (ch)
    }             // end convolveImage()

    //**********************************************************************

    inline float
    access_image(Image<float> const &im, float x, float y, int const ch = 0)
    {
        int const w = im.width(), h = im.height();
        x = std::max(0.0f, std::min(w - 1.0f, x));
        y = std::max(0.0f, std::min(h - 1.0f, y));
        int x0 = int(floor(x)), y0 = int(floor(y));
        if (x0 == w - 1)
            x0 = w - 2;
        if (y0 == h - 1)
            y0 = h - 2;
        int const x1 = x0 + 1, y1 = y0 + 1;

        float const wx = x - floor(x), wy = y - floor(y);
        float const w00 = (1.0f - wx) * (1.0f - wy), w10 = wx * (1.0f - wy), w01 = (1.0f - wx) * wy, w11 = wx * wy;

        return w00 * im(x0, y0, ch) + w10 * im(x1, y0, ch) + w01 * im(x0, y1, ch) + w11 * im(x1, y1, ch);
    }

    inline V3D::Vector2f
    access_image(Image<V3D::Vector2f> const &im, float x, float y, int const ch = 0)
    {
        int const w = im.width(), h = im.height();
        x = std::max(0.0f, std::min(w - 1.0f, x));
        y = std::max(0.0f, std::min(h - 1.0f, y));
        int x0 = int(floor(x)), y0 = int(floor(y));
        if (x0 == w - 1)
            x0 = w - 2;
        if (y0 == h - 1)
            y0 = h - 2;
        int const x1 = x0 + 1, y1 = y0 + 1;

        float const wx = x - floor(x), wy = y - floor(y);
        float const w00 = (1.0f - wx) * (1.0f - wy), w10 = wx * (1.0f - wy), w01 = (1.0f - wx) * wy, w11 = wx * wy;

        return w00 * im(x0, y0, ch) + w10 * im(x1, y0, ch) + w01 * im(x0, y1, ch) + w11 * im(x1, y1, ch);
    }

    inline void
    computeGradient(Image<float> const &im, Image<V3D::Vector2f> &dst)
    {
        int const w = im.width();
        int const h = im.height();

        dst.resize(w, h, 1);

        for (int y = 0; y < h; ++y)
        {
            int const Y0 = (y > 0) ? (y - 1) : 1;
            int const Y1 = (y < h - 1) ? (y + 1) : (h - 2);
            for (int x = 0; x < w; ++x)
            {
                int const X0 = (x > 0) ? (x - 1) : 1;
                int const X1 = (x < w - 1) ? (x + 1) : (w - 2);
                dst(x, y)[0] = 0.5 * (im(X1, y, 0) - im(X0, y, 0));
                dst(x, y)[1] = 0.5 * (im(x, Y1, 0) - im(x, Y0, 0));
            }
        } // end for (y)
    }     // end computeGradient()


#endif


