#include <iostream>
#include <string>
#include "nlsq.h"
#include "stochastic_nlsq.h"
#include "image_utils.h"
#include "common_utils.h"
#include "twoview_models.h"

#include "Base/v3d_image.h"
#include "Math/v3d_nonlinlsq.h"


using namespace std;
using namespace V3D;

//**********************************************************************

int main(int argc, char const *argv[])
{
    if (argc < 3)
    {
        cerr << "Usage: " << argv[0] << " <image 0> <image 1>" << endl;
        return -1;
    }

    std::string output_name = "reg_";
    if (argc == 4)
    {
        output_name = string(argv[3]);    
    }

    Image<unsigned char> im0_src, im1_src;    
    loadImageFile(argv[1], im0_src);
    loadImageFile(argv[2], im1_src);

    int const w = im0_src.width(), h = im0_src.height();
    int a = 0;

    Image<float> im0(w, h, 1), im1(w, h, 1);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
        {
            im0(x, y) = im0_src(x, y) / 255.0f;
            im1(x, y) = im1_src(x, y) / 255.0f;
        }

     // Test setup: move im1 by (dX, dY) pixels
    if (0)
    {
        int const dX = -3, dY = 2;

        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
            {
                int const xx = std::max(0, std::min(w-1, x+dX));
                int const yy = std::max(0, std::min(h-1, y+dY));
                im1(x, y) = im0(xx, yy);
            }
    }

    // Blur the images for larger convergence basin
    if (1)
    {
        const float kernel[] = { 1.0/16, 4.0/16, 6.0/16, 4.0/16, 1.0/16 };
        Image<float> tmp(w, h, 1);

        convolveImageHorizontal(im0, 5, 2, kernel, tmp);
        convolveImageVertical(tmp, 5, 2, kernel, im0);

        convolveImageHorizontal(im1, 5, 2, kernel, tmp);
        convolveImageVertical(tmp, 5, 2, kernel, im1);
    }

    saveImageChannel(im0, 0, 0.0f, 1.0f, "im0.png");
    saveImageChannel(im1, 0, 0.0f, 1.0f, "im1.png");

    Image<V3D::Vector2f> grad_im0, grad_im1;
    computeGradient(im0, grad_im0);
    computeGradient(im1, grad_im1);

    
    std::string run_str = output_name;
    Config config;
    config.sampling_rate = 0.1;
    config.prob = 0.1;
    config.inlierThreshold = 0.1;
    config.innerMaxIter = 1;
    config.alpha = 0.5;
    config.maxIter = 1000;
        //Parameters
    Eigen::VectorXd params;
    Eigen::VectorXd init_params;

    int modelDimension = 8;
    params.resize(modelDimension);
    init_params.resize(modelDimension);
    
    //Init to identity matrix
    init_params << 1, 0, 0, 0, 1, 0, 0, 0;
    //
    // Preapre the vector to store the correspondences
    std::vector<Vec3> x1;
    std::vector<Vec3> x2;


    for (int k = 0; k < w*h; ++k)
    {
        int const y = k/w, x = k - y*w;
        Vec3 v; v << x,y,1;
        x1.push_back(v);
    }

    for (int i = 0; i < modelDimension; ++i)
        params[i] = init_params[i];

    HomographyRegModel model(x1, x2, params, config, im0, im1, grad_im0, grad_im1);

    Stat opt_stat;
    TwoViewOptimizer *opt = new TwoViewOptimizer(x1, x2, model, params, config, opt_stat);
    opt->optimize();
    /* std::string opt_log_path = "./logs/lm/" + run_str + ".txt"; */
    /* opt_stat.WriteToFile(opt_log_path); */
    /* std::string homo_out = "./homo_results/" + run_str + ".txt"; */
    /* model.writeHomographyMatrix(homo_out); */

    cout << "----- LM finished. Now running ProBLM------" << endl;
    for (int i = 0; i < modelDimension; ++i)
        params[i] = init_params[i];
    Stat sto_stat;  
    StochasticTwoViewOptimizer *sto_opt = new StochasticTwoViewOptimizer(x1, x2, model, params, config, sto_stat);
    
    sto_opt->optimize();
    /* std::string sto_log_path = "./logs/sto/" + run_str + ".txt"; */
    /* sto_stat.WriteToFile(sto_log_path); */

    return 0;
}
