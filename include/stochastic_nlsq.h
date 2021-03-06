#ifndef STOCHASTIC_NLSQ_H
#define STOCHASTIC_NLSQ_H

#include "common_utils.h"
#include "nlsq.h"
#include "twoview_models.h"

#include <Eigen/Core>
#include <Eigen/Dense>

#include<random>
#include<ctime>
#include<chrono>
#include<vector>

struct StochasticTwoViewOptimizer:public TwoViewOptimizer
{

    StochasticTwoViewOptimizer(std::vector<Vec3> const &x1, std::vector<Vec3> const & x2, TwoViewModel  &model,
                             Eigen::VectorXd &params, Config const &config, Stat &stat);

    // void randomSampling(int sampleSize, std::vector<int>& sampled_indices);

    double computeSubsetCost(vector<double>& res, bool fillGrad = true);
    
    virtual void optimize();

    //Optimization with relaxed condition
    virtual void optimize_relaxed();

    protected:
        vector<int> _sampled_idx;
        Sampler _sampler;

};


#endif
