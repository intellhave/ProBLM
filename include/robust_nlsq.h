#ifndef ROBUST_NLSQ_H
#define ROBUST_NLSQ_H

#include "common_utils.h"
#include "twoview_models.h"
#include "nlsq.h"
#include <Eigen/Core>
#include <Eigen/Dense>

#include<random>
#include<ctime>
#include<chrono>
#include<vector>

struct RobustTwoViewOptimizer_Lifted: public TwoViewOptimizer
{

    RobustTwoViewOptimizer_Lifted(std::vector<Vec3> const &x1, std::vector<Vec3> const & x2, TwoViewModel &model,
                            Eigen::VectorXd &params, Config const &config, Stat &stat);
    
    virtual double compute_cost(bool fillGrad = true);

    int countInliers();

    virtual void updateParameters();

    virtual void retractSolution();

    virtual void optimize();

    double _tau = 1.0;
    double _inlierThreshold = 0.5;
    std::vector<int> _inlierSet;

    protected:
        //Weight for each residual
        Eigen::VectorXd _weights;
        Eigen::VectorXd _delta_weights;
        Eigen::VectorXd _residuals;
        Eigen::MatrixXd _Jtr;

        double _currentWeightedCost;
        double _currentRobustCost;


};
                        





#endif