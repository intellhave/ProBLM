#ifndef ROBUST_NLSQ_GNC
#define ROBUST_NLSQ_GNC

#include "common_utils.h"
#include "twoview_models.h"
#include "nlsq.h"
#include <Eigen/Core>
#include <Eigen/Dense>

#include<random>
#include<ctime>
#include<chrono>
#include<vector>



struct RobustTwoViewOptimizer_GNC: public TwoViewOptimizer
{

    RobustTwoViewOptimizer_GNC(std::vector<Vec3> const &x1, std::vector<Vec3> const & x2, 
                            TwoViewModel &model,Eigen::VectorXd &params, Config const &config, 
                            Stat &stat);

    
    int countInliers();

    virtual void optimize();

    virtual double compute_cost(bool fillGrad = true);
    
    // void compute_weights(int i);
    // virtual void compute_scaled_errors(double scale);
    // virtual void compute_level_cost(int level);

    
    std::vector<int> _inlierSet;


    protected:
        Eigen::VectorXd _weights;
        Eigen::VectorXd _residuals;
        Eigen::VectorXd _costs;
        Eigen::VectorXd _errors;
        Eigen::VectorXd _etas, _etas2;

        double _inlierThreshold = 0.1;    
        int _currentLevel;        
        double _eta_multiplier = 2.0;        
        int _NLevels;
        double _tau2;
        double _alpha = 1.0;
        double _levelCost, _currentRobustCost; 
};


#endif