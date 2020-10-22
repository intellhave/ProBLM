#ifndef NLSQ_H
#define NLSQ_H

#include "common_utils.h"
#include "twoview_models.h"
#include <Eigen/Core>
#include <Eigen/Dense>

#include<random>
#include<ctime>
#include<chrono>
#include<vector>

struct TwoViewOptimizer{

    TwoViewOptimizer(std::vector<Vec3> const &x1, std::vector<Vec3> const & x2, TwoViewModel &model,    
                    Eigen::VectorXd &params, Config const &config, Stat &stat);


    //Compute Essential Matrix
    virtual void computeE();
    virtual void computeTx();

    virtual double sampson_error(int i);
    virtual void optimize();

    //Compute Jacobian for the i-th residual
    virtual void computeJacobian(int i, Eigen::VectorXd &J);

    virtual double compute_cost(bool fillGrad = true);

    void solveDelta();

    void saveParams(Eigen::VectorXd &storeVector);
    void restoreParams(Eigen::VectorXd const &storeVector);

    virtual int getInliers(double const &inlierThreshold, Eigen::VectorXd const &residuals, std::vector<int> &inlierSet);
    
    virtual void updateParameters();

    virtual void retractSolution();


    protected:
        //Parameters
        TwoViewModel &_model;
        Eigen::VectorXd &_params;        
        Eigen::Matrix3d _Rx, _Ry, _Rz, _R, _E, _Tx;
        Eigen::VectorXd _grad;        
        Eigen::MatrixXd _JtJ;
        Eigen::VectorXd _delta;        
        double _lambda = 1e-3;
        int _nMeasurements;
        int _modelDimension;
        int _maxIter, _innerMaxIter;
        // Input Data        
        std::vector<Vec3> const &_x1;
        std::vector<Vec3> const &_x2;
        //Config and run statistic
        Config const &_config;
        Stat &_stat;

        //Model
        

};





#endif
