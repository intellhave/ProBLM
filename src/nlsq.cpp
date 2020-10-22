#include "nlsq.h"
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

//----------------------------------------------------------------------------------------------
TwoViewOptimizer::TwoViewOptimizer(std::vector<Vec3> const &x1, std::vector<Vec3> const & x2,
                    TwoViewModel &model, Eigen::VectorXd &params, Config const &config, Stat &stat):
                    _x1(x1), _x2(x2), _params(params), _config(config), _stat(stat), _model(model)
                    
{
    _maxIter = config.maxIter;
    _innerMaxIter = config.innerMaxIter;
    
    _nMeasurements = x1.size();
    _modelDimension = (int) params.size();

    _grad.resize(_modelDimension);    
    _delta.resize(_modelDimension);
    _JtJ.resize(_modelDimension,_modelDimension);
      

    _grad.setZero();
    _JtJ.setZero();
}
                
//----------------------------------------------------------------------------------------------
void TwoViewOptimizer::saveParams(Eigen::VectorXd &storeVector)
{  
  for (int i = 0; i < _modelDimension; ++i)
    storeVector[i] = _params[i];

}

//----------------------------------------------------------------------------------------------
int TwoViewOptimizer::getInliers(double const &inlierThreshold, Eigen::VectorXd const &residuals, std::vector<int> &inlierSet)
{
  int nInliers = 0;
  inlierSet.clear();
  for (int i = 0; i < _nMeasurements; ++i)
    if (abs(residuals[i]) <= inlierThreshold)
    {
      nInliers++;
      inlierSet.push_back(i); 
    }
  return nInliers;
}

//----------------------------------------------------------------------------------------------
void TwoViewOptimizer::restoreParams(Eigen::VectorXd const &storeVector)
{
  for (int i = 0; i < _modelDimension; ++i)
    _params[i] = storeVector[i];
}

//----------------------------------------------------------------------------------------------
void TwoViewOptimizer::computeE()
{
    _model.computeModel();  
}

//----------------------------------------------------------------------------------------------
void TwoViewOptimizer::computeTx()
{
    double t = _params[3], p = _params[4];
    double tx = sin(t) * cos(p);
    double ty = sin(t) * sin(p);
    double tz = cos(t);
    _Tx << 0, -tz, ty,   tz, 0, -tx,  -ty, tx, 0;
}
//----------------------------------------------------------------------------------------------
double TwoViewOptimizer::sampson_error(int i)
{
    return _model.error(i);
}

//----------------------------------------------------------------------------------------------
void TwoViewOptimizer::computeJacobian(int i, Eigen::VectorXd &J)
{
    _model.computeJacobian(i, J);
 
}
//----------------------------------------------------------------------------------------------
double TwoViewOptimizer::compute_cost(bool fillGrad)
{
    double total_cost = 0.0;
    computeE();

    if (fillGrad) {  _grad.setZero();   _JtJ.setZero(); }

    VectorXd Ji;  Ji.resize(_modelDimension);
    for (int  i = 0; i < _nMeasurements; ++i)
    {
      double fid = sampson_error(i);
      total_cost += fid * fid;

      if (fillGrad)
      {         
        computeJacobian(i, Ji);    
        _grad += fid * Ji;  
        _JtJ += Ji * Ji.transpose();
      }
    }
    return total_cost;
}
//----------------------------------------------------------------------------------------------
void TwoViewOptimizer::solveDelta()
{
    _JtJ += _lambda * Eigen::MatrixXd::Identity(_modelDimension,_modelDimension);    
    _delta = _JtJ.ldlt().solve( (-1)*_grad);      
    updateParameters();
}
//----------------------------------------------------------------------------------------------
void TwoViewOptimizer::updateParameters()
{
    _params += _delta;
}
//----------------------------------------------------------------------------------------------
void TwoViewOptimizer::retractSolution()
{
    _JtJ -= _lambda * Eigen::MatrixXd::Identity(_modelDimension,_modelDimension);
    _params -= _delta;  
}

//----------------------------------------------------------------------------------------------
void TwoViewOptimizer::optimize()
{
    double init_cost;
    _lambda = 1e-3;
    auto start_time = std::chrono::steady_clock::now();
    auto end_time  = std::chrono::steady_clock::now();

    for (int iter = 0; iter < _config.maxIter; ++iter)
    {                 
        init_cost = compute_cost(true);
        end_time = std::chrono::steady_clock::now();                
        int elapsed_time = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();               
        _stat.Log(elapsed_time*1e-6, init_cost);       
        start_time = std::chrono::steady_clock::now();         

        double new_cost;        
        std::cout  << "Iter=  " << iter  << " Init cost = " << init_cost << " Lambda = " << _lambda << endl;

        while (true){
          solveDelta();
          new_cost = compute_cost(false);          
          if (abs(init_cost - new_cost) < 1e-6)
            break;

          if (new_cost < init_cost)          
          {    
            _lambda = max(_lambda * 0.1, 1e-12);
            break;
          }
          else{
            retractSolution();                
            _lambda *= 10;
          }
        }            

        if (abs(init_cost - new_cost) < 1e-6)
          break;
    }
}
