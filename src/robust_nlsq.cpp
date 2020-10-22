//Not properly implemented
#include<iostream>
#include "robust_nlsq.h"
#include "nlsq.h"


RobustTwoViewOptimizer_Lifted::RobustTwoViewOptimizer_Lifted(std::vector<Vec3> const &x1, std::vector<Vec3> const & x2, TwoViewModel &model,
                                                             Eigen::VectorXd &params, Config const &config, Stat &stat):
         TwoViewOptimizer(x1, x2, model, params, config, stat)   
{
    // Weight Vector -- Initialize all weights to 1.0;
    _weights.resize(_nMeasurements);
    _residuals.resize(_nMeasurements);
    _delta_weights.resize(_nMeasurements);

    for (int i = 0; i < _nMeasurements; ++i)
    {
      _weights[i] = 1.0; _residuals[i] = 0.0;
    }
    // This matrix to store Jtr for all measurements;
    _Jtr.resize(5, _nMeasurements);
    _inlierThreshold = config.inlierThreshold;

}            


double RobustTwoViewOptimizer_Lifted::compute_cost(bool fillGrad)
{
    double total_cost = 0.0;
    _currentWeightedCost = 0.0;
    _currentRobustCost = 0.0;
    computeE();
    
    if (fillGrad)
    {
      _grad.setZero();   _JtJ.setZero();
    }

    VectorXd Ji;
    Ji.resize(5);
    for (int  i = 0; i < _nMeasurements; ++i)
    {
      double const w = _weights[i];
      double fid = sampson_error(i);
      _residuals[i] = fid;

      double const r2 = fid * fid;
      double const w2 = w * w;

      _currentWeightedCost += w * r2;
      _currentRobustCost += rho(sqr(_inlierThreshold), r2);

      total_cost += w * r2 + sqr(kappa(_inlierThreshold, w2));

      if (fillGrad)
      {         
        computeJacobian(i, Ji);    
        // New way to compute JtJ
                
        double const num = r2 + 2.0 * dkappa(_inlierThreshold, w2) * kappa(_inlierThreshold, w2);
        double const denom = r2 + 4.0 * w2 * sqr(dkappa(_inlierThreshold, w2)) + _lambda;
        double const f = w2 * (1.0 - num/denom);
        _grad += fid * f * Ji;  

        //Store this to compute the weight updates later
        _Jtr.col(i) = fid * Ji;

        //Outer product matrix in this case is the square of the residual
        double C = fid * fid;        
        double scale = -1.0/(r2 + 4.0 * w2 * sqr(dkappa(_inlierThreshold, w2)) + _lambda );
        C = scale * C + 1.0;
        C = w2 * C;
        // Eigen::MatrixXd C_J2 = C * Ji.transpose();
        _JtJ += Ji * (C * Ji.transpose());
      }
    }
    return total_cost;  

}

int RobustTwoViewOptimizer_Lifted::countInliers()
{
    int nInliers = 0;
    _inlierSet.clear();
    for (int i = 0; i < _nMeasurements; ++i)
      if (abs(_residuals[i]) <= _inlierThreshold)      
      {
        nInliers++;                
        _inlierSet.push_back(i);
      }
    return nInliers;
}

void RobustTwoViewOptimizer_Lifted::updateParameters()
{
    _params += _delta;    
    //Update the weights
    for ( int k = 0; k < _nMeasurements; ++k)
    {
      double const w = _weights[k], w2 = w * w, dw = 1.0, dw2  = 1.0;
      double const rk = _residuals[k], r2 = rk * rk;

      double const rt_J_delta = _delta.transpose() *_Jtr.col(k);
      double num = 2.0 * dkappa(_inlierThreshold, w2) * kappa(_inlierThreshold, w2) + r2 + rt_J_delta;
      double denom = 4.0 * w2 * sqr(dkappa (_inlierThreshold, w2)) + r2 + _lambda;
      double delta_w = -w/dw * num /denom;

      _delta_weights[k] = delta_w;
      _weights[k] += delta_w;
    }
}


void RobustTwoViewOptimizer_Lifted::retractSolution()
{
    _params -= _delta;
    for (int k = 0; k < _nMeasurements; ++k)
    _weights[k] -= _delta_weights[k];  
}

void RobustTwoViewOptimizer_Lifted::optimize()
{
    auto start_time = std::chrono::steady_clock::now();
    auto end_time  = std::chrono::steady_clock::now();

    for (int iter = 0; iter < _config.maxIter; ++iter)
    {                 
        double init_cost = compute_cost(true);
        end_time = std::chrono::steady_clock::now();                
        int elapsed_time = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();       
        
        _stat.Log(elapsed_time*1e-6, init_cost);       
        start_time = std::chrono::steady_clock::now();         

        double new_cost;        
        int inls = countInliers();
        cout  << "Iter=  " << iter  << " Total cost = " << init_cost 
              << " Robust Cost = " << _currentRobustCost << " Inls = " << inls <<  " Lambda = " << _lambda << endl;

        while (true){
          solveDelta();
          new_cost = compute_cost(false);          
          if (abs(init_cost - new_cost) < 1e-6)
            break;

          if (new_cost < init_cost)          
          {    
            _lambda = max(_lambda * 0.1, 1e-8);
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
