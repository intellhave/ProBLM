#include<iostream>
#include "robust_nlsq_gnc.h"


RobustTwoViewOptimizer_GNC::RobustTwoViewOptimizer_GNC(std::vector<Vec3> const &x1, std::vector<Vec3> const & x2, 
                            TwoViewModel &model,Eigen::VectorXd &params, Config const &config, 
                            Stat &stat): 
    TwoViewOptimizer(x1, x2, model, params, config, stat)
{
    
    _NLevels = config.GNC_levels;
    _inlierThreshold = config.inlierThreshold;
    _tau2 = sqr(_inlierThreshold);

    //Fill Etas;
    _etas.resize(_NLevels);
    _etas2.resize(_NLevels);
    _etas[0] =  1.0; _etas2[0] = 1.0;
    for (int i = 1; i < _NLevels; ++i)
    {   
        _etas[i] = _eta_multiplier * _etas[i-1];
        _etas2[i] = sqr(_etas[i]);
    }

    _residuals.resize(_nMeasurements);
    _errors.resize(_nMeasurements);
    _costs.resize(_nMeasurements);

}

int RobustTwoViewOptimizer_GNC::countInliers()
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

double RobustTwoViewOptimizer_GNC::compute_cost(bool fillGrad)
{
    double total_cost = 0.0;    
    _currentRobustCost = 0.0;    
    _levelCost = 0.0;

    //Compute Model;
    computeE();

    if (fillGrad) { _grad.setZero();   _JtJ.setZero();  }

    VectorXd Ji;
    Ji.resize(_modelDimension);

    double const scale2 = _etas2[_currentLevel];    
    double const tau2 = _tau2 * scale2, W = tau2;

    for (int  i = 0; i < _nMeasurements; ++i)
    {      
        double fid = sampson_error(i);
        double const r2 = fid * fid;

        _residuals[i] = fid;
        _errors[i] = r2;      
        _costs[i] = W * Psi::fun(r2/tau2);

        //Total Cost stores robust cost
        _currentRobustCost += rho(sqr(_inlierThreshold), r2);
        total_cost += _costs[i];

        if (fillGrad)
        {         
            computeJacobian(i, Ji);    
            // New way to compute JtJ 
            double wi =  Psi::weight_fun(r2/tau2);         
            
            _grad += wi * fid *  Ji;              

            // Eigen::MatrixXd C_J2 = C * Ji.transpose();
            _JtJ += wi * (Ji * Ji.transpose());
        }
    }
    return total_cost;  
}


void RobustTwoViewOptimizer_GNC::optimize()
{

    auto start_time = std::chrono::steady_clock::now();
    auto end_time  = std::chrono::steady_clock::now();

    for (int level = _NLevels-1; level >=0;  --level)
    {
        _currentLevel = level;

        for (int iter = 0; iter < _config.innerMaxIter;  ++iter)
        {
            double init_level_cost = compute_cost();
            
            end_time = std::chrono::steady_clock::now();                
            int nInls = countInliers();

            int elapsed_time = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();               
            _stat.Log(elapsed_time*1e-6, _currentRobustCost);       
            _stat.Log(elapsed_time*1e-6, nInls);
            
            std::cout << " Level = " << level  << " iter = " << iter << 
                          " Level cost = " << init_level_cost << 
                          " Robust Cost = "  << _currentRobustCost << 
                          " Inls = " << nInls << endl;
            
            start_time = std::chrono::steady_clock::now();         
            double new_level_cost;
            while (true)
            {   
                solveDelta();
                new_level_cost = compute_cost(false);
                if (abs(init_level_cost - new_level_cost) < 1e-6)
                    break;

                if (new_level_cost < init_level_cost)          
                {    
                    _lambda = max(_lambda * 0.1, 1e-8);
                    break;
                }
                else{
                    retractSolution();                
                    _lambda *= 10;
                }
            }

            if (abs(init_level_cost - new_level_cost) < 1e-6)
                break;
                        
        }
    }

}
