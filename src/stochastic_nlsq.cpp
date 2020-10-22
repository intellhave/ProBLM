#include "stochastic_nlsq.h"
#include "nlsq.h"
#include <cmath>
#include "common_utils.h"
#include <iostream>

using namespace std;

StochasticTwoViewOptimizer::StochasticTwoViewOptimizer(std::vector<Vec3> const &x1, std::vector<Vec3> const &x2, TwoViewModel &model,
                                                       Eigen::VectorXd &params, Config const &config, Stat &stat) : TwoViewOptimizer(x1, x2, model, params, config, stat),
    _sampler(x1.size())
{
}

//-----------------------------------------------------------------------------------------------------
double StochasticTwoViewOptimizer::computeSubsetCost(vector<double> &res, bool fillGrad)
{
  double total_cost = 0.0;
  _model.computeModel();
  if (fillGrad) { _grad.setZero();  _JtJ.setZero(); }
  res.clear();
  Eigen::VectorXd Ji;  Ji.resize(_modelDimension);

  int sample_size = _sampled_idx.size();

  /* int sample_size = 1; */
  for (int j = 0; j < _sampled_idx.size(); ++j)
  {
    int i = _sampled_idx[j];
    double fid = sampson_error(i);
    total_cost += fid * fid;
    res.push_back(fid * fid);
    if (fillGrad)
    {      
      computeJacobian(i, Ji);
      _grad += (1.0 / sample_size) * fid * Ji;
      _JtJ += (1.0 / sample_size) * Ji * Ji.transpose();      
    }
  }

  return total_cost;
}

//-----------------------------------------------------------------------------------------------------
void StochasticTwoViewOptimizer::optimize_v3()
{

  int sampleSize = (int)(_config.sampling_rate * _nMeasurements);
  vector<int> sampled_idx;
  /* randomSampling(_nMeasurements, sampleSize, _sampled_idx); */
  _sampler.fillSamples(sampleSize, _sampled_idx);
  vector<double> res, init_res, new_res;

  Eigen::VectorXd savedParams;
  savedParams.resize(_modelDimension);

  double init_all_cost;
  auto start_time = std::chrono::steady_clock::now();
  auto end_time = std::chrono::steady_clock::now();

  double alpha = _config.alpha;
  double ldt = log(_config.prob);
  bool reset = true;

  double init_cost, new_cost, total_reduction;
  int innerIter = 0;
  bool converged = false;
  for (int iter = 0; iter < _maxIter; ++iter)
  {
    bool converged = false;
    for (int in_iter = 0; in_iter < _innerMaxIter; ++in_iter)
    {
      init_cost = computeSubsetCost(res, true);

      // saveParams(savedParams);
      if (reset)
      {
        init_res.clear();
        for (int i = 0; i < res.size(); ++i)
          init_res.push_back(res[i]);        
        total_reduction = 0.0;
        innerIter = 0;       
      }
      end_time = std::chrono::steady_clock::now();
      int elapsed_time = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();
      init_all_cost = compute_cost(false);
      _stat.Log(elapsed_time * 1e-6, init_all_cost);

      cout << "Iter=  " << iter << " Inner Iter = " << in_iter << " Sample Size = " << _sampled_idx.size() << " Subset cost = " << init_cost
           << " Total cost = " << init_all_cost << " Lambda = " << _lambda << endl;

      start_time = std::chrono::steady_clock::now();

      converged = false;
      while (true)
      {
        solveDelta();
        new_cost = computeSubsetCost(new_res, false);
        
        //Check for convergence
        if (abs(new_cost - init_cost) < 1e-8)     { converged = true;      break;  }

        if (new_cost >= init_cost)                
        { retractSolution();    _lambda *= 10; }
        else
        {
          _lambda = max(_lambda * 0.1, 1e-8);
          innerIter++;
          total_reduction += new_cost - init_cost;
          break;
        }
      }
      //Break if inner loop converged
      if (converged)
      {
        break;
      }
    } //end for in_iter

    // double const T = sqrt(innerIter);
    //Now, check if the descent obtained from the subset also reduce the original cost with high probability
    
    if (_sampled_idx.size() < _nMeasurements)
    {
      std::vector<double> res_diff;
      double Uk = 0.0, maxDiff = 0.0;
      for (int i = 0; i < sampleSize; ++i)
      {
        double diff = new_res[i] - init_res[i];
        Uk += diff;
        if (diff < 0)
          res_diff.push_back(diff);
        else
        {
          if (diff > maxDiff)
            maxDiff = diff;
        }
      }
      std::sort(res_diff.begin(), res_diff.end());

      bool ok = false;
      double a, b, acc_sum = 0.0;

      b = maxDiff;
      int minSize = _nMeasurements;
      Uk = total_reduction;

      for (int i = 0; i < res_diff.size(); ++i)
      {
        acc_sum += res_diff[i];
        a = res_diff[i];

        double const uk = Uk - acc_sum + (i + 1) * a;
        double const  t = (1- alpha)/(b-a);
        double const uk_bound = -(t * sqrt(-0.5 * sampleSize * ldt));
        int const S = (int) (0.5* sqr(sampleSize/Uk) * sqr(t) * (-ldt) );
        
        
        // uk = uk / T;
        if (uk <= uk_bound)
        {
          printf("i = %d a = %.4f Sk = %.4f bound = %.4f \n", i, a, uk, uk_bound);
          ok = true;
          break;
        }

        else
        {
          if (uk < 0 && S < minSize && S > sampleSize)
          {
            minSize = S;
          }
        }

        if (uk > 0)
        {
          cout << " break sk " << minSize << endl;
          break;
        }
      }

      if (!ok)
      {
        int coin =  rand() % 10;
        if (coin == 0)
        {
          retractSolution();
          sampleSize = min(minSize, _nMeasurements);
          _sampler.fillSamples(sampleSize, _sampled_idx);
          /* randomSampling(_nMeasurements, sampleSize, _sampled_idx); */
          reset = true;
        }
      }
    }
    else
    {
      if (converged) break;
    }
    
  }
}

//-----------------------------------------------------------------------------------------------------
void StochasticTwoViewOptimizer::optimize_v2()
//Relaxed condition
{
  int sampleSize = (int)(_config.sampling_rate * _nMeasurements);
  vector<int> sampled_idx;
  randomSampling(_nMeasurements, sampleSize, _sampled_idx);
  vector<double> res, init_res, new_res;

  Eigen::VectorXd savedParams;
  savedParams.resize(_modelDimension);

  double init_all_cost;
  auto start_time = std::chrono::steady_clock::now();
  auto end_time = std::chrono::steady_clock::now();

  double ldt = log(_config.prob);
  bool reset = true;
  //Optimizer the subset for T iterations
  double init_cost, new_cost, total_reduction;
  int innerIter = 0;
  for (int iter = 0; iter < _maxIter; ++iter)
  {

    bool converged = false;
    for (int in_iter = 0; in_iter < _innerMaxIter; ++in_iter)
    {
      init_cost = computeSubsetCost(res, true);

      if (reset)
      {
        init_res.clear();
        for (int i = 0; i < res.size(); ++i)
          init_res.push_back(res[i]);
        saveParams(savedParams);
        total_reduction = 0.0;
        innerIter = 0;
      }

      end_time = std::chrono::steady_clock::now();
      int elapsed_time = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

      init_all_cost = compute_cost(false);
      _stat.Log(elapsed_time * 1e-6, init_all_cost);

      cout << "Iter=  " << iter << " Inner Iter = " << in_iter << " Sample Size = " << _sampled_idx.size() << " Subset cost = " << init_cost
           << " Total cost = " << init_all_cost << " Lambda = " << _lambda << endl;

      bool converged = false;
      while (true)
      {
        solveDelta();
        new_cost = computeSubsetCost(new_res, false);
        //Check for convergence
        if (abs(new_cost - init_cost) < 1e-8)
        {
          converged = true;
          break;
        }

        if (new_cost >= init_cost)
        {
          retractSolution();
          _lambda *= 10;
        }
        else
        {
          _lambda = max(_lambda * 0.1, 1e-8);
          innerIter++;
          total_reduction += new_cost - init_cost;
          break;
        }
      }
      //Break if inner loop converged
      if (converged)
      {
        break;
      }
    } //end for in_iter

    double const T = sqrt(innerIter);

    //Now, check if the descent obtained from the subset also reduce the original cost with high probability
    if (!converged)
    {
      if (_sampled_idx.size() < _nMeasurements)
      {
        std::vector<double> res_diff; //For searching a
        double Uk = 0.0;
        double maxDiff = 0.0;

        for (int i = 0; i < sampleSize; ++i)
        {
          double diff = new_res[i] - init_res[i];
          Uk += diff;
          if (diff < 0)
            res_diff.push_back(diff);
          else
          {
            if (diff > maxDiff)
              maxDiff = diff;
          }
        }
        std::sort(res_diff.begin(), res_diff.end());

        bool ok = false;
        double a, b, acc_sum = 0.0;
        b = maxDiff;
        int minSize = _nMeasurements;
        Uk = total_reduction;

        for (int i = 0; i < res_diff.size(); ++i)
        {
          acc_sum += res_diff[i];
          a = res_diff[i];

          double uk = Uk - acc_sum + (i + 1) * a;
          
          double uk_bound = -sqrt(-0.5 * sampleSize * (b - a) * (b - a) * ldt);
          int S = (int)(-0.5 * T * T * sampleSize * sampleSize * (b - a) * (b - a) * ldt / (total_reduction * total_reduction));
          // int S = (int)(-2 * uk * uk / (T * T * (b - a) * (b - a) * ldt ));
          // std::cout << uk << " -- " << S << endl;

          uk = uk / T;
          if (uk < uk_bound)
          {
            printf("i = %d a = %.4f Sk = %.4f bound = %.4f \n", i, a, uk, uk_bound);
            ok = true;
            break;
          }

          else
          {
            if (uk < 0 && S < minSize && S > sampleSize)
            {
              minSize = S;
            }
          }

          if (uk > 0)
          {
            cout << " break sk " << minSize << endl;
            break;
          }
        }

        if (!ok)
        {

          restoreParams(savedParams);
          printf("a = %.4f sample size = %d \n", a, minSize);
          sampleSize = min(minSize, _nMeasurements);
          randomSampling(_nMeasurements, sampleSize, _sampled_idx);
          reset = true;
        }
      }
      else
      {
        break;
      }
    } //end if !converged
    else
    {
      std::cout << "Converged! Do something!!!!!";
    }
  }
}

//-----------------------------------------------------------------------------------------------------
void StochasticTwoViewOptimizer::optimize()
{

  int sampleSize = (int)(_config.sampling_rate * _nMeasurements);
  vector<int> sampled_idx;
  randomSampling(_nMeasurements, sampleSize, _sampled_idx);
  vector<double> init_res, new_res;
  double ldt = log(_config.prob);

  auto start_time = std::chrono::steady_clock::now();
  auto end_time = std::chrono::steady_clock::now();
  double init_all_cost, prev_cost;
  bool updated = false;
  Eigen::VectorXd prev_params;
  prev_params.resize(_modelDimension);

  for (int iter = 0; iter < _maxIter; ++iter)
  {
    double init_cost = computeSubsetCost(init_res, true);
    end_time = std::chrono::steady_clock::now();
    int elapsed_time = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    prev_cost = init_all_cost;
    init_all_cost = compute_cost(false);
    _stat.Log(elapsed_time * 1e-6, init_all_cost);

    if (abs(prev_cost - init_all_cost) < 1e-6 && _sampled_idx.size() == _nMeasurements && iter > 10)
    {
      printf("%.5f %.5f\n", prev_cost, init_all_cost);
      break;
    }
    updated = false;

    cout << "Iter=  " << iter << " Sample Size = " << _sampled_idx.size() << " Subset cost = " << init_cost
         << " Total cost = " << init_all_cost << " Lambda = " << _lambda << endl;

    double a, b;

    start_time = std::chrono::steady_clock::now();
    while (true)
    {

      solveDelta();
      // Compute the updated cost
      double new_cost = computeSubsetCost(new_res, false);
      if (abs(init_cost - new_cost) < 1e-6)
        break;

      //If the new solution leads to 
      if (new_cost < init_cost)
      {
        //Perform test if not all residuals are used
        if (_sampled_idx.size() < _nMeasurements)
        {
          std::vector<double> res_diff;
          double sZ = 0.0, maxdiff = 0;
          for (int i = 0; i < sampleSize; ++i)
          {
            double diff = new_res[i] - init_res[i];
            sZ += diff;
            if (diff < 0)
              res_diff.push_back(diff);
              
            if (abs(diff) > maxdiff)
              maxdiff = abs(diff);
          }
          std::sort(res_diff.begin(), res_diff.end());

          double acc_sum = 0;
          bool ok = false;
          double sk = sZ;
          int minSize = _nMeasurements;

          // Iterate through the samples to determine a and the best sample size
          b = maxdiff;
          for (int i = 0; i < res_diff.size(); ++i)
          {
            if (res_diff[i] > 0)
              break;

            acc_sum += res_diff[i];
            a = res_diff[i];
            double sk = sZ - acc_sum + (i + 1) * a;
            double sk_bound = -sqrt(-0.5 * sampleSize * (b - a) * (b - a) * ldt);
            int S = (int)(-0.5 * sampleSize * sampleSize * (b - a) * (b - a) * ldt / (sk * sk));

            // Found a value of a that satisfies the test
            if (sk < sk_bound)
            {
              ok = true; break;
            }
            else
            {
              if (sk < 0 && S < minSize && S > sampleSize)
                minSize = S;
            }

            if (sk > 0)
            {
              break;
            }
          }

          if (!ok)
          {
            retractSolution();
            printf("a = %.4f sample size = %d \n", a, minSize);
            sampleSize = min(minSize, _nMeasurements);
            _sampler.fillSamples(minSize, _sampled_idx);
            /* randomSampling(_nMeasurements, sampleSize, _sampled_idx); */
          }
          else
          {
            updated = true;
            _lambda = max(_lambda * 0.1, 1e-8);
          }
        } //end if (_sampled_idx.size() < _nMeasurements)
        else
        {
          updated = true;
          _lambda = max(_lambda * 0.1, 1e-8);
        }
        break;
      }
      else //if new_cost > init_cost
      {
        retractSolution();
        _lambda *= 10;
      }
    }
  }
}


//===============================================================================

//



