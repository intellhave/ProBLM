#include "common_utils.h"
#include "nlsq.h"
#include "stochastic_nlsq.h"
#include "twoview_models.h"
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main(int argvc, char **argv) {

  // Preapre the vector to store the correspondences
  std::vector<Vec3> x1;
  std::vector<Vec3> x2;

  /* std::string file_name = "./data/putative.txt"; */
  std::string file_name = std::string(argv[1]);

  // Parameters
  Eigen::VectorXd params;
  Eigen::VectorXd init_params;

  params.resize(5);
  init_params.resize(5);

  Config config;
  config.sampling_rate = 0.1;
  config.prob = 0.1;
  config.inlierThreshold = 0.1;
  config.innerMaxIter = 100;
  config.alpha = 0.9;

  //Read data 
  readData(file_name, x1, x2, init_params);
  printf("Data Read. Size = %ld %ld \n", x1.size(), x2.size());

  //Random initialization
  init_params.setRandom();
  std::string run_str = "test_run";

  for (int i = 0; i < 5; ++i)
    params[i] = init_params[i];
  // Prepare a two-view model
  EssentialModel model(x1, x2, params, config);
  

  //Normal LM optimization (this is for comparison)
  Stat opt_stat;
  TwoViewOptimizer *opt =
      new TwoViewOptimizer(x1, x2, model, params, config, opt_stat);
  
  //Comment out this if you don't want to run LM 
  opt->optimize();
  
  //Uncomment this if you would like to log cost vs runtime
  /* std::string opt_log_path = "./logs/lm/" + run_str + ".txt"; */
  /* opt_stat.WriteToFile(opt_log_path); */

  cout << " -----------Finished LM. Now running ProBLM-------\n";
  //Prepare ProBLM
  for (int i = 0; i < 5; ++i)
    params[i] = init_params[i];
 
  config.sampling_rate = 0.1;
  Stat sto_stat;
  for (int i = 0; i < 5; ++i)
    params[i] = init_params[i];
  StochasticTwoViewOptimizer *sto_opt =
      new StochasticTwoViewOptimizer(x1, x2, model, params, config, sto_stat);

  sto_opt->optimize();
  /* sto_opt->optimize_relaxed();//Relaxed condition */

  /* std::string sto_log_path = "./logs/sto/" + run_str + ".txt"; */
  /* sto_stat.WriteToFile(sto_log_path); */
  
 }
