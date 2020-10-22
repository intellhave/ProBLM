// -*- C++ -*-
#ifndef COMMON_UTILS_H
#define COMMON_UTILS_H

#include<iostream>
#include<vector>
#include<string>
#include<Eigen/Sparse>
#include<Eigen/Core>
#include<fstream>
#include <random>

using namespace std;
using namespace Eigen;

typedef Eigen::Vector2d Vec2;
typedef Eigen::Vector3d Vec3;
typedef Eigen::Matrix3d Mat3;

//Spare matrix
typedef Eigen::SparseMatrix<double> SparseMat;

//Borrowed from V3D
inline double sqr(double const x) { return x*x; }
inline double cube(double const x) { return x*x*x; }
inline double kappa(double const tau, double const w2) { return 0.70710678118*tau*(w2 - 1); }
inline double dkappa(double const tau, double const w2) { return 0.70710678118*tau; }

// ------------------------------------------------------------------------------
inline double rho(double const tau2, double const r2)
{
  return (r2 < tau2) ? (r2*(1.0 - r2/2.0/tau2)/2.0) : (tau2/4.0);
}

template <typename T>
inline void copySubsetVector(std::vector<T> const &src, std::vector<int> const &subset_idx,
        std::vector<T> &dst)
{
    dst.clear();
    for (int i = 0; i < subset_idx.size(); ++i)
    {
        dst.push_back(src[subset_idx[i]]);
    }
}
// ------------------------------------------------------------------------------
struct Psi_SmoothTrunc
{
   static double fun(double r2) { return 0.25 * ((r2 <= 1.0) ? r2 * (2.0 - r2) : 1.0); }
   static double weight_fun(double r2) { return std::max(0.0, 1.0 - r2); }
   static double weight_fun_deriv(double r2) { return (r2 <= 1.0) ? -2.0 * sqrt(r2) : 0.0; }
   static double gamma_omega_fun(double r2) { return 0.25 * sqr(Psi_SmoothTrunc::weight_fun(r2) - 1.0); }
   static double get_convex_range() { return sqrt(0.333333333); }
};

// ------------------------------------------------------------------------------
static void randomSampling(int N, int sampleSize, std::vector<int>& sampled_indices)
{
    if (sampleSize > N)
      sampleSize = N;

    std::vector<int> all_indices;
    for (int i = 0; i < N; ++i)
      all_indices.push_back(i);
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(all_indices.begin(), all_indices.end(), g);

    sampled_indices.clear();
    for (int i = 0; i < sampleSize; ++i)
      sampled_indices.push_back(all_indices[i]);
}

// ------------------------------------------------------------------------------
// Conducting basic random sampling 
struct Sampler
{
    Sampler(int const n): _n(n)
    {
        randomSampling(n, n, _sampled_indices);
    }

    void fillSamples(int const sampleSize, std::vector<int> &samples, bool reshuffle = false){
        //For this application, we do not need to reshuffle the indices.
        // Set reshuffle to true to conduct sampling again
        if (!reshuffle)
        {
            samples.clear();
            for (int i = 0; i < sampleSize; ++i)
                samples.push_back(_sampled_indices[i]);
        }
        else {
            randomSampling(_n, sampleSize, samples);
        }
    }

    protected:
     int _n;
     std::vector<int> _sampled_indices;
};

// ------------------------------------------------------------------------------
// Store necessary running statistic
struct Stat 
{
  std::vector<double> time;  
  std::vector<double> cost;
  std::vector<int> inls;

  void Log(double t, double c)
  {
    if (time.size() > 0)
    {
      t += time[time.size()-1];
    }
    time.push_back(t); cost.push_back(c);
  }  

  void Log(double t, int nInls )
  {    
    inls.push_back(nInls);
  }  

  void WriteToFile(std::string filename)
  {    
    double best_cost = 1e20;
    FILE *of = fopen(filename.c_str(), "w");
    fprintf(of, "%d\n", (int) time.size());    
    for (int i = 0; i < time.size(); ++i)    
    {
      if (cost[i] < best_cost)
        best_cost = cost[i];
      fprintf(of, "%.6f %.6f\n", time[i], best_cost);  
    }
    fclose(of);   
  }

  void WriteInlsToFile(std::string filename)
  {    
    int bestInls = 0;
    FILE *of = fopen(filename.c_str(), "w");
    fprintf(of, "%d\n", (int) time.size());    
    for (int i = 0; i < time.size(); ++i)    
    {
      if (inls[i] > bestInls)
        bestInls = inls[i];
      fprintf(of, "%.6f %d\n", time[i], bestInls);  
    }
    fclose(of);   
  }

  void Clear()
  {
    time.clear();
    cost.clear();
  }
}; //Store run statistics



// ------------------------------------------------------------------------------
struct Config
{
  double  sampling_rate = 0.1;
  double  prob = 1e-1;
  double  inlierThreshold = 0.1;

  //Sufficient reduction
  double alpha = 0.9;

  int GNC_levels = 10;

  int maxIter = 100;
  int innerMaxIter  = 5; 
};

// ------------------------------------------------------------------------------
static std::string RandomStringFromTime()
{
  time_t rawtime;
  char buffer[80];
  struct tm * timeinfo;
  time (&rawtime);
  timeinfo = localtime(&rawtime);
  strftime(buffer, sizeof(buffer), "%d%m%y%H%M%S", timeinfo);
  std::string str(buffer);
  return str;
}

// ------------------------------------------------------------------------------
static void readData(std::string const &filename, std::vector<Vec3> &x1, std::vector<Vec3> &x2, Eigen::VectorXd &params)
{
  ifstream inp(filename.c_str());
  int N;  inp >> N;
  for (int i=0; i < N; ++i)
  {
    // std::cout << "Reading correspondences " << i << "/" << N << endl;
    Vec3 u, v; 
    double ux, uy, vx, vy;
    inp >> ux >> uy >> vx >> vy;
    u << ux, uy, 1.0;   v << vx, vy, 1.0;
    x1.push_back(u);    x2.push_back(v);
  }

  // Read initial solutions, if available
  // double a, b, c, t, p;
  // inp >> a >> b >> c >> t >> p;
  // params << a, b, c, t, p;
}

// ------------------------------------------------------------------------------

typedef Psi_SmoothTrunc Psi;
#endif
