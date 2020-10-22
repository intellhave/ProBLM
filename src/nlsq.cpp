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

    _Rx = Eigen::Matrix3d(); _Rx.setZero(); 
    _Rx(0,0) = 1.0; 
    _Ry = Eigen::Matrix3d(); _Ry.setZero(); 
    _Ry(1,1) = 1.0;
    _Rz = Eigen::Matrix3d(); _Rz.setZero();
    _Rz(2,2) = 1.0;

    _Tx = Mat3::Zero();

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

    // //Compute the Sampson Error following: https://cseweb.ucsd.edu/classes/sp04/cse252b/notes/lec11/lec11.pdf
    // Vec3 u = _x1[i];
    // Vec3 v = _x2[i];
    // const double num = v.transpose() * _E * u;
    // double den = (_E.row(0) * u).dot(_E.row(0) * u) + (_E.row(1) * u).dot(_E.row(1) * u); 
    // den += (v.transpose()*_E.col(0)).dot(v.transpose()*_E.col(0)) + (v.transpose()*_E.col(1)).dot(v.transpose()*_E.col(1)); 
    // den = sqrt(den);

    // double temp = _model.error(i);
    // double temp2 = num/den; 
    // std::cout << temp << "--" << temp2 << "\t";
    // return num/den;      
}

//----------------------------------------------------------------------------------------------
void TwoViewOptimizer::computeJacobian(int i, Eigen::VectorXd &J)
{
    _model.computeJacobian(i, J);
    // const double u0 = _x1[i][0], u1 = _x1[i][1], u2 = 1.0;
    // const double v0 = _x2[i][0], v1 = _x2[i][1], v2 = 1.0;
    // const double a = _params[0], b = _params[1], g = _params[2];
    // const double t = _params[3], p = _params[4];

    // J[0] = (u0*(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*cos(t) + (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(p)*sin(t)) + v1*(-sin(a)*sin(b)*cos(g) - sin(g)*cos(a))*sin(t)*cos(p) + v2*(-sin(a)*sin(g) + sin(b)*cos(a)*cos(g))*sin(t)*cos(p)) + u1*(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*cos(t) + (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(p)*sin(t)) + v1*(sin(a)*sin(b)*sin(g) - cos(a)*cos(g))*sin(t)*cos(p) + v2*(-sin(a)*cos(g) - sin(b)*sin(g)*cos(a))*sin(t)*cos(p)) + u2*(v0*(-sin(a)*sin(p)*sin(t)*cos(b) + cos(a)*cos(b)*cos(t)) + v1*sin(a)*sin(t)*cos(b)*cos(p) - v2*sin(t)*cos(a)*cos(b)*cos(p)))*std::pow(std::pow(u0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + u1*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + u2*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b)), 2) + std::pow(u0*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + u1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + u2*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p)), 2) + std::pow(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g)), 2) + std::pow(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b)), 2), -0.5) + (u0*(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g))) + u1*(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b))) + u2*(v0*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b)) + v1*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p)) + v2*(-sin(a)*sin(t)*cos(b)*cos(p) - sin(b)*sin(p)*sin(t))))*(-0.5*(2*u0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*cos(t) + (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(p)*sin(t)) + 2*u1*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*cos(t) + (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(p)*sin(t)) + 2*u2*(-sin(a)*sin(p)*sin(t)*cos(b) + cos(a)*cos(b)*cos(t)))*(u0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + u1*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + u2*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b))) - 0.5*(u0*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + u1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + u2*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p)))*(2*u0*(-sin(a)*sin(b)*cos(g) - sin(g)*cos(a))*sin(t)*cos(p) + 2*u1*(sin(a)*sin(b)*sin(g) - cos(a)*cos(g))*sin(t)*cos(p) + 2*u2*sin(a)*sin(t)*cos(b)*cos(p)) - 0.5*(2*v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*cos(t) + (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(p)*sin(t)) + 2*v1*(-sin(a)*sin(b)*cos(g) - sin(g)*cos(a))*sin(t)*cos(p) + 2*v2*(-sin(a)*sin(g) + sin(b)*cos(a)*cos(g))*sin(t)*cos(p))*(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g))) - 0.5*(2*v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*cos(t) + (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(p)*sin(t)) + 2*v1*(sin(a)*sin(b)*sin(g) - cos(a)*cos(g))*sin(t)*cos(p) + 2*v2*(-sin(a)*cos(g) - sin(b)*sin(g)*cos(a))*sin(t)*cos(p))*(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b))))*std::pow(std::pow(u0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + u1*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + u2*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b)), 2) + std::pow(u0*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + u1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + u2*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p)), 2) + std::pow(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g)), 2) + std::pow(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b)), 2), -1.5);
    // J[1] = (u0*(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g))) + u1*(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b))) + u2*(v0*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b)) + v1*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p)) + v2*(-sin(a)*sin(t)*cos(b)*cos(p) - sin(b)*sin(p)*sin(t))))*(-0.5*(u0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + u1*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + u2*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b)))*(2*u0*(-sin(a)*cos(b)*cos(g)*cos(t) - sin(p)*sin(t)*cos(a)*cos(b)*cos(g)) + 2*u1*(sin(a)*sin(g)*cos(b)*cos(t) + sin(g)*sin(p)*sin(t)*cos(a)*cos(b)) + 2*u2*(-sin(a)*sin(b)*cos(t) - sin(b)*sin(p)*sin(t)*cos(a))) - 0.5*(u0*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + u1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + u2*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p)))*(2*u0*(-sin(b)*cos(g)*cos(t) + sin(t)*cos(a)*cos(b)*cos(g)*cos(p)) + 2*u1*(sin(b)*sin(g)*cos(t) - sin(g)*sin(t)*cos(a)*cos(b)*cos(p)) + 2*u2*(sin(b)*sin(t)*cos(a)*cos(p) + cos(b)*cos(t))) - 0.5*(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g)))*(2*v0*(-sin(a)*cos(b)*cos(g)*cos(t) - sin(p)*sin(t)*cos(a)*cos(b)*cos(g)) + 2*v1*(-sin(b)*cos(g)*cos(t) + sin(t)*cos(a)*cos(b)*cos(g)*cos(p)) + 2*v2*(sin(a)*sin(t)*cos(b)*cos(g)*cos(p) + sin(b)*sin(p)*sin(t)*cos(g))) - 0.5*(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b)))*(2*v0*(sin(a)*sin(g)*cos(b)*cos(t) + sin(g)*sin(p)*sin(t)*cos(a)*cos(b)) + 2*v1*(sin(b)*sin(g)*cos(t) - sin(g)*sin(t)*cos(a)*cos(b)*cos(p)) + 2*v2*(-sin(a)*sin(g)*sin(t)*cos(b)*cos(p) - sin(b)*sin(g)*sin(p)*sin(t))))*std::pow(std::pow(u0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + u1*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + u2*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b)), 2) + std::pow(u0*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + u1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + u2*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p)), 2) + std::pow(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g)), 2) + std::pow(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b)), 2), -1.5) + (u0*(v0*(-sin(a)*cos(b)*cos(g)*cos(t) - sin(p)*sin(t)*cos(a)*cos(b)*cos(g)) + v1*(-sin(b)*cos(g)*cos(t) + sin(t)*cos(a)*cos(b)*cos(g)*cos(p)) + v2*(sin(a)*sin(t)*cos(b)*cos(g)*cos(p) + sin(b)*sin(p)*sin(t)*cos(g))) + u1*(v0*(sin(a)*sin(g)*cos(b)*cos(t) + sin(g)*sin(p)*sin(t)*cos(a)*cos(b)) + v1*(sin(b)*sin(g)*cos(t) - sin(g)*sin(t)*cos(a)*cos(b)*cos(p)) + v2*(-sin(a)*sin(g)*sin(t)*cos(b)*cos(p) - sin(b)*sin(g)*sin(p)*sin(t))) + u2*(v0*(-sin(a)*sin(b)*cos(t) - sin(b)*sin(p)*sin(t)*cos(a)) + v1*(sin(b)*sin(t)*cos(a)*cos(p) + cos(b)*cos(t)) + v2*(sin(a)*sin(b)*sin(t)*cos(p) - sin(p)*sin(t)*cos(b))))*std::pow(std::pow(u0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + u1*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + u2*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b)), 2) + std::pow(u0*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + u1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + u2*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p)), 2) + std::pow(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g)), 2) + std::pow(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b)), 2), -0.5);
    // J[2] = (u0*(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) + (sin(a)*sin(b)*sin(g) - cos(a)*cos(g))*cos(t)) + v1*((-sin(a)*cos(g) - sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b))) + u1*(v0*((-sin(a)*sin(g) + sin(b)*cos(a)*cos(g))*sin(p)*sin(t) + (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) - cos(b)*cos(g)*cos(t)) + v2*((-sin(a)*sin(b)*cos(g) - sin(g)*cos(a))*sin(t)*cos(p) + sin(p)*sin(t)*cos(b)*cos(g))))*std::pow(std::pow(u0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + u1*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + u2*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b)), 2) + std::pow(u0*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + u1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + u2*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p)), 2) + std::pow(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g)), 2) + std::pow(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b)), 2), -0.5) + (u0*(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g))) + u1*(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b))) + u2*(v0*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b)) + v1*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p)) + v2*(-sin(a)*sin(t)*cos(b)*cos(p) - sin(b)*sin(p)*sin(t))))*(-0.5*(2*u0*((-sin(a)*cos(g) - sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + 2*u1*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) - cos(b)*cos(g)*cos(t)))*(u0*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + u1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + u2*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p))) - 0.5*(2*u0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) + (sin(a)*sin(b)*sin(g) - cos(a)*cos(g))*cos(t)) + 2*u1*((-sin(a)*sin(g) + sin(b)*cos(a)*cos(g))*sin(p)*sin(t) + (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)))*(u0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + u1*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + u2*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b))) - 0.5*(2*v0*((-sin(a)*sin(g) + sin(b)*cos(a)*cos(g))*sin(p)*sin(t) + (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + 2*v1*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) - cos(b)*cos(g)*cos(t)) + 2*v2*((-sin(a)*sin(b)*cos(g) - sin(g)*cos(a))*sin(t)*cos(p) + sin(p)*sin(t)*cos(b)*cos(g)))*(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b))) - 0.5*(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g)))*(2*v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) + (sin(a)*sin(b)*sin(g) - cos(a)*cos(g))*cos(t)) + 2*v1*((-sin(a)*cos(g) - sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + 2*v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b))))*std::pow(std::pow(u0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + u1*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + u2*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b)), 2) + std::pow(u0*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + u1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + u2*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p)), 2) + std::pow(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g)), 2) + std::pow(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b)), 2), -1.5);
    // J[3] = (u0*(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g))) + u1*(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b))) + u2*(v0*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b)) + v1*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p)) + v2*(-sin(a)*sin(t)*cos(b)*cos(p) - sin(b)*sin(p)*sin(t))))*(-0.5*(2*u0*((-sin(a)*sin(g) + sin(b)*cos(a)*cos(g))*cos(p)*cos(t) - sin(t)*cos(b)*cos(g)) + 2*u1*((-sin(a)*cos(g) - sin(b)*sin(g)*cos(a))*cos(p)*cos(t) + sin(g)*sin(t)*cos(b)) + 2*u2*(-sin(b)*sin(t) - cos(a)*cos(b)*cos(p)*cos(t)))*(u0*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + u1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + u2*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p))) - 0.5*(u0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + u1*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + u2*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b)))*(2*u0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*cos(t) - (-sin(a)*sin(b)*cos(g) - sin(g)*cos(a))*sin(t)) + 2*u1*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*cos(t) - (sin(a)*sin(b)*sin(g) - cos(a)*cos(g))*sin(t)) + 2*u2*(-sin(a)*sin(t)*cos(b) + sin(p)*cos(a)*cos(b)*cos(t))) - 0.5*(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g)))*(2*v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*cos(t) - (-sin(a)*sin(b)*cos(g) - sin(g)*cos(a))*sin(t)) + 2*v1*((-sin(a)*sin(g) + sin(b)*cos(a)*cos(g))*cos(p)*cos(t) - sin(t)*cos(b)*cos(g)) + 2*v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(p)*cos(t) - sin(p)*cos(b)*cos(g)*cos(t))) - 0.5*(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b)))*(2*v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*cos(t) - (sin(a)*sin(b)*sin(g) - cos(a)*cos(g))*sin(t)) + 2*v1*((-sin(a)*cos(g) - sin(b)*sin(g)*cos(a))*cos(p)*cos(t) + sin(g)*sin(t)*cos(b)) + 2*v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(p)*cos(t) + sin(g)*sin(p)*cos(b)*cos(t))))*std::pow(std::pow(u0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + u1*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + u2*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b)), 2) + std::pow(u0*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + u1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + u2*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p)), 2) + std::pow(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g)), 2) + std::pow(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b)), 2), -1.5) + (u0*(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*cos(t) - (-sin(a)*sin(b)*cos(g) - sin(g)*cos(a))*sin(t)) + v1*((-sin(a)*sin(g) + sin(b)*cos(a)*cos(g))*cos(p)*cos(t) - sin(t)*cos(b)*cos(g)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(p)*cos(t) - sin(p)*cos(b)*cos(g)*cos(t))) + u1*(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*cos(t) - (sin(a)*sin(b)*sin(g) - cos(a)*cos(g))*sin(t)) + v1*((-sin(a)*cos(g) - sin(b)*sin(g)*cos(a))*cos(p)*cos(t) + sin(g)*sin(t)*cos(b)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(p)*cos(t) + sin(g)*sin(p)*cos(b)*cos(t))) + u2*(v0*(-sin(a)*sin(t)*cos(b) + sin(p)*cos(a)*cos(b)*cos(t)) + v1*(-sin(b)*sin(t) - cos(a)*cos(b)*cos(p)*cos(t)) + v2*(-sin(a)*cos(b)*cos(p)*cos(t) - sin(b)*sin(p)*cos(t))))*std::pow(std::pow(u0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + u1*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + u2*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b)), 2) + std::pow(u0*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + u1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + u2*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p)), 2) + std::pow(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g)), 2) + std::pow(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b)), 2), -0.5);
    // J[4] = (u0*(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g))) + u1*(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b))) + u2*(v0*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b)) + v1*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p)) + v2*(-sin(a)*sin(t)*cos(b)*cos(p) - sin(b)*sin(p)*sin(t))))*(-0.5*(u0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + u1*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + u2*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b)))*(2*u0*(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + 2*u1*(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) + 2*u2*sin(t)*cos(a)*cos(b)*cos(p)) - 0.5*(u0*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + u1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + u2*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p)))*(-2*u0*(-sin(a)*sin(g) + sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - 2*u1*(-sin(a)*cos(g) - sin(b)*sin(g)*cos(a))*sin(p)*sin(t) + 2*u2*sin(p)*sin(t)*cos(a)*cos(b)) - 0.5*(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g)))*(2*v0*(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) - 2*v1*(-sin(a)*sin(g) + sin(b)*cos(a)*cos(g))*sin(p)*sin(t) + 2*v2*(-(sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(p)*sin(t) - sin(t)*cos(b)*cos(g)*cos(p))) - 0.5*(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b)))*(2*v0*(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - 2*v1*(-sin(a)*cos(g) - sin(b)*sin(g)*cos(a))*sin(p)*sin(t) + 2*v2*(-(-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(p)*sin(t) + sin(g)*sin(t)*cos(b)*cos(p))))*std::pow(std::pow(u0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + u1*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + u2*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b)), 2) + std::pow(u0*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + u1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + u2*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p)), 2) + std::pow(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g)), 2) + std::pow(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b)), 2), -1.5) + (u0*(v0*(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) - v1*(-sin(a)*sin(g) + sin(b)*cos(a)*cos(g))*sin(p)*sin(t) + v2*(-(sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(p)*sin(t) - sin(t)*cos(b)*cos(g)*cos(p))) + u1*(v0*(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - v1*(-sin(a)*cos(g) - sin(b)*sin(g)*cos(a))*sin(p)*sin(t) + v2*(-(-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(p)*sin(t) + sin(g)*sin(t)*cos(b)*cos(p))) + u2*(v0*sin(t)*cos(a)*cos(b)*cos(p) + v1*sin(p)*sin(t)*cos(a)*cos(b) + v2*(sin(a)*sin(p)*sin(t)*cos(b) - sin(b)*sin(t)*cos(p))))*std::pow(std::pow(u0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + u1*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + u2*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b)), 2) + std::pow(u0*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + u1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + u2*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p)), 2) + std::pow(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g)), 2) + std::pow(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b)), 2), -0.5);
 
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
    // init_cost = compute_cost(false);
    // _stat.Log(0.0, init_cost);
    _lambda = 1e-3;
    std::cout<<" RUNNING IN DEBUG MODE WITH LAMBDA = 0";
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
