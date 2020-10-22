#ifndef TWOVIEW_MODELS_H
#define TWOVIEW_MODELS_H


#include "common_utils.h"
#include <Eigen/Core>
#include <Eigen/Dense>

#include<random>
#include<ctime>
#include<chrono>
#include<vector>


#include "Base/v3d_image.h"
#include "Math/v3d_nonlinlsq.h"



struct TwoViewModel
{
    
    TwoViewModel(std::vector<Vec3> const &x1, std::vector<Vec3> const &x2, Eigen::VectorXd const &params, Config const &config);

    //Residual (Could be re-projection or sampson error)
    virtual double error(int i);

    virtual void computeJacobian(int i, Eigen::VectorXd &J);

    virtual void computeModel(); //Compute Model based on params;

    virtual void setModel(Eigen::Matrix3d const &model);
    
    Eigen::Matrix3d _E;        
    Eigen::Matrix3d _Rx, _Ry, _Rz, _R, _Tx;    
    
    protected:
        //Image corrdinates of keypoints
        std::vector<Vec3> const &_x1;
        std::vector<Vec3> const &_x2;
        
        //Main Matrix Model (Essential, Fundamental, Homography,...)
        Eigen::VectorXd const &_params;
        
        Config const &_config;
        bool _freeze_model = false;

};


struct AffineRegModel: public TwoViewModel
{
    AffineRegModel(std::vector<Vec3> const &x1, std::vector<Vec3> const &x2, Eigen::VectorXd const &params, Config const &config, 
                   V3D::Image<float> const &im0, V3D::Image<float> const &im1, V3D::Image<V3D::Vector2f> const &grad_im0, V3D::Image<V3D::Vector2f> const &grad_im1);

    
    virtual double error(int i);

    virtual void computeModel(); //Compute Model based on params;

    virtual void computeJacobian(int i, Eigen::VectorXd &J);

    protected:
        V3D::Image<float> _im0, _im1;
        V3D::Image<V3D::Vector2f> _grad_im0, _grad_im1;


};

struct HomographyRegModel:public AffineRegModel
{
    HomographyRegModel(std::vector<Vec3> const &x1, std::vector<Vec3> const &x2, Eigen::VectorXd const &params, Config const &config, 
                   V3D::Image<float> const &im0, V3D::Image<float> const &im1, V3D::Image<V3D::Vector2f> const &grad_im0, V3D::Image<V3D::Vector2f> const &grad_im1);

    virtual double error(int i);

    virtual void computeModel(); //Compute Model based on params;

    void fillHomography(VectorXd const &params, V3D::Matrix3x3d &H);

    void extractParams(V3D::Matrix3x3d const& H, VectorXd &params);

    virtual void computeJacobian(int i, Eigen::VectorXd &J);

    virtual void writeHomographyMatrix(std::string file_name = "homo.txt");

};


struct EssentialModel: public TwoViewModel
{
    EssentialModel(std::vector<Vec3> const &x1, std::vector<Vec3> const &x2, Eigen::VectorXd const &params, Config const &config);

    virtual void computeModel(); //Compute Model based on params;

    virtual void computeJacobian(int i, Eigen::VectorXd &J);

    void computeTx();

    // protected: 
    // Eigen::Matrix3d _Rx, _Ry, _Rz, _R, _Tx;

};


struct FundamentalModel: public TwoViewModel
{
    FundamentalModel(std::vector<Vec3> const &x1, std::vector<Vec3> const &x2, Eigen::VectorXd const &params, Config const &config);

    virtual void computeModel(); //Compute Model based on params;

    virtual void computeJacobian(int i, Eigen::VectorXd &J);

    // protected: 
    // Eigen::Matrix3d _Rx, _Ry, _Rz, _R, _Tx;

};



#endif