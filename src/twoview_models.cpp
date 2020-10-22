#include <iostream>
#include "twoview_models.h"
#include "image_utils.h"

TwoViewModel::TwoViewModel(std::vector<Vec3> const &x1, std::vector<Vec3> const &x2, Eigen::VectorXd const &params,
                          Config const &config):
                _x1(x1), _x2(x2), _params(params), _config(config)
{

}

double TwoViewModel::error(int i)
{
    //Compute the Sampson Error following: https://cseweb.ucsd.edu/classes/sp04/cse252b/notes/lec11/lec11.pdf
    Vec3 u = _x1[i];
    Vec3 v = _x2[i];
    const double num = v.transpose() * _E * u;
    double den = (_E.row(0) * u).dot(_E.row(0) * u) + (_E.row(1) * u).dot(_E.row(1) * u); 
    den += (v.transpose()*_E.col(0)).dot(v.transpose()*_E.col(0)) + (v.transpose()*_E.col(1)).dot(v.transpose()*_E.col(1)); 
    den = sqrt(den);
    return num/den;      
}

void TwoViewModel::setModel(Eigen::Matrix3d const &model)
{
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            _E(i,j) = model(i,j);

    _freeze_model = true;
}

void TwoViewModel::computeModel()
{
    
}

void TwoViewModel::computeJacobian(int i, Eigen::VectorXd &J)
{
    
}
//---------------------------------------------------------------------------------------------------------------------

AffineRegModel::AffineRegModel(std::vector<Vec3> const &x1, std::vector<Vec3> const &x2, Eigen::VectorXd const &params, Config const &config, 
                   V3D::Image<float> const &im0, V3D::Image<float> const &im1, V3D::Image<V3D::Vector2f> const &grad_im0, V3D::Image<V3D::Vector2f> const &grad_im1):
                   TwoViewModel(x1, x2, params, config), _im0(im0), _im1(im1), _grad_im0(grad_im0), _grad_im1(grad_im1)
{
        
}

double AffineRegModel::error(int i)
{
    float const x = _x1[i][0];
    float const y = _x1[i][1];

    float const xx = x + _params[0];
    float const yy = y + _params[1];

    double res = access_image(_im1, xx, yy) - _im0(x, y);
    return res; 

}

void AffineRegModel::computeModel() {}

void AffineRegModel::computeJacobian(int i, Eigen::VectorXd &J)
{
    float const x = _x1[i][0];
    float const y = _x1[i][1];

    float const xx = x + _params[0];
    float const yy = y + _params[1];


    V3D::Vector2f const grad0 = _grad_im0(x, y);
    V3D::Vector2f const grad1 = access_image(_grad_im1, xx, yy);
    V3D::Vector2f grad = 0.5f * (grad0 + grad1);

    J[0] = grad[0];
    J[1] = grad[1];

}

//---------------------------------------------------------------------------------------------------------------------
HomographyRegModel::HomographyRegModel(std::vector<Vec3> const &x1, std::vector<Vec3> const &x2, Eigen::VectorXd const &params, Config const &config, 
                   V3D::Image<float> const &im0, V3D::Image<float> const &im1, V3D::Image<V3D::Vector2f> const &grad_im0, V3D::Image<V3D::Vector2f> const &grad_im1):
                    AffineRegModel(x1, x2, params, config, im0, im1, grad_im0, grad_im1)
{

}

void HomographyRegModel::fillHomography(VectorXd const &params, V3D::Matrix3x3d &H)
{
    for (int i = 0; i < 8; ++i) H.begin()[i] = params[i];
    H[2][2] = 1.0;
}

void HomographyRegModel::extractParams(V3D::Matrix3x3d const& H, VectorXd &params)
{
    for (int i = 0; i < 8; ++i) params[i] = H.begin()[i];
}

void HomographyRegModel::computeModel()
{
    
}

double HomographyRegModel::error(int i)
{
    float const x = _x1[i][0];
    float const y = _x1[i][1];
    
    V3D::Matrix3x3d H; fillHomography(_params, H);

    V3D::Vector2d q; multiply_A_v_projective(H, V3D::Vector2d(x, y), q);

    float const xx = q[0];
    float const yy = q[1];

    return access_image(_im1, xx, yy) - _im0(x, y);    
}

            
void HomographyRegModel::computeJacobian(int i, Eigen::VectorXd &J)
{
    float const x = _x1[i][0];
    float const y = _x1[i][1];

    V3D::Vector2d const p(x, y);

    V3D::Matrix3x3d H; fillHomography(_params, H);
    V3D::Vector3d Hp;  multiply_A_v_affine(H, p, Hp);
    double const z = 1.0 / Hp[2];

    float const xx = z*Hp[0], yy = z*Hp[1];
    V3D::Vector2f const grad0 = _grad_im0(x, y);
    V3D::Vector2f const grad1 = access_image(_grad_im1, xx, yy);
    V3D::Vector2f grad = 0.5f * (grad0 + grad1);

    V3D::InlineMatrix<double, 3, 9> J1; makeZeroMatrix(J1);
    J1[0][0] = p[0]; J1[0][1] = p[1]; J1[0][2] = 1;
    J1[1][3] = p[0]; J1[1][4] = p[1]; J1[1][5] = 1;
    J1[2][6] = p[0]; J1[2][7] = p[1]; J1[2][8] = 1;

    V3D::InlineMatrix<double, 2, 3> J2;
    J2[0][0] = 1; J2[0][1] = 0; J2[0][2] = -Hp[0]*z;
    J2[1][0] = 0; J2[1][1] = 1; J2[1][2] = -Hp[1]*z;
    scaleMatrixIP(z, J2);

    V3D::InlineMatrix<double, 2, 9> J_H;
    multiply_A_B(J2, J1, J_H);
    V3D::InlineMatrix<double, 2, 8> inner;
    copyMatrixSlice(J_H, 0, 0, 2, 8, inner, 0, 0);

    V3D::InlineVector<double, 8> Jf;
    multiply_At_v(inner, grad, Jf);

    for (int i = 0; i < 8; ++i)  J[i] = Jf[i];

}


void HomographyRegModel::writeHomographyMatrix(std::string file_name)
{
    V3D::Matrix3x3d H;
    fillHomography(_params, H);
    ofstream of(file_name.c_str());
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
            of << H[i][j] << " ";
        of << endl;
    }
    of.close();

}

//---------------------------------------------------------------------------------------------------------------------
FundamentalModel::FundamentalModel(std::vector<Vec3> const &x1, std::vector<Vec3> const &x2, Eigen::VectorXd const &params,  Config const &config):
TwoViewModel(x1, x2, params, config)
{
    
}

void FundamentalModel::computeModel()
{
    _E(0,0) = _params[0]; _E(0,1) = _params[3]; _E(0,2) = _params[6];
    _E(1,0) = _params[1]; _E(1,1) = _params[4]; _E(1,2) = _params[7];
    _E(2,0) = _params[2]; _E(2,1) = _params[5]; _E(2,2) = 1.0;
}

void FundamentalModel::computeJacobian(int i, Eigen::VectorXd &J)
{
    double const a = _params[0], b = _params[1], c = _params[2], d = _params[3], e = _params[4], f = _params[5], g = _params[6], h = _params[7];
    const double u0 = _x1[i][0], u1 = _x1[i][1], u2 = 1.0;
    const double v0 = _x2[i][0], v1 = _x2[i][1], v2 = 1.0;
    
    J[0] = u0*v0*std::pow(std::pow(a*u0 + d*u1 + g*u2, 2) + std::pow(a*v0 + b*v1 + c*v2, 2) + std::pow(b*u0 + e*u1 + h*u2, 2) + std::pow(d*v0 + e*v1 + f*v2, 2), -0.5) + (-1.0*u0*(a*u0 + d*u1 + g*u2) - 1.0*v0*(a*v0 + b*v1 + c*v2))*(u0*(a*v0 + b*v1 + c*v2) + u1*(d*v0 + e*v1 + f*v2) + u2*(g*v0 + h*v1 + 1.0*v2))*std::pow(std::pow(a*u0 + d*u1 + g*u2, 2) + std::pow(a*v0 + b*v1 + c*v2, 2) + std::pow(b*u0 + e*u1 + h*u2, 2) + std::pow(d*v0 + e*v1 + f*v2, 2), -1.5);
    J[1] = u0*v1*std::pow(std::pow(a*u0 + d*u1 + g*u2, 2) + std::pow(a*v0 + b*v1 + c*v2, 2) + std::pow(b*u0 + e*u1 + h*u2, 2) + std::pow(d*v0 + e*v1 + f*v2, 2), -0.5) + (-1.0*u0*(b*u0 + e*u1 + h*u2) - 1.0*v1*(a*v0 + b*v1 + c*v2))*(u0*(a*v0 + b*v1 + c*v2) + u1*(d*v0 + e*v1 + f*v2) + u2*(g*v0 + h*v1 + 1.0*v2))*std::pow(std::pow(a*u0 + d*u1 + g*u2, 2) + std::pow(a*v0 + b*v1 + c*v2, 2) + std::pow(b*u0 + e*u1 + h*u2, 2) + std::pow(d*v0 + e*v1 + f*v2, 2), -1.5);
    J[2] = u0*v2*std::pow(std::pow(a*u0 + d*u1 + g*u2, 2) + std::pow(a*v0 + b*v1 + c*v2, 2) + std::pow(b*u0 + e*u1 + h*u2, 2) + std::pow(d*v0 + e*v1 + f*v2, 2), -0.5) - 1.0*v2*(a*v0 + b*v1 + c*v2)*(u0*(a*v0 + b*v1 + c*v2) + u1*(d*v0 + e*v1 + f*v2) + u2*(g*v0 + h*v1 + 1.0*v2))*std::pow(std::pow(a*u0 + d*u1 + g*u2, 2) + std::pow(a*v0 + b*v1 + c*v2, 2) + std::pow(b*u0 + e*u1 + h*u2, 2) + std::pow(d*v0 + e*v1 + f*v2, 2), -1.5);
    J[3] = u1*v0*std::pow(std::pow(a*u0 + d*u1 + g*u2, 2) + std::pow(a*v0 + b*v1 + c*v2, 2) + std::pow(b*u0 + e*u1 + h*u2, 2) + std::pow(d*v0 + e*v1 + f*v2, 2), -0.5) + (-1.0*u1*(a*u0 + d*u1 + g*u2) - 1.0*v0*(d*v0 + e*v1 + f*v2))*(u0*(a*v0 + b*v1 + c*v2) + u1*(d*v0 + e*v1 + f*v2) + u2*(g*v0 + h*v1 + 1.0*v2))*std::pow(std::pow(a*u0 + d*u1 + g*u2, 2) + std::pow(a*v0 + b*v1 + c*v2, 2) + std::pow(b*u0 + e*u1 + h*u2, 2) + std::pow(d*v0 + e*v1 + f*v2, 2), -1.5);
    J[4] = u1*v1*std::pow(std::pow(a*u0 + d*u1 + g*u2, 2) + std::pow(a*v0 + b*v1 + c*v2, 2) + std::pow(b*u0 + e*u1 + h*u2, 2) + std::pow(d*v0 + e*v1 + f*v2, 2), -0.5) + (-1.0*u1*(b*u0 + e*u1 + h*u2) - 1.0*v1*(d*v0 + e*v1 + f*v2))*(u0*(a*v0 + b*v1 + c*v2) + u1*(d*v0 + e*v1 + f*v2) + u2*(g*v0 + h*v1 + 1.0*v2))*std::pow(std::pow(a*u0 + d*u1 + g*u2, 2) + std::pow(a*v0 + b*v1 + c*v2, 2) + std::pow(b*u0 + e*u1 + h*u2, 2) + std::pow(d*v0 + e*v1 + f*v2, 2), -1.5);
    J[5] = u1*v2*std::pow(std::pow(a*u0 + d*u1 + g*u2, 2) + std::pow(a*v0 + b*v1 + c*v2, 2) + std::pow(b*u0 + e*u1 + h*u2, 2) + std::pow(d*v0 + e*v1 + f*v2, 2), -0.5) - 1.0*v2*(d*v0 + e*v1 + f*v2)*(u0*(a*v0 + b*v1 + c*v2) + u1*(d*v0 + e*v1 + f*v2) + u2*(g*v0 + h*v1 + 1.0*v2))*std::pow(std::pow(a*u0 + d*u1 + g*u2, 2) + std::pow(a*v0 + b*v1 + c*v2, 2) + std::pow(b*u0 + e*u1 + h*u2, 2) + std::pow(d*v0 + e*v1 + f*v2, 2), -1.5);
    J[6] = u2*v0*std::pow(std::pow(a*u0 + d*u1 + g*u2, 2) + std::pow(a*v0 + b*v1 + c*v2, 2) + std::pow(b*u0 + e*u1 + h*u2, 2) + std::pow(d*v0 + e*v1 + f*v2, 2), -0.5) - 1.0*u2*(a*u0 + d*u1 + g*u2)*(u0*(a*v0 + b*v1 + c*v2) + u1*(d*v0 + e*v1 + f*v2) + u2*(g*v0 + h*v1 + 1.0*v2))*std::pow(std::pow(a*u0 + d*u1 + g*u2, 2) + std::pow(a*v0 + b*v1 + c*v2, 2) + std::pow(b*u0 + e*u1 + h*u2, 2) + std::pow(d*v0 + e*v1 + f*v2, 2), -1.5);
    J[7] = u2*v1*std::pow(std::pow(a*u0 + d*u1 + g*u2, 2) + std::pow(a*v0 + b*v1 + c*v2, 2) + std::pow(b*u0 + e*u1 + h*u2, 2) + std::pow(d*v0 + e*v1 + f*v2, 2), -0.5) - 1.0*u2*(b*u0 + e*u1 + h*u2)*(u0*(a*v0 + b*v1 + c*v2) + u1*(d*v0 + e*v1 + f*v2) + u2*(g*v0 + h*v1 + 1.0*v2))*std::pow(std::pow(a*u0 + d*u1 + g*u2, 2) + std::pow(a*v0 + b*v1 + c*v2, 2) + std::pow(b*u0 + e*u1 + h*u2, 2) + std::pow(d*v0 + e*v1 + f*v2, 2), -1.5);

}

//---------------------------------------------------------------------------------------------------------------------
EssentialModel::EssentialModel(std::vector<Vec3> const &x1, std::vector<Vec3> const &x2, Eigen::VectorXd const &params,  Config const &config):
    TwoViewModel(x1, x2, params, config)
{

    _Rx = Eigen::Matrix3d(); _Rx.setZero(); 
    _Rx(0,0) = 1.0; 
    _Ry = Eigen::Matrix3d(); _Ry.setZero(); 
    _Ry(1,1) = 1.0;
    _Rz = Eigen::Matrix3d(); _Rz.setZero();
    _Rz(2,2) = 1.0;

    _Tx = Mat3::Zero();

}

void EssentialModel::computeTx()
{
    double t = _params[3], p = _params[4];
    double tx = sin(t) * cos(p);
    double ty = sin(t) * sin(p);
    double tz = cos(t);
    _Tx << 0, -tz, ty,   tz, 0, -tx,  -ty, tx, 0;
}

void EssentialModel::computeModel()
{
    if (_freeze_model) 
        return;
    // std::cout << "Computing Model\n";
    _Rx(1,1) = cos(_params[0]); _Rx(1,2) = -sin(_params[0]);
    _Rx(2,1) = sin(_params[0]); _Rx(2,2) = cos(_params[0]);

    _Ry(0,0) = cos(_params[1]); _Ry(0,2) = sin(_params[1]);
    _Ry(2,0) = -sin(_params[1]); _Ry(2,2) = cos(_params[1]);

    _Rz(0,0) = cos(_params[2]); _Rz(0,1) = -sin(_params[2]);
    _Rz(1,0) = sin(_params[2]); _Rz(1,1) = cos(_params[2]);
    
    _R = _Rx * _Ry * _Rz;
    computeTx();
    _E = _Tx * _R;
}

void EssentialModel::computeJacobian(int i, Eigen::VectorXd &J)
{
    // std::cout << "Computing Jacobian\n";
    const double u0 = _x1[i][0], u1 = _x1[i][1], u2 = 1.0;
    const double v0 = _x2[i][0], v1 = _x2[i][1], v2 = 1.0;
    const double a = _params[0], b = _params[1], g = _params[2];
    const double t = _params[3], p = _params[4];

    J[0] = (u0*(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*cos(t) + (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(p)*sin(t)) + v1*(-sin(a)*sin(b)*cos(g) - sin(g)*cos(a))*sin(t)*cos(p) + v2*(-sin(a)*sin(g) + sin(b)*cos(a)*cos(g))*sin(t)*cos(p)) + u1*(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*cos(t) + (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(p)*sin(t)) + v1*(sin(a)*sin(b)*sin(g) - cos(a)*cos(g))*sin(t)*cos(p) + v2*(-sin(a)*cos(g) - sin(b)*sin(g)*cos(a))*sin(t)*cos(p)) + u2*(v0*(-sin(a)*sin(p)*sin(t)*cos(b) + cos(a)*cos(b)*cos(t)) + v1*sin(a)*sin(t)*cos(b)*cos(p) - v2*sin(t)*cos(a)*cos(b)*cos(p)))*std::pow(std::pow(u0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + u1*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + u2*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b)), 2) + std::pow(u0*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + u1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + u2*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p)), 2) + std::pow(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g)), 2) + std::pow(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b)), 2), -0.5) + (u0*(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g))) + u1*(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b))) + u2*(v0*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b)) + v1*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p)) + v2*(-sin(a)*sin(t)*cos(b)*cos(p) - sin(b)*sin(p)*sin(t))))*(-0.5*(2*u0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*cos(t) + (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(p)*sin(t)) + 2*u1*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*cos(t) + (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(p)*sin(t)) + 2*u2*(-sin(a)*sin(p)*sin(t)*cos(b) + cos(a)*cos(b)*cos(t)))*(u0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + u1*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + u2*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b))) - 0.5*(u0*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + u1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + u2*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p)))*(2*u0*(-sin(a)*sin(b)*cos(g) - sin(g)*cos(a))*sin(t)*cos(p) + 2*u1*(sin(a)*sin(b)*sin(g) - cos(a)*cos(g))*sin(t)*cos(p) + 2*u2*sin(a)*sin(t)*cos(b)*cos(p)) - 0.5*(2*v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*cos(t) + (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(p)*sin(t)) + 2*v1*(-sin(a)*sin(b)*cos(g) - sin(g)*cos(a))*sin(t)*cos(p) + 2*v2*(-sin(a)*sin(g) + sin(b)*cos(a)*cos(g))*sin(t)*cos(p))*(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g))) - 0.5*(2*v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*cos(t) + (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(p)*sin(t)) + 2*v1*(sin(a)*sin(b)*sin(g) - cos(a)*cos(g))*sin(t)*cos(p) + 2*v2*(-sin(a)*cos(g) - sin(b)*sin(g)*cos(a))*sin(t)*cos(p))*(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b))))*std::pow(std::pow(u0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + u1*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + u2*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b)), 2) + std::pow(u0*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + u1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + u2*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p)), 2) + std::pow(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g)), 2) + std::pow(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b)), 2), -1.5);
    J[1] = (u0*(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g))) + u1*(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b))) + u2*(v0*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b)) + v1*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p)) + v2*(-sin(a)*sin(t)*cos(b)*cos(p) - sin(b)*sin(p)*sin(t))))*(-0.5*(u0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + u1*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + u2*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b)))*(2*u0*(-sin(a)*cos(b)*cos(g)*cos(t) - sin(p)*sin(t)*cos(a)*cos(b)*cos(g)) + 2*u1*(sin(a)*sin(g)*cos(b)*cos(t) + sin(g)*sin(p)*sin(t)*cos(a)*cos(b)) + 2*u2*(-sin(a)*sin(b)*cos(t) - sin(b)*sin(p)*sin(t)*cos(a))) - 0.5*(u0*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + u1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + u2*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p)))*(2*u0*(-sin(b)*cos(g)*cos(t) + sin(t)*cos(a)*cos(b)*cos(g)*cos(p)) + 2*u1*(sin(b)*sin(g)*cos(t) - sin(g)*sin(t)*cos(a)*cos(b)*cos(p)) + 2*u2*(sin(b)*sin(t)*cos(a)*cos(p) + cos(b)*cos(t))) - 0.5*(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g)))*(2*v0*(-sin(a)*cos(b)*cos(g)*cos(t) - sin(p)*sin(t)*cos(a)*cos(b)*cos(g)) + 2*v1*(-sin(b)*cos(g)*cos(t) + sin(t)*cos(a)*cos(b)*cos(g)*cos(p)) + 2*v2*(sin(a)*sin(t)*cos(b)*cos(g)*cos(p) + sin(b)*sin(p)*sin(t)*cos(g))) - 0.5*(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b)))*(2*v0*(sin(a)*sin(g)*cos(b)*cos(t) + sin(g)*sin(p)*sin(t)*cos(a)*cos(b)) + 2*v1*(sin(b)*sin(g)*cos(t) - sin(g)*sin(t)*cos(a)*cos(b)*cos(p)) + 2*v2*(-sin(a)*sin(g)*sin(t)*cos(b)*cos(p) - sin(b)*sin(g)*sin(p)*sin(t))))*std::pow(std::pow(u0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + u1*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + u2*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b)), 2) + std::pow(u0*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + u1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + u2*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p)), 2) + std::pow(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g)), 2) + std::pow(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b)), 2), -1.5) + (u0*(v0*(-sin(a)*cos(b)*cos(g)*cos(t) - sin(p)*sin(t)*cos(a)*cos(b)*cos(g)) + v1*(-sin(b)*cos(g)*cos(t) + sin(t)*cos(a)*cos(b)*cos(g)*cos(p)) + v2*(sin(a)*sin(t)*cos(b)*cos(g)*cos(p) + sin(b)*sin(p)*sin(t)*cos(g))) + u1*(v0*(sin(a)*sin(g)*cos(b)*cos(t) + sin(g)*sin(p)*sin(t)*cos(a)*cos(b)) + v1*(sin(b)*sin(g)*cos(t) - sin(g)*sin(t)*cos(a)*cos(b)*cos(p)) + v2*(-sin(a)*sin(g)*sin(t)*cos(b)*cos(p) - sin(b)*sin(g)*sin(p)*sin(t))) + u2*(v0*(-sin(a)*sin(b)*cos(t) - sin(b)*sin(p)*sin(t)*cos(a)) + v1*(sin(b)*sin(t)*cos(a)*cos(p) + cos(b)*cos(t)) + v2*(sin(a)*sin(b)*sin(t)*cos(p) - sin(p)*sin(t)*cos(b))))*std::pow(std::pow(u0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + u1*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + u2*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b)), 2) + std::pow(u0*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + u1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + u2*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p)), 2) + std::pow(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g)), 2) + std::pow(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b)), 2), -0.5);
    J[2] = (u0*(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) + (sin(a)*sin(b)*sin(g) - cos(a)*cos(g))*cos(t)) + v1*((-sin(a)*cos(g) - sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b))) + u1*(v0*((-sin(a)*sin(g) + sin(b)*cos(a)*cos(g))*sin(p)*sin(t) + (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) - cos(b)*cos(g)*cos(t)) + v2*((-sin(a)*sin(b)*cos(g) - sin(g)*cos(a))*sin(t)*cos(p) + sin(p)*sin(t)*cos(b)*cos(g))))*std::pow(std::pow(u0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + u1*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + u2*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b)), 2) + std::pow(u0*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + u1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + u2*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p)), 2) + std::pow(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g)), 2) + std::pow(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b)), 2), -0.5) + (u0*(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g))) + u1*(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b))) + u2*(v0*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b)) + v1*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p)) + v2*(-sin(a)*sin(t)*cos(b)*cos(p) - sin(b)*sin(p)*sin(t))))*(-0.5*(2*u0*((-sin(a)*cos(g) - sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + 2*u1*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) - cos(b)*cos(g)*cos(t)))*(u0*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + u1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + u2*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p))) - 0.5*(2*u0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) + (sin(a)*sin(b)*sin(g) - cos(a)*cos(g))*cos(t)) + 2*u1*((-sin(a)*sin(g) + sin(b)*cos(a)*cos(g))*sin(p)*sin(t) + (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)))*(u0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + u1*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + u2*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b))) - 0.5*(2*v0*((-sin(a)*sin(g) + sin(b)*cos(a)*cos(g))*sin(p)*sin(t) + (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + 2*v1*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) - cos(b)*cos(g)*cos(t)) + 2*v2*((-sin(a)*sin(b)*cos(g) - sin(g)*cos(a))*sin(t)*cos(p) + sin(p)*sin(t)*cos(b)*cos(g)))*(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b))) - 0.5*(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g)))*(2*v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) + (sin(a)*sin(b)*sin(g) - cos(a)*cos(g))*cos(t)) + 2*v1*((-sin(a)*cos(g) - sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + 2*v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b))))*std::pow(std::pow(u0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + u1*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + u2*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b)), 2) + std::pow(u0*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + u1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + u2*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p)), 2) + std::pow(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g)), 2) + std::pow(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b)), 2), -1.5);
    J[3] = (u0*(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g))) + u1*(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b))) + u2*(v0*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b)) + v1*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p)) + v2*(-sin(a)*sin(t)*cos(b)*cos(p) - sin(b)*sin(p)*sin(t))))*(-0.5*(2*u0*((-sin(a)*sin(g) + sin(b)*cos(a)*cos(g))*cos(p)*cos(t) - sin(t)*cos(b)*cos(g)) + 2*u1*((-sin(a)*cos(g) - sin(b)*sin(g)*cos(a))*cos(p)*cos(t) + sin(g)*sin(t)*cos(b)) + 2*u2*(-sin(b)*sin(t) - cos(a)*cos(b)*cos(p)*cos(t)))*(u0*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + u1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + u2*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p))) - 0.5*(u0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + u1*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + u2*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b)))*(2*u0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*cos(t) - (-sin(a)*sin(b)*cos(g) - sin(g)*cos(a))*sin(t)) + 2*u1*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*cos(t) - (sin(a)*sin(b)*sin(g) - cos(a)*cos(g))*sin(t)) + 2*u2*(-sin(a)*sin(t)*cos(b) + sin(p)*cos(a)*cos(b)*cos(t))) - 0.5*(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g)))*(2*v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*cos(t) - (-sin(a)*sin(b)*cos(g) - sin(g)*cos(a))*sin(t)) + 2*v1*((-sin(a)*sin(g) + sin(b)*cos(a)*cos(g))*cos(p)*cos(t) - sin(t)*cos(b)*cos(g)) + 2*v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(p)*cos(t) - sin(p)*cos(b)*cos(g)*cos(t))) - 0.5*(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b)))*(2*v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*cos(t) - (sin(a)*sin(b)*sin(g) - cos(a)*cos(g))*sin(t)) + 2*v1*((-sin(a)*cos(g) - sin(b)*sin(g)*cos(a))*cos(p)*cos(t) + sin(g)*sin(t)*cos(b)) + 2*v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(p)*cos(t) + sin(g)*sin(p)*cos(b)*cos(t))))*std::pow(std::pow(u0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + u1*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + u2*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b)), 2) + std::pow(u0*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + u1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + u2*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p)), 2) + std::pow(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g)), 2) + std::pow(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b)), 2), -1.5) + (u0*(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*cos(t) - (-sin(a)*sin(b)*cos(g) - sin(g)*cos(a))*sin(t)) + v1*((-sin(a)*sin(g) + sin(b)*cos(a)*cos(g))*cos(p)*cos(t) - sin(t)*cos(b)*cos(g)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(p)*cos(t) - sin(p)*cos(b)*cos(g)*cos(t))) + u1*(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*cos(t) - (sin(a)*sin(b)*sin(g) - cos(a)*cos(g))*sin(t)) + v1*((-sin(a)*cos(g) - sin(b)*sin(g)*cos(a))*cos(p)*cos(t) + sin(g)*sin(t)*cos(b)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(p)*cos(t) + sin(g)*sin(p)*cos(b)*cos(t))) + u2*(v0*(-sin(a)*sin(t)*cos(b) + sin(p)*cos(a)*cos(b)*cos(t)) + v1*(-sin(b)*sin(t) - cos(a)*cos(b)*cos(p)*cos(t)) + v2*(-sin(a)*cos(b)*cos(p)*cos(t) - sin(b)*sin(p)*cos(t))))*std::pow(std::pow(u0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + u1*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + u2*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b)), 2) + std::pow(u0*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + u1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + u2*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p)), 2) + std::pow(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g)), 2) + std::pow(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b)), 2), -0.5);
    J[4] = (u0*(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g))) + u1*(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b))) + u2*(v0*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b)) + v1*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p)) + v2*(-sin(a)*sin(t)*cos(b)*cos(p) - sin(b)*sin(p)*sin(t))))*(-0.5*(u0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + u1*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + u2*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b)))*(2*u0*(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + 2*u1*(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) + 2*u2*sin(t)*cos(a)*cos(b)*cos(p)) - 0.5*(u0*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + u1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + u2*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p)))*(-2*u0*(-sin(a)*sin(g) + sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - 2*u1*(-sin(a)*cos(g) - sin(b)*sin(g)*cos(a))*sin(p)*sin(t) + 2*u2*sin(p)*sin(t)*cos(a)*cos(b)) - 0.5*(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g)))*(2*v0*(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) - 2*v1*(-sin(a)*sin(g) + sin(b)*cos(a)*cos(g))*sin(p)*sin(t) + 2*v2*(-(sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(p)*sin(t) - sin(t)*cos(b)*cos(g)*cos(p))) - 0.5*(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b)))*(2*v0*(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - 2*v1*(-sin(a)*cos(g) - sin(b)*sin(g)*cos(a))*sin(p)*sin(t) + 2*v2*(-(-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(p)*sin(t) + sin(g)*sin(t)*cos(b)*cos(p))))*std::pow(std::pow(u0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + u1*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + u2*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b)), 2) + std::pow(u0*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + u1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + u2*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p)), 2) + std::pow(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g)), 2) + std::pow(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b)), 2), -1.5) + (u0*(v0*(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) - v1*(-sin(a)*sin(g) + sin(b)*cos(a)*cos(g))*sin(p)*sin(t) + v2*(-(sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(p)*sin(t) - sin(t)*cos(b)*cos(g)*cos(p))) + u1*(v0*(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - v1*(-sin(a)*cos(g) - sin(b)*sin(g)*cos(a))*sin(p)*sin(t) + v2*(-(-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(p)*sin(t) + sin(g)*sin(t)*cos(b)*cos(p))) + u2*(v0*sin(t)*cos(a)*cos(b)*cos(p) + v1*sin(p)*sin(t)*cos(a)*cos(b) + v2*(sin(a)*sin(p)*sin(t)*cos(b) - sin(b)*sin(t)*cos(p))))*std::pow(std::pow(u0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + u1*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + u2*(sin(a)*cos(b)*cos(t) + sin(p)*sin(t)*cos(a)*cos(b)), 2) + std::pow(u0*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + u1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + u2*(sin(b)*cos(t) - sin(t)*cos(a)*cos(b)*cos(p)), 2) + std::pow(v0*((sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(p)*sin(t) - (sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*cos(t)) + v1*(-(sin(a)*sin(g) - sin(b)*cos(a)*cos(g))*sin(t)*cos(p) + cos(b)*cos(g)*cos(t)) + v2*((sin(a)*sin(b)*cos(g) + sin(g)*cos(a))*sin(t)*cos(p) - sin(p)*sin(t)*cos(b)*cos(g)), 2) + std::pow(v0*((sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(p)*sin(t) - (-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*cos(t)) + v1*(-(sin(a)*cos(g) + sin(b)*sin(g)*cos(a))*sin(t)*cos(p) - sin(g)*cos(b)*cos(t)) + v2*((-sin(a)*sin(b)*sin(g) + cos(a)*cos(g))*sin(t)*cos(p) + sin(g)*sin(p)*sin(t)*cos(b)), 2), -0.5);
}

