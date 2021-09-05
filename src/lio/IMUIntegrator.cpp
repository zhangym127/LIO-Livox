#include "IMUIntegrator/IMUIntegrator.h"

IMUIntegrator::IMUIntegrator(){
  Reset();
  noise.setZero();
  noise.block<3, 3>(0, 0) =  Eigen::Matrix3d::Identity() * gyr_n * gyr_n;
  noise.block<3, 3>(3, 3) =  Eigen::Matrix3d::Identity() * acc_n * acc_n;
  noise.block<3, 3>(6, 6) =  Eigen::Matrix3d::Identity() * gyr_w * gyr_w;
  noise.block<3, 3>(9, 9) =  Eigen::Matrix3d::Identity() * acc_w * acc_w;
}

/** \brief constructor of IMUIntegrator
 * \param[in] vIMU: IMU messages need to be integrated
 */
IMUIntegrator::IMUIntegrator(std::vector<sensor_msgs::ImuConstPtr> vIMU):
vimuMsg(std::move(vIMU)){
  Reset();
  noise.setZero();
  noise.block<3, 3>(0, 0) =  Eigen::Matrix3d::Identity() * gyr_n * gyr_n;
  noise.block<3, 3>(3, 3) =  Eigen::Matrix3d::Identity() * acc_n * acc_n;
  noise.block<3, 3>(6, 6) =  Eigen::Matrix3d::Identity() * gyr_w * gyr_w;
  noise.block<3, 3>(9, 9) =  Eigen::Matrix3d::Identity() * acc_w * acc_w;
}

void IMUIntegrator::Reset(){
  dq.setIdentity();
  dp.setZero();
  dv.setZero();
  dtime = 0;
  covariance.setZero();
  jacobian.setIdentity();
  linearized_bg.setZero();
  linearized_ba.setZero();
}

const Eigen::Quaterniond & IMUIntegrator::GetDeltaQ() const {return dq;}

const Eigen::Vector3d & IMUIntegrator::GetDeltaP() const {return dp;}

const Eigen::Vector3d & IMUIntegrator::GetDeltaV() const {return dv;}

const double & IMUIntegrator::GetDeltaTime() const {return dtime;}

const Eigen::Vector3d & IMUIntegrator::GetBiasGyr() const {return linearized_bg;}

const Eigen::Vector3d& IMUIntegrator::GetBiasAcc() const {return linearized_ba;}

const Eigen::Matrix<double, 15, 15>& IMUIntegrator::GetCovariance(){return covariance;}

const Eigen::Matrix<double, 15, 15> & IMUIntegrator::GetJacobian() const {return jacobian;}

void IMUIntegrator::PushIMUMsg(const sensor_msgs::ImuConstPtr& imu){
  vimuMsg.push_back(imu);
}
void IMUIntegrator::PushIMUMsg(const std::vector<sensor_msgs::ImuConstPtr>& vimu){
  vimuMsg.insert(vimuMsg.end(), vimu.begin(), vimu.end());
}
const std::vector<sensor_msgs::ImuConstPtr> & IMUIntegrator::GetIMUMsg() const {return vimuMsg;}

void IMUIntegrator::GyroIntegration(double lastTime){
  double current_time = lastTime;
  for(auto & imu : vimuMsg){
    Eigen::Vector3d gyr;
    gyr << imu->angular_velocity.x,
            imu->angular_velocity.y,
            imu->angular_velocity.z;
    double dt = imu->header.stamp.toSec() - current_time;
    ROS_ASSERT(dt >= 0);
    Eigen::Matrix3d dR = Sophus::SO3d::exp(gyr*dt).matrix();
    Eigen::Quaterniond qr(dq*dR);
    if (qr.w()<0)
      qr.coeffs() *= -1;
    dq = qr.normalized();
    current_time = imu->header.stamp.toSec();
  }
}

/** @brief 对IMU进行预积分，并求得预积分测量噪声的协方差矩阵
  * @param [in] lastTime 上一帧点云的时间戳
  * @param [in] bg 上一帧点云对应的角速度偏差
  * @param [in] ba 上一帧点云对应的加速度偏差
  */
void IMUIntegrator::PreIntegration(double lastTime, const Eigen::Vector3d& bg, const Eigen::Vector3d& ba){
  Reset();
  linearized_bg = bg;
  linearized_ba = ba;
  /* 获得第一帧IMU的起始时间戳 */
  double current_time = lastTime;
  /* 对每一帧IMU数据 */
  for(auto & imu : vimuMsg){
    /* 加载角速度到gyr */
    Eigen::Vector3d gyr;
    gyr <<  imu->angular_velocity.x,
            imu->angular_velocity.y,
            imu->angular_velocity.z;
    /* 加载加速度到acc */
    Eigen::Vector3d acc;
    acc << imu->linear_acceleration.x * gnorm,
            imu->linear_acceleration.y * gnorm,
            imu->linear_acceleration.z * gnorm;
    /* 获得当前帧对应的dt */
    double dt = imu->header.stamp.toSec() - current_time;
    if(dt <= 0 )
      ROS_WARN("dt <= 0");
    /* gyr和acc分别减去偏差 */
    gyr -= bg;
    acc -= ba;
    
    double dt2 = dt*dt;
    /* 求得当前帧的姿态增量 */
    Eigen::Vector3d gyr_dt = gyr*dt;
    Eigen::Matrix3d dR = Sophus::SO3d::exp(gyr_dt).matrix();
    /* 求右乘BCH近似雅可比，Jr中的r表示右乘 */
    Eigen::Matrix3d Jr = Eigen::Matrix3d::Identity();
    double gyr_dt_norm = gyr_dt.norm();
    if(gyr_dt_norm > 0.00001){
      Eigen::Vector3d k = gyr_dt.normalized();
      Eigen::Matrix3d K = Sophus::SO3d::hat(k);
      Jr =   Eigen::Matrix3d::Identity()
             - (1-cos(gyr_dt_norm))/gyr_dt_norm*K
             + (1-sin(gyr_dt_norm)/gyr_dt_norm)*K*K;
    }
    /* 求预积分测量噪声迭代方程中的参数矩阵A和B */
    Eigen::Matrix<double,15,15> A = Eigen::Matrix<double,15,15>::Identity();
    A.block<3,3>(0,3) = -0.5*dq.matrix()*Sophus::SO3d::hat(acc)*dt2;
    A.block<3,3>(0,6) = Eigen::Matrix3d::Identity()*dt;
    A.block<3,3>(0,12) = -0.5*dq.matrix()*dt2;
    A.block<3,3>(3,3) = dR.transpose();
    A.block<3,3>(3,9) = - Jr*dt;
    A.block<3,3>(6,3) = -dq.matrix()*Sophus::SO3d::hat(acc)*dt;
    A.block<3,3>(6,12) = -dq.matrix()*dt;
    Eigen::Matrix<double,15,12> B = Eigen::Matrix<double,15,12>::Zero();
    B.block<3,3>(0,3) = 0.5*dq.matrix()*dt2;
    B.block<3,3>(3,0) = Jr*dt;
    B.block<3,3>(6,3) = dq.matrix()*dt;
    B.block<3,3>(9,6) = Eigen::Matrix3d::Identity()*dt;
    B.block<3,3>(12,9) = Eigen::Matrix3d::Identity()*dt;
    jacobian = A * jacobian;
    /* 求得预积分测量噪声的协方差矩阵 */
    covariance = A * covariance * A.transpose() + B * noise * B.transpose();

    /* 下面是对dp、dv、dq的积分，与A、B无关，很容易理解 */
    /* 更新位置增量dp：经典牛顿公式S=vt+0.5at^2 */
    dp += dv*dt + 0.5*dq.matrix()*acc*dt2;
    /* 更新速度增量dv：加速度直接乘以dt再叠加旋转 */
    dv += dq.matrix()*acc*dt;
    /* 更新姿态增量dq：dq直接乘以当前帧的增量dR */
    Eigen::Matrix3d m3dR = dq.matrix()*dR;
    Eigen::Quaterniond qtmp(m3dR); //转成四元数
    if (qtmp.w()<0)
      qtmp.coeffs() *= -1;
    dq = qtmp.normalized(); //归一化
    dtime += dt;
    /* 更新时间戳指针 */
    current_time = imu->header.stamp.toSec();
  }
}

/** @brief 求加速度测量均值
  * @return 返回当前帧的重力加速度均值
  */
Eigen::Vector3d IMUIntegrator::GetAverageAcc() {
  int i = 0;
  Eigen::Vector3d sum_acc(0, 0, 0);
  for(auto & imu : vimuMsg){
    Eigen::Vector3d acc;
    acc << imu->linear_acceleration.x * gnorm,
           imu->linear_acceleration.y * gnorm,
           imu->linear_acceleration.z * gnorm;
    sum_acc += acc;
    i++;
    if(i > 30) break;
  }
  return sum_acc / i;
}

