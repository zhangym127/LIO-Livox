#include "Estimator/Estimator.h"
typedef pcl::PointXYZINormal PointType;

int WINDOWSIZE;
bool LidarIMUInited = false;
boost::shared_ptr<std::list<Estimator::LidarFrame>> lidarFrameList;
pcl::PointCloud<PointType>::Ptr laserCloudFullRes;
Estimator* estimator;

ros::Publisher pubLaserOdometry;
ros::Publisher pubLaserOdometryPath;
ros::Publisher pubFullLaserCloud;
tf::StampedTransform laserOdometryTrans;
tf::TransformBroadcaster* tfBroadcaster;
ros::Publisher pubGps;

bool newfullCloud = false;

Eigen::Matrix4d transformAftMapped = Eigen::Matrix4d::Identity();

std::mutex _mutexLidarQueue;
std::queue<sensor_msgs::PointCloud2ConstPtr> _lidarMsgQueue;
std::mutex _mutexIMUQueue;
std::queue<sensor_msgs::ImuConstPtr> _imuMsgQueue;
Eigen::Matrix4d exTlb;
Eigen::Matrix3d exRlb, exRbl;
Eigen::Vector3d exPlb, exPbl;
Eigen::Vector3d GravityVector;
float filter_parameter_corner = 0.2;
float filter_parameter_surf = 0.4;
int IMU_Mode = 2;
sensor_msgs::NavSatFix gps;
int pushCount = 0;
double startTime = 0;

nav_msgs::Path laserOdoPath;

/** \brief publish odometry infomation
  * \param[in] newPose: pose to be published
  * \param[in] timefullCloud: time stamp
  */
void pubOdometry(const Eigen::Matrix4d& newPose, double& timefullCloud){
  nav_msgs::Odometry laserOdometry;

  Eigen::Matrix3d Rcurr = newPose.topLeftCorner(3, 3);
  Eigen::Quaterniond newQuat(Rcurr);
  Eigen::Vector3d newPosition = newPose.topRightCorner(3, 1);
  laserOdometry.header.frame_id = "/world";
  laserOdometry.child_frame_id = "/livox_frame";
  laserOdometry.header.stamp = ros::Time().fromSec(timefullCloud);
  laserOdometry.pose.pose.orientation.x = newQuat.x();
  laserOdometry.pose.pose.orientation.y = newQuat.y();
  laserOdometry.pose.pose.orientation.z = newQuat.z();
  laserOdometry.pose.pose.orientation.w = newQuat.w();
  laserOdometry.pose.pose.position.x = newPosition.x();
  laserOdometry.pose.pose.position.y = newPosition.y();
  laserOdometry.pose.pose.position.z = newPosition.z();
  pubLaserOdometry.publish(laserOdometry);

  geometry_msgs::PoseStamped laserPose;
  laserPose.header = laserOdometry.header;
  laserPose.pose = laserOdometry.pose.pose;
  laserOdoPath.header.stamp = laserOdometry.header.stamp;
  laserOdoPath.poses.push_back(laserPose);
  laserOdoPath.header.frame_id = "/world";
  pubLaserOdometryPath.publish(laserOdoPath);

  laserOdometryTrans.frame_id_ = "/world";
  laserOdometryTrans.child_frame_id_ = "/livox_frame";
  laserOdometryTrans.stamp_ = ros::Time().fromSec(timefullCloud);
  laserOdometryTrans.setRotation(tf::Quaternion(newQuat.x(), newQuat.y(), newQuat.z(), newQuat.w()));
  laserOdometryTrans.setOrigin(tf::Vector3(newPosition.x(), newPosition.y(), newPosition.z()));
  tfBroadcaster->sendTransform(laserOdometryTrans);

	gps.header.stamp = ros::Time().fromSec(timefullCloud);
	gps.header.frame_id = "world";
	gps.latitude = newPosition.x();
	gps.longitude = newPosition.y();
	gps.altitude = newPosition.z();
	gps.position_covariance = {
					Rcurr(0, 0), Rcurr(1, 0), Rcurr(2, 0),
					Rcurr(0, 1), Rcurr(1, 1), Rcurr(2, 1),
					Rcurr(0, 2), Rcurr(1, 2), Rcurr(2, 2)
	};
	pubGps.publish(gps);

}

/** @brief 订阅点云的回调函数，收到点云后添加到_lidarMsgQueue中
  * @param msg 回调函数的固定参数，含有完整点云
  */
void fullCallBack(const sensor_msgs::PointCloud2ConstPtr &msg){
    /* 入队列操作加锁，在lock变量的生命周期内，一直保持上锁状态 */
  // push lidar msg to queue
	std::unique_lock<std::mutex> lock(_mutexLidarQueue);
  _lidarMsgQueue.push(msg);
}

/** @brief 订阅IMU的回调函数，收到IMU数据后添加到_imuMsgQueue中
  * @param msg 回调函数的固定参数，含有IMU数据
  */
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg){
  // push IMU msg to queue
  std::unique_lock<std::mutex> lock(_mutexIMUQueue);
  _imuMsgQueue.push(imu_msg);
}

/** \brief get IMU messages in a certain time interval
  * \param[in] startTime: left boundary of time interval
  * \param[in] endTime: right boundary of time interval
  * \param[in] vimuMsg: store IMU messages
  */
bool fetchImuMsgs(double startTime, double endTime, std::vector<sensor_msgs::ImuConstPtr> &vimuMsg){
  std::unique_lock<std::mutex> lock(_mutexIMUQueue);
  double current_time = 0;
  vimuMsg.clear();
  while(true)
  {
    if(_imuMsgQueue.empty())
      break;
    if(_imuMsgQueue.back()->header.stamp.toSec()<endTime ||
       _imuMsgQueue.front()->header.stamp.toSec()>=endTime)
      break;
    sensor_msgs::ImuConstPtr& tmpimumsg = _imuMsgQueue.front();
    double time = tmpimumsg->header.stamp.toSec();
    if(time<=endTime && time>startTime){
      vimuMsg.push_back(tmpimumsg);
      current_time = time;
      _imuMsgQueue.pop();
      if(time == endTime) break;
    } else{
      if(time<=startTime){
        _imuMsgQueue.pop();
      } else{
        double dt_1 = endTime - current_time;
        double dt_2 = time - endTime;
        ROS_ASSERT(dt_1 >= 0);
        ROS_ASSERT(dt_2 >= 0);
        ROS_ASSERT(dt_1 + dt_2 > 0);
        double w1 = dt_2 / (dt_1 + dt_2);
        double w2 = dt_1 / (dt_1 + dt_2);
        sensor_msgs::ImuPtr theLastIMU(new sensor_msgs::Imu);
        theLastIMU->linear_acceleration.x = w1 * vimuMsg.back()->linear_acceleration.x + w2 * tmpimumsg->linear_acceleration.x;
        theLastIMU->linear_acceleration.y = w1 * vimuMsg.back()->linear_acceleration.y + w2 * tmpimumsg->linear_acceleration.y;
        theLastIMU->linear_acceleration.z = w1 * vimuMsg.back()->linear_acceleration.z + w2 * tmpimumsg->linear_acceleration.z;
        theLastIMU->angular_velocity.x = w1 * vimuMsg.back()->angular_velocity.x + w2 * tmpimumsg->angular_velocity.x;
        theLastIMU->angular_velocity.y = w1 * vimuMsg.back()->angular_velocity.y + w2 * tmpimumsg->angular_velocity.y;
        theLastIMU->angular_velocity.z = w1 * vimuMsg.back()->angular_velocity.z + w2 * tmpimumsg->angular_velocity.z;
        theLastIMU->header.stamp.fromSec(endTime);
        vimuMsg.emplace_back(theLastIMU);
        break;
      }
    }
  }
  return !vimuMsg.empty();
}

/** \brief Remove Lidar Distortion 点云帧内校正，采用时间偏移插值的方式实现
  * \param[in] cloud: lidar cloud need to be undistorted 待校正的点云帧
  * \param[in] dRlc: delta rotation 当前点云帧对应的旋转增量
  * \param[in] dtlc: delta displacement 当前点云帧对应的平移增量
  */
void RemoveLidarDistortion(pcl::PointCloud<PointType>::Ptr& cloud,
                           const Eigen::Matrix3d& dRlc, const Eigen::Vector3d& dtlc){
  int PointsNum = cloud->points.size();
  for (int i = 0; i < PointsNum; i++) {
    Eigen::Vector3d startP;
    /* 获得当前点时间偏移相对于点云帧时长的比值 */
    float s = cloud->points[i].normal_x;
    /* 利用时间比值对旋转增量进行插值 */
    Eigen::Quaterniond qlc = Eigen::Quaterniond(dRlc).normalized();
    Eigen::Quaterniond delta_qlc = Eigen::Quaterniond::Identity().slerp(s, qlc).normalized();
    /* 利用时间比值对平移增量进行插值 */
    const Eigen::Vector3d delta_Plc = s * dtlc;
    /* 将当前点的位姿变换到起始点时刻 */
    /* 点云中目标的运动方向和车辆的运动方向是反的，因此当前点的位姿【乘加】位姿增量才是变换到起始位置 */
    startP = delta_qlc * Eigen::Vector3d(cloud->points[i].x,cloud->points[i].y,cloud->points[i].z) + delta_Plc;
    /* 将当前点的位姿变换到结束点时刻 */
    /* 同样的道理，起始点的位姿【减去】平移增量并乘以旋转增量的【转置】才是变换到结束点时刻 */
    Eigen::Vector3d _po = dRlc.transpose() * (startP - dtlc);

    cloud->points[i].x = _po(0);
    cloud->points[i].y = _po(1);
    cloud->points[i].z = _po(2);
    cloud->points[i].normal_x = 1.0; //结束点的时间
  }
}

/** @brief 进行位姿的初始化，获得重力加速度的方向，IMU的偏差等
  *   1.在初始化之前，应该有三帧扫描点云完成了到Map的位姿优化
  *   2.计算重力加速度的初始方向
  *   3.以三帧点云的优化位姿和IMU的预积分位姿为残差，求得重力加速度的方向，以及IMU的偏差
  *   4.基于优化后的IMU偏差，对三帧扫描点云之间的IMU数据重新进行预积分
  */
bool TryMAPInitialization() {

  /* 获得起始帧对应的加速度均值 */
  Eigen::Vector3d average_acc = -lidarFrameList->begin()->imuIntegrator.GetAverageAcc();
  double info_g = std::fabs(9.805 - average_acc.norm());
  /* 加速度均值归一化，再乘以重力加速度 */
  average_acc = average_acc * 9.805 / average_acc.norm();

  /* 通过Ceres优化求初始重力加速度方向 */
  // calculate the initial gravity direction
  
  /* 定义状态变量——重力加速度方向 */
  double para_quat[4];
  para_quat[0] = 1;
  para_quat[1] = 0;
  para_quat[2] = 0;
  para_quat[3] = 0;

  ceres::LocalParameterization *quatParam = new ceres::QuaternionParameterization();
  ceres::Problem problem_quat;
  
  /* 添加状态变量到优化问题 */
  problem_quat.AddParameterBlock(para_quat, 4, quatParam);

  /* 添加代价函数到优化问题，残差就是用(para_quat*重力加速度-加速度均值) */
  problem_quat.AddResidualBlock(Cost_Initial_G::Create(average_acc),
                                nullptr,
                                para_quat);
  /* 优化 */
  ceres::Solver::Options options_quat;
  ceres::Solver::Summary summary_quat;
  ceres::Solve(options_quat, &problem_quat, &summary_quat);

  /* 取得重力加速度方向 */
  Eigen::Quaterniond q_wg(para_quat[0], para_quat[1], para_quat[2], para_quat[3]);

  /* 定义状态变量的初始值，偏差为0 */
  //build prior factor of LIO initialization
  Eigen::Vector3d prior_r = Eigen::Vector3d::Zero();
  Eigen::Vector3d prior_ba = Eigen::Vector3d::Zero();
  Eigen::Vector3d prior_bg = Eigen::Vector3d::Zero();
  std::vector<Eigen::Vector3d> prior_v;
  int v_size = lidarFrameList->size(); //应该有三帧点云
  for(int i = 0; i < v_size; i++) {
    prior_v.push_back(Eigen::Vector3d::Zero());
  }
  /* 重力加速度方向转成旋转矩阵，再转成李代数，获得起始帧对应的初始位姿prior_r */
  Sophus::SO3d SO3_R_wg(q_wg.toRotationMatrix());
  prior_r = SO3_R_wg.log();
  
  /* 求每帧对应的初始速度prior_v */
  for (int i = 1; i < v_size; i++){
    auto iter = lidarFrameList->begin();
    auto iter_next = lidarFrameList->begin();
    std::advance(iter, i-1);
    std::advance(iter_next, i);

    Eigen::Vector3d velo_imu = (iter_next->P - iter->P + iter_next->Q*exPlb - iter->Q*exPlb) / (iter_next->timeStamp - iter->timeStamp);
    prior_v[i] = velo_imu;
  }
  prior_v[0] = prior_v[1];

  /* 定义待优化的状态变量，包括速度、姿态、偏差 */
  double para_v[v_size][3];
  double para_r[3];
  double para_ba[3];
  double para_bg[3];

  for(int i = 0; i < 3; i++) {
    para_r[i] = 0;
    para_ba[i] = 0;
    para_bg[i] = 0;
  }

  for(int i = 0; i < v_size; i++) {
    for(int j = 0; j < 3; j++) {
      para_v[i][j] = prior_v[i][j];
    }
  }

  Eigen::Matrix<double, 3, 3> sqrt_information_r = 2000.0 * Eigen::Matrix<double, 3, 3>::Identity();
  Eigen::Matrix<double, 3, 3> sqrt_information_ba = 1000.0 * Eigen::Matrix<double, 3, 3>::Identity();
  Eigen::Matrix<double, 3, 3> sqrt_information_bg = 4000.0 * Eigen::Matrix<double, 3, 3>::Identity();
  Eigen::Matrix<double, 3, 3> sqrt_information_v = 4000.0 * Eigen::Matrix<double, 3, 3>::Identity();

  /* 将待优化的状态变量添加到优化问题 */
  ceres::Problem::Options problem_options;
  ceres::Problem problem(problem_options);
  problem.AddParameterBlock(para_r, 3);
  problem.AddParameterBlock(para_ba, 3);
  problem.AddParameterBlock(para_bg, 3);
  for(int i = 0; i < v_size; i++) {
    problem.AddParameterBlock(para_v[i], 3);
  }
  
  /* 添加代价函数 */
  // add CostFunction
  
  /* 姿态残差就是prior_r与para_r之间的姿态增量 */
  problem.AddResidualBlock(Cost_Initialization_Prior_R::Create(prior_r, sqrt_information_r),
                           nullptr,
                           para_r);
  /* FIXME:下面这两个代价函数中prior_ba和prior_bg的值都是0，进行优化不就变成了使para_ba和para_bg趋近于0？这样的优化有什么意思？*/
  /* IMU偏差残差就是prior_ba与para_ba之差*/
  problem.AddResidualBlock(Cost_Initialization_Prior_bv::Create(prior_ba, sqrt_information_ba),
                           nullptr,
                           para_ba);
  /* IMU偏差残差就是prior_bg与para_bg之差*/
  problem.AddResidualBlock(Cost_Initialization_Prior_bv::Create(prior_bg, sqrt_information_bg),
                           nullptr,
                           para_bg);
  /* 速度残差就是prior_v与para_v之差*/
  for(int i = 0; i < v_size; i++) {
    problem.AddResidualBlock(Cost_Initialization_Prior_bv::Create(prior_v[i], sqrt_information_v),
                             nullptr,
                             para_v[i]);
  }

  for(int i = 1; i < v_size; i++) {
    auto iter = lidarFrameList->begin();
    auto iter_next = lidarFrameList->begin();
    std::advance(iter, i-1);
    std::advance(iter_next, i);

    /* 从lidarFrame.Q、lidarFrame.P中取得位置和姿态，这个位姿来自扫描点云到Map的匹配优化 */
    Eigen::Vector3d pi = iter->P + iter->Q*exPlb;
    Sophus::SO3d SO3_Ri(iter->Q*exRlb);
    Eigen::Vector3d ri = SO3_Ri.log();
    Eigen::Vector3d pj = iter_next->P + iter_next->Q*exPlb;
    Sophus::SO3d SO3_Rj(iter_next->Q*exRlb);
    Eigen::Vector3d rj = SO3_Rj.log();

    /* 添加IMU预积分代价函数到优化问题 */
    /* 与完成初始化之后的IMU预积分代价函数构造方式基本上一致 */
    /* 残差就是点云匹配的位姿和IMU预积分的位姿之差 */
    problem.AddResidualBlock(Cost_Initialization_IMU::Create(iter_next->imuIntegrator,
                                                                   ri,
                                                                   rj,
                                                                   pj-pi,
                                                                   Eigen::LLT<Eigen::Matrix<double, 9, 9>>
                                                                    (iter_next->imuIntegrator.GetCovariance().block<9,9>(0,0).inverse())
                                                                    .matrixL().transpose()),
                             nullptr,
                             para_r,
                             para_v[i-1],
                             para_v[i],
                             para_ba,
                             para_bg);
  }

  /* 开始优化 */
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = false;
  options.linear_solver_type = ceres::DENSE_QR;
  options.num_threads = 6;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  /* 取得优化后的姿态，即重力加速度方向 */
  /* FIXME:似乎只有在初始化时才对重力加速度方向进行更新，难道不怕方向跑偏？ */
  Eigen::Vector3d r_wg(para_r[0], para_r[1], para_r[2]);
  GravityVector = Sophus::SO3d::exp(r_wg) * Eigen::Vector3d(0, 0, -9.805);

  /* 取得优化后的IMU偏差 */
  Eigen::Vector3d ba_vec(para_ba[0], para_ba[1], para_ba[2]);
  Eigen::Vector3d bg_vec(para_bg[0], para_bg[1], para_bg[2]);

  /* 如果偏差太大则初始化失败 */
  if(ba_vec.norm() > 0.5 || bg_vec.norm() > 0.5) {
    ROS_WARN("Too Large Biases! Initialization Failed!");
    return false;
  }

  /* 用优化后的偏差和速度来更新点云帧列表lidarFrameList */
  for(int i = 0; i < v_size; i++) {
    auto iter = lidarFrameList->begin();
    std::advance(iter, i);
    iter->ba = ba_vec;
    iter->bg = bg_vec;
    Eigen::Vector3d bv_vec(para_v[i][0], para_v[i][1], para_v[i][2]);
    if((bv_vec - prior_v[i]).norm() > 2.0) {
      ROS_WARN("Too Large Velocity! Initialization Failed!");
      std::cout<<"delta v norm: "<<(bv_vec - prior_v[i]).norm()<<std::endl;
      return false;
    }
    iter->V = bv_vec;
  }

  /* 基于IMU偏差的优化值，进行i、j两帧之间的IMU预积分 */
  for(size_t i = 0; i < v_size - 1; i++){
    auto laser_trans_i = lidarFrameList->begin();
    auto laser_trans_j = lidarFrameList->begin();
    std::advance(laser_trans_i, i);
    std::advance(laser_trans_j, i+1);
    laser_trans_j->imuIntegrator.PreIntegration(laser_trans_i->timeStamp, laser_trans_i->bg, laser_trans_i->ba);
  }

  /* 删除超出滑动窗口的点云帧 */
  // //if IMU success initialized
  WINDOWSIZE = Estimator::SLIDEWINDOWSIZE;
  while(lidarFrameList->size() > WINDOWSIZE){
	  lidarFrameList->pop_front();
  }
  /* 将位置和姿态转成IMU坐标系 */
	Eigen::Vector3d Pwl = lidarFrameList->back().P;
	Eigen::Quaterniond Qwl = lidarFrameList->back().Q;
	lidarFrameList->back().P = Pwl + Qwl*exPlb;
	lidarFrameList->back().Q = Qwl * exRlb;

	// std::cout << "\n=============================\n| Initialization Successful |"<<"\n=============================\n" << std::endl;
  
  return true;
}


/** \brief Mapping main thread
  *   建图主线程
  *    1. 角速度和线速度积分，位姿估计
  *    2. 点云帧内校正
  *    3. 位姿优化
  */
void process(){
    double time_last_lidar = -1;
    double time_curr_lidar = -1;
    Eigen::Matrix3d delta_Rl = Eigen::Matrix3d::Identity();
    Eigen::Vector3d delta_tl = Eigen::Vector3d::Zero();
	Eigen::Matrix3d delta_Rb = Eigen::Matrix3d::Identity();
	Eigen::Vector3d delta_tb = Eigen::Vector3d::Zero();
    std::vector<sensor_msgs::ImuConstPtr> vimuMsg;
  
    while(ros::ok()){

        /* 从队列中获取一帧点云，获取之前先加锁 */
        newfullCloud = false;
        laserCloudFullRes.reset(new pcl::PointCloud<PointType>());
        std::unique_lock<std::mutex> lock_lidar(_mutexLidarQueue);
        if(!_lidarMsgQueue.empty()){
            // get new lidar msg
            time_curr_lidar = _lidarMsgQueue.front()->header.stamp.toSec();
            pcl::fromROSMsg(*_lidarMsgQueue.front(), *laserCloudFullRes);
            _lidarMsgQueue.pop();
            newfullCloud = true;
        }
        lock_lidar.unlock();

        if(newfullCloud){ //获取点云成功

            /* 获取两帧点云之间的IMU数据 */
            nav_msgs::Odometry debugInfo;
            debugInfo.pose.pose.position.x = 0;
            debugInfo.pose.pose.position.y = 0;
            debugInfo.pose.pose.position.z = 0;
            if(IMU_Mode > 0 && time_last_lidar > 0){
                // get IMU msg int the Specified time interval
                vimuMsg.clear();
                int countFail = 0;
                while (!fetchImuMsgs(time_last_lidar, time_curr_lidar, vimuMsg)) {
                    countFail++;
                    if (countFail > 100){
                        break;
                    }
                    std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
                }
            }
      
            /* 初始化当前点云帧 */
            // this lidar frame init
            Estimator::LidarFrame lidarFrame;
            lidarFrame.laserCloud = laserCloudFullRes;
            lidarFrame.timeStamp = time_curr_lidar;

            /* 创建共享指针lidar_list */
            boost::shared_ptr<std::list<Estimator::LidarFrame>> lidar_list;
            if(!vimuMsg.empty()){
                if(!LidarIMUInited) { /* 尚未完成初始化 */
                
                    /* 所谓尚未完成初始化，主要指的是尚未找到重力加速度的方向 */
                    /* 对比初始化和未初始化，一个重要区别是是否使用了重力加速度GravityVector */
                    /* 未初始化时，尚未找到重力加速度方向，因此没有使用重力加速度 */
                    
                    /* 对比初始化和未初始化，另一个重要区别是： */
                    /* 初始化之前，仅仅是对角速度进行积分 */
                    /* 初始化之后，进行完整的IMU预积分，计算P、V、Q，计算测量噪声的协方差矩阵 */
                    
                    // if get IMU msg successfully, use gyro integration to update delta_Rl
                    /* 对IMU的角速度进行积分 */
                    lidarFrame.imuIntegrator.PushIMUMsg(vimuMsg);
                    lidarFrame.imuIntegrator.GyroIntegration(time_last_lidar);
                    /* 获得IMU的旋转增量，获得Lidar的旋转增量 */
                    delta_Rb = lidarFrame.imuIntegrator.GetDeltaQ().toRotationMatrix();
                    /* FIXME: 这里为什么要用外参矩阵及其转置夹乘IMU旋转增量的方式转成Lidar旋转增量？ */
                    delta_Rl = exTlb.topLeftCorner(3, 3) * delta_Rb * exTlb.topLeftCorner(3, 3).transpose();

                    /* 对比初始化和未初始化，再一个重要区别是： */
                    /* 初始化之前，从transformAftMapped获得前一帧位姿，叠加IMU积分增量后获得当前帧的位姿 */
                    /* 初始化之后，从上一帧即lidarFrameList->back()获得前一帧位姿，叠加IMU积分增量后获得当前帧的位姿 */

                    // predict current lidar pose
                    /* 估计雷达的位置P和姿态Q */
                    lidarFrame.P = transformAftMapped.topLeftCorner(3,3) * delta_tb
                                   + transformAftMapped.topRightCorner(3,1);
                    Eigen::Matrix3d m3d = transformAftMapped.topLeftCorner(3,3) * delta_Rb;
                    lidarFrame.Q = m3d;

                    lidar_list.reset(new std::list<Estimator::LidarFrame>);
                    lidar_list->push_back(lidarFrame);
                    
                }else{ /* 已经完成初始化 */
                
                    /* FIXME:似乎只有在初始化时才对重力加速度方向GravityVector进行更新，难道不怕方向跑偏？ */
                
                    // if get IMU msg successfully, use pre-integration to update delta lidar pose
                    /* 对角速度和线速度进行预积分 */
                    lidarFrame.imuIntegrator.PushIMUMsg(vimuMsg);
                    lidarFrame.imuIntegrator.PreIntegration(lidarFrameList->back().timeStamp, lidarFrameList->back().bg, lidarFrameList->back().ba);

                    /* 获得之前（即截至上一点云帧）的IMU在世界坐标系下的位置P、姿态Q、速度V */
                    /* Pwbpre中的P表示位置，w表示世界坐标系，b表示IMU，pre表示上一点云帧对应的 */
                    const Eigen::Vector3d& Pwbpre = lidarFrameList->back().P;
                    const Eigen::Quaterniond& Qwbpre = lidarFrameList->back().Q;
                    const Eigen::Vector3d& Vwbpre = lidarFrameList->back().V;

                    /* 获得当前（即当前点云帧对应的）的IMU位置增量dP，姿态增量dQ，速度增量dV */
                    const Eigen::Quaterniond& dQ =  lidarFrame.imuIntegrator.GetDeltaQ();
                    const Eigen::Vector3d& dP = lidarFrame.imuIntegrator.GetDeltaP();
                    const Eigen::Vector3d& dV = lidarFrame.imuIntegrator.GetDeltaV();
                    double dt = lidarFrame.imuIntegrator.GetDeltaTime();

                    /* 叠加增量到上一帧，更新当前帧的IMU位置、姿态、速度和偏差 */
                    /* 注意这里的偏差直接从上一帧获得，上一帧的偏差是经过优化的 */
                    lidarFrame.Q = Qwbpre * dQ;
                    lidarFrame.P = Pwbpre + Vwbpre*dt + 0.5*GravityVector*dt*dt + Qwbpre*(dP);
                    lidarFrame.V = Vwbpre + GravityVector*dt + Qwbpre*(dV);
                    lidarFrame.bg = lidarFrameList->back().bg;
                    lidarFrame.ba = lidarFrameList->back().ba;

                    /* 利用外参将上一帧的IMU的姿态和位置转到Lidar坐标系 */
                    Eigen::Quaterniond Qwlpre = Qwbpre * Eigen::Quaterniond(exRbl);
                    Eigen::Vector3d Pwlpre = Qwbpre * exPbl + Pwbpre;

                    /* 利用外参将当前帧的IMU的姿态和位置转到Lidar坐标系 */
                    Eigen::Quaterniond Qwl = lidarFrame.Q * Eigen::Quaterniond(exRbl);
                    Eigen::Vector3d Pwl = lidarFrame.Q * exPbl + lidarFrame.P;

                    /* 求Lidar的旋转增量和平移增量 */
                    delta_Rl = Qwlpre.conjugate() * Qwl;
                    delta_tl = Qwlpre.conjugate() * (Pwl - Pwlpre);
                    /* 获得IMU的旋转增量和平移增量 */
                    delta_Rb = dQ.toRotationMatrix();
                    delta_tb = dP;

                    /* 将当前点云帧添加到点云帧列表末尾中，并清除旧列表头端点云 */
                    /* 从IMU预积分的算法来看，队列中应该始终只有两帧点云：i和j */
                    /* 在IMU_Mode=1（即用IMU进行帧内校正）时lidarFrameList的长度为1 */
                    lidarFrameList->push_back(lidarFrame);
                    lidarFrameList->pop_front();
                    lidar_list = lidarFrameList;
                }
            }else{
                if(LidarIMUInited)
                  break;
                else{
                    // predict current lidar pose
                    lidarFrame.P = transformAftMapped.topLeftCorner(3,3) * delta_tb
                                   + transformAftMapped.topRightCorner(3,1);
                    Eigen::Matrix3d m3d = transformAftMapped.topLeftCorner(3,3) * delta_Rb;
                    lidarFrame.Q = m3d;

                    lidar_list.reset(new std::list<Estimator::LidarFrame>);
                    lidar_list->push_back(lidarFrame);
                }
            }

            /* 不论IMU是否初始化，下面都会进行扫描点云到Map的匹配优化，从而获得每一帧的位姿，包括lidarFrame.Q、
             * lidarFrame.P、lidarFrame.V等，这样才能在IMU初始化时，基于IMU预积分代价函数，通过优化获得重力加
             * 速度的方向和IMU偏差等，这也是为什么将IMU的初始化TryMAPInitialization()放在最后面的原因，必须等
             * 到至少有了三帧的点云数据之后才会启动初始化。 */
            
            /* 点云帧内校正 */
            // remove lidar distortion
            RemoveLidarDistortion(laserCloudFullRes, delta_Rl, delta_tl);

            /* 位姿优化 */
            // optimize current lidar pose with IMU
            estimator->EstimateLidarPose(*lidar_list, exTlb, GravityVector, debugInfo);

            pcl::PointCloud<PointType>::Ptr laserCloudCornerMap(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr laserCloudSurfMap(new pcl::PointCloud<PointType>());

            /* 获得优化后的Lidar位姿 */
            Eigen::Matrix4d transformTobeMapped = Eigen::Matrix4d::Identity();
            transformTobeMapped.topLeftCorner(3,3) = lidar_list->front().Q * exRbl;
            transformTobeMapped.topRightCorner(3,1) = lidar_list->front().Q * exPbl + lidar_list->front().P;

            /* 更新优化后的位姿增量 */
            // update delta transformation
            /* 从优化后的IMU位姿中扣减掉上一帧的IMU位姿，得到优化后的IMU位姿增量 */
            delta_Rb = transformAftMapped.topLeftCorner(3, 3).transpose() * lidar_list->front().Q.toRotationMatrix();
            delta_tb = transformAftMapped.topLeftCorner(3, 3).transpose() * (lidar_list->front().P - transformAftMapped.topRightCorner(3, 1));
            /* 利用外参将将上一帧的IMU的位姿转成上一帧的Lidar的位姿 */
            Eigen::Matrix3d Rwlpre = transformAftMapped.topLeftCorner(3, 3) * exRbl;
            Eigen::Vector3d Pwlpre = transformAftMapped.topLeftCorner(3, 3) * exPbl + transformAftMapped.topRightCorner(3, 1);
            /* 从优化后的Lidar位姿中扣减掉上一帧的Lidar位姿，得到优化后的Lidar位姿增量 */
            delta_Rl = Rwlpre.transpose() * transformTobeMapped.topLeftCorner(3,3);
            delta_tl = Rwlpre.transpose() * (transformTobeMapped.topRightCorner(3,1) - Pwlpre);
            /* 将优化后的IMU位姿更新到transformAftMapped中 */
            transformAftMapped.topLeftCorner(3,3) = lidar_list->front().Q.toRotationMatrix();
            transformAftMapped.topRightCorner(3,1) = lidar_list->front().P;

            /* 发布优化后的Ldiar位姿 */
            // publish odometry rostopic
            pubOdometry(transformTobeMapped, lidar_list->front().timeStamp);

            /* 将当前点云帧转到地图坐标系并发布 */
            // publish lidar points
            int laserCloudFullResNum = lidar_list->front().laserCloud->points.size();
            pcl::PointCloud<PointType>::Ptr laserCloudAfterEstimate(new pcl::PointCloud<PointType>());
            laserCloudAfterEstimate->reserve(laserCloudFullResNum);
            for (int i = 0; i < laserCloudFullResNum; i++) {
                PointType temp_point;
                MAP_MANAGER::pointAssociateToMap(&lidar_list->front().laserCloud->points[i], &temp_point, transformTobeMapped);
                laserCloudAfterEstimate->push_back(temp_point);
            }
            sensor_msgs::PointCloud2 laserCloudMsg;
            pcl::toROSMsg(*laserCloudAfterEstimate, laserCloudMsg);
            laserCloudMsg.header.frame_id = "/world";
            laserCloudMsg.header.stamp.fromSec(lidar_list->front().timeStamp);
            pubFullLaserCloud.publish(laserCloudMsg);

            /* 如果尚未完成初始化，则进行初始化 */
            // if tightly coupled IMU message, start IMU initialization
            if(IMU_Mode > 1 && !LidarIMUInited){
                // update lidar frame pose
                lidarFrame.P = transformTobeMapped.topRightCorner(3,1);
                Eigen::Matrix3d m3d = transformTobeMapped.topLeftCorner(3,3);
                lidarFrame.Q = m3d;

                // static int pushCount = 0;
                if(pushCount == 0){
                    lidarFrameList->push_back(lidarFrame);
                    lidarFrameList->back().imuIntegrator.Reset();
                    if(lidarFrameList->size() > WINDOWSIZE)
                        lidarFrameList->pop_front();
                }else{
                    lidarFrameList->back().laserCloud = lidarFrame.laserCloud;
                    lidarFrameList->back().imuIntegrator.PushIMUMsg(vimuMsg);
                    lidarFrameList->back().timeStamp = lidarFrame.timeStamp;
                    lidarFrameList->back().P = lidarFrame.P;
                    lidarFrameList->back().Q = lidarFrame.Q;
                }
                pushCount++;
                
                /* 在收到三帧点云后才能开始初始化 */
                /* 初始化之前，确保这三帧点云都进行了扫描点云到Map的匹配优化，这三帧优化确保每帧点云的 */
                /* lidarFrame.P、Q、V都有初值，基于这个初值才能在初始化过程中通过优化获得重力加速度方向、
                 * IMU偏差等 */
                if (pushCount >= 3){
                    pushCount = 0;
                    if(lidarFrameList->size() > 1){
                        auto iterRight = std::prev(lidarFrameList->end());
                        auto iterLeft = std::prev(std::prev(lidarFrameList->end()));
                        iterRight->imuIntegrator.PreIntegration(iterLeft->timeStamp, iterLeft->bg, iterLeft->ba);
                    }

                    if (lidarFrameList->size() == int(WINDOWSIZE / 1.5)) {
                        startTime = lidarFrameList->back().timeStamp;
                    }
                    
                    /* 开始Map的初始化 */
                    if (!LidarIMUInited && lidarFrameList->size() == WINDOWSIZE && lidarFrameList->front().timeStamp >= startTime){
                        std::cout<<"**************Start MAP Initialization!!!******************"<<std::endl;
                        if(TryMAPInitialization()){
                            /* 初始化成功 */
                            LidarIMUInited = true;
                            pushCount = 0;
                            startTime = 0;
                        }
                        std::cout<<"**************Finish MAP Initialization!!!******************"<<std::endl;
                    }
                }
            }
            time_last_lidar = time_curr_lidar;
        }
    }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "PoseEstimation");
  ros::NodeHandle nodeHandler("~");

  ros::param::get("~filter_parameter_corner",filter_parameter_corner);
  ros::param::get("~filter_parameter_surf",filter_parameter_surf);
  ros::param::get("~IMU_Mode",IMU_Mode);
	std::vector<double> vecTlb;
	ros::param::get("~Extrinsic_Tlb",vecTlb);

  /* 设置雷达和IMU之间的外参 */
  /* Livox雷达与其内部集成的IMU之间只有平移，没有旋转  */
  // set extrinsic matrix between lidar & IMU
  Eigen::Matrix3d R;
  Eigen::Vector3d t;
	R << vecTlb[0], vecTlb[1], vecTlb[2],
	     vecTlb[4], vecTlb[5], vecTlb[6],
	     vecTlb[8], vecTlb[9], vecTlb[10];
	t << vecTlb[3], vecTlb[7], vecTlb[11];
  /* 旋转矩阵直接初始化为四元数 */
  Eigen::Quaterniond qr(R);
  /* 四元数再转成旋转矩阵 */
  R = qr.normalized().toRotationMatrix();
  exTlb.topLeftCorner(3,3) = R;
  exTlb.topRightCorner(3,1) = t;
  exRlb = R;            //从Lidar到IMU的旋转
  exRbl = R.transpose();//从IMU到Lidar的旋转
  exPlb = t;            //从Lidar到IMU的平移
  exPbl = -1.0 * exRbl * exPlb;//从IMU到Lidar的平移

  /* 订阅完整点云 */
  ros::Subscriber subFullCloud = nodeHandler.subscribe<sensor_msgs::PointCloud2>("/livox_full_cloud", 10, fullCallBack);
  /* 订阅IMU */
  ros::Subscriber sub_imu;
  if(IMU_Mode > 0)
    sub_imu = nodeHandler.subscribe("/livox/imu", 2000, imu_callback, ros::TransportHints().unreliable());
  if(IMU_Mode < 2)
    WINDOWSIZE = 1;
  else
    WINDOWSIZE = 20;

  /* 发布建图后的完整点云 */
  pubFullLaserCloud = nodeHandler.advertise<sensor_msgs::PointCloud2>("/livox_full_cloud_mapped", 10);
  /* 发布建图后的里程 */
  pubLaserOdometry = nodeHandler.advertise<nav_msgs::Odometry> ("/livox_odometry_mapped", 5);
  /* 发布…… */
  pubLaserOdometryPath = nodeHandler.advertise<nav_msgs::Path> ("/livox_odometry_path_mapped", 5);
	pubGps = nodeHandler.advertise<sensor_msgs::NavSatFix>("/lidar", 1000);

  tfBroadcaster = new tf::TransformBroadcaster();

  laserCloudFullRes.reset(new pcl::PointCloud<PointType>);
  estimator = new Estimator(filter_parameter_corner, filter_parameter_surf);
	lidarFrameList.reset(new std::list<Estimator::LidarFrame>);

  /* 启动位姿估计线程 */
  std::thread thread_process{process};
  ros::spin();

  return 0;
}

