#include "Estimator/Estimator.h"

Estimator::Estimator(const float& filter_corner, const float& filter_surf){
  laserCloudCornerFromLocal.reset(new pcl::PointCloud<PointType>);
  laserCloudSurfFromLocal.reset(new pcl::PointCloud<PointType>);
  laserCloudNonFeatureFromLocal.reset(new pcl::PointCloud<PointType>);
  laserCloudCornerLast.resize(SLIDEWINDOWSIZE); //滑动窗口的Size是2
  for(auto& p:laserCloudCornerLast)
    p.reset(new pcl::PointCloud<PointType>);
  laserCloudSurfLast.resize(SLIDEWINDOWSIZE);
  for(auto& p:laserCloudSurfLast)
    p.reset(new pcl::PointCloud<PointType>);
  laserCloudNonFeatureLast.resize(SLIDEWINDOWSIZE);
  for(auto& p:laserCloudNonFeatureLast)
    p.reset(new pcl::PointCloud<PointType>);
  laserCloudCornerStack.resize(SLIDEWINDOWSIZE);
  for(auto& p:laserCloudCornerStack)
    p.reset(new pcl::PointCloud<PointType>);
  laserCloudSurfStack.resize(SLIDEWINDOWSIZE);
  for(auto& p:laserCloudSurfStack)
    p.reset(new pcl::PointCloud<PointType>);
  laserCloudNonFeatureStack.resize(SLIDEWINDOWSIZE);
  for(auto& p:laserCloudNonFeatureStack)
    p.reset(new pcl::PointCloud<PointType>);
  laserCloudCornerForMap.reset(new pcl::PointCloud<PointType>);
  laserCloudSurfForMap.reset(new pcl::PointCloud<PointType>);
  laserCloudNonFeatureForMap.reset(new pcl::PointCloud<PointType>);
  transformForMap.setIdentity();
  kdtreeCornerFromLocal.reset(new pcl::KdTreeFLANN<PointType>);
  kdtreeSurfFromLocal.reset(new pcl::KdTreeFLANN<PointType>);
  kdtreeNonFeatureFromLocal.reset(new pcl::KdTreeFLANN<PointType>);

  for(int i = 0; i < localMapWindowSize; i++){
    localCornerMap[i].reset(new pcl::PointCloud<PointType>);
    localSurfMap[i].reset(new pcl::PointCloud<PointType>);
    localNonFeatureMap[i].reset(new pcl::PointCloud<PointType>);
  }

  downSizeFilterCorner.setLeafSize(filter_corner, filter_corner, filter_corner);
  downSizeFilterSurf.setLeafSize(filter_surf, filter_surf, filter_surf);
  downSizeFilterNonFeature.setLeafSize(0.4, 0.4, 0.4);
  
  /* 创建地图生成器，并启动独立线程，负责地图的生成 */
  map_manager = new MAP_MANAGER(filter_corner, filter_surf);
  threadMap = std::thread(&Estimator::threadMapIncrement, this);
}

Estimator::~Estimator(){
  delete map_manager;
}

/** @brief 建图线程
  *   将ForMap容器中的点云添加到Map
  *   通过互斥信号量mtx_Map实现与EstimateLidarPose实现同步，一个消费，一个生产
  */
[[noreturn]] void Estimator::threadMapIncrement(){
  pcl::PointCloud<PointType>::Ptr laserCloudCorner(new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr laserCloudSurf(new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr laserCloudNonFeature(new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr laserCloudCorner_to_map(new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr laserCloudSurf_to_map(new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr laserCloudNonFeature_to_map(new pcl::PointCloud<PointType>);
  Eigen::Matrix4d transform;
  while(true){
    /* 加锁保护点云确保ForMap容器的原子操作 */
    std::unique_lock<std::mutex> locker(mtx_Map);
    if(!laserCloudCornerForMap->empty()){

      map_update_ID ++;

      /* 将点云转成Map坐标系 */
      map_manager->featureAssociateToMap(laserCloudCornerForMap,    //转换前的角点
                                         laserCloudSurfForMap,      //转换前的平面点
                                         laserCloudNonFeatureForMap,//转换前的不规则点
                                         laserCloudCorner,          //转换后的角点
                                         laserCloudSurf,            //转换后的平面点
                                         laserCloudNonFeature,      //转换后的不规则点
                                         transformForMap);          //转换矩阵
      /* 清空转换前的ForMap容器 */
      laserCloudCornerForMap->clear();
      laserCloudSurfForMap->clear();
      laserCloudNonFeatureForMap->clear();
      transform = transformForMap;
      locker.unlock();
    
      /* 转换后的点云添加到_to_map容器 */
      *laserCloudCorner_to_map += *laserCloudCorner;
      *laserCloudSurf_to_map += *laserCloudSurf;
      *laserCloudNonFeature_to_map += *laserCloudNonFeature;

      /* 清空转换后的容器 */
      laserCloudCorner->clear();
      laserCloudSurf->clear();
      laserCloudNonFeature->clear();

      /* 将新的点云添加到Map */
      if(map_update_ID % map_skip_frame == 0){
        map_manager->MapIncrement(laserCloudCorner_to_map, 
                                  laserCloudSurf_to_map, 
                                  laserCloudNonFeature_to_map,
                                  transform);

        laserCloudCorner_to_map->clear();
        laserCloudSurf_to_map->clear();
        laserCloudNonFeature_to_map->clear();
      }
      
    }else
      locker.unlock();

    /* 休眠2毫秒，避免死等 */
    std::chrono::milliseconds dura(2);
    std::this_thread::sleep_for(dura);
  }
}

/** @brief 对于待匹配角点特征点云中的每个点p，在Map找到最近的线特征，在线特征上构造a、b两个点，然后以点p到
  * 直线ab的距离为优化目标，构造一个代价函数，添加到edges中。同时，直接计算点p到直线ab的距离，保存到vLineFeatures
  * 中，如果该距离小于某个阈值，则对应的代价函数不需要优化。
  * @param [out] edges 构造好的代价函数
  * @param [out] vLineFeatures 线特征容器
  * @param [in] laserCloudCorner 角点特征点云，即待匹配点云
  * @param [in] laserCloudCornerLocal 本地Map
  * @param [in] kdtreeLocal 用本地Map建立的KDtree
  * @param [in] exTlb Lidar与IMU之间的外参矩阵
  * @param [in] m4d 待匹配点云的估计位姿
  */
void Estimator::processPointToLine(std::vector<ceres::CostFunction *>& edges,
                                   std::vector<FeatureLine>& vLineFeatures,
                                   const pcl::PointCloud<PointType>::Ptr& laserCloudCorner,
                                   const pcl::PointCloud<PointType>::Ptr& laserCloudCornerLocal,
                                   const pcl::KdTreeFLANN<PointType>::Ptr& kdtreeLocal,
                                   const Eigen::Matrix4d& exTlb,
                                   const Eigen::Matrix4d& m4d){

  /* 求IMU到Lidar的坐标转换矩阵 */
  Eigen::Matrix4d Tbl = Eigen::Matrix4d::Identity();
  Tbl.topLeftCorner(3,3) = exTlb.topLeftCorner(3,3).transpose();
  Tbl.topRightCorner(3,1) = -1.0 * Tbl.topLeftCorner(3,3) * exTlb.topRightCorner(3,1);
  
  /* 如果vLineFeatures不空，直接从中取得p、a、b三个点构造代价函数 */
  /* 免去求最近5点，求质心，求协方差矩阵，求特征向量的过程 */
  if(!vLineFeatures.empty()){
    for(const auto& l : vLineFeatures){
      auto* e = Cost_NavState_IMU_Line::Create(l.pointOri,
                                               l.lineP1,
                                               l.lineP2,
                                               Tbl,
                                               Eigen::Matrix<double, 1, 1>(1/IMUIntegrator::lidar_m));
      edges.push_back(e);
    }
    return;
  }
  
  PointType _pointOri, _pointSel, _coeff;
  std::vector<int> _pointSearchInd;
  std::vector<float> _pointSearchSqDis;
  std::vector<int> _pointSearchInd2;
  std::vector<float> _pointSearchSqDis2;

  Eigen::Matrix< double, 3, 3 > _matA1;
  _matA1.setZero();

  int laserCloudCornerStackNum = laserCloudCorner->points.size();
  pcl::PointCloud<PointType>::Ptr kd_pointcloud(new pcl::PointCloud<PointType>);
  int debug_num1 = 0;
  int debug_num2 = 0;
  int debug_num12 = 0;
  int debug_num22 = 0;
  
  /* 为待匹配的特征点云中的每一个点构造一个代价函数添加到edges中 */
  for (int i = 0; i < laserCloudCornerStackNum; i++) {
    /* 从待匹配点云中取一个点 */
    _pointOri = laserCloudCorner->points[i];
    /* 转到Map坐标系。注意：这里为了速度快，没有用到外参矩阵 */
    MAP_MANAGER::pointAssociateToMap(&_pointOri, &_pointSel, m4d);
    /* 找到与该点对应的MapID */
    int id = map_manager->FindUsedCornerMap(&_pointSel,laserCenWidth_last,laserCenHeight_last,laserCenDepth_last);

    if(id == 5000) continue;

    if(std::isnan(_pointSel.x) || std::isnan(_pointSel.y) ||std::isnan(_pointSel.z)) continue;

    /* 基于当前点到全局Map的距离构造残差，构造代价函数 */
    /* 如果对应的Map规模大于100个点，则在Map中查找5个最近点，求质心，求协方差矩阵，求特征向量
     * 然后沿主方向在质心两侧构造a、b两个点，求p到ab的距离，当p位于ab中心时残差具有最小值	*/
    if(GlobalCornerMap[id].points.size() > 100) {
      CornerKdMap[id].nearestKSearch(_pointSel, 5, _pointSearchInd, _pointSearchSqDis);
      
      /* 确保五个点足够近 */
      if (_pointSearchSqDis[4] < thres_dist) {

        /* 求五个点的质心 */
        debug_num1 ++;
        float cx = 0;
        float cy = 0;
        float cz = 0;
        for (int j = 0; j < 5; j++) {
          cx += GlobalCornerMap[id].points[_pointSearchInd[j]].x;
          cy += GlobalCornerMap[id].points[_pointSearchInd[j]].y;
          cz += GlobalCornerMap[id].points[_pointSearchInd[j]].z;
        }
        cx /= 5;
        cy /= 5;
        cz /= 5;

        /* 求五个点的协方差矩阵_matA1 */
        float a11 = 0;
        float a12 = 0;
        float a13 = 0;
        float a22 = 0;
        float a23 = 0;
        float a33 = 0;
        for (int j = 0; j < 5; j++) {
          float ax = GlobalCornerMap[id].points[_pointSearchInd[j]].x - cx;
          float ay = GlobalCornerMap[id].points[_pointSearchInd[j]].y - cy;
          float az = GlobalCornerMap[id].points[_pointSearchInd[j]].z - cz;

          a11 += ax * ax;
          a12 += ax * ay;
          a13 += ax * az;
          a22 += ay * ay;
          a23 += ay * az;
          a33 += az * az;
        }
        a11 /= 5;
        a12 /= 5;
        a13 /= 5;
        a22 /= 5;
        a23 /= 5;
        a33 /= 5;

        _matA1(0, 0) = a11;
        _matA1(0, 1) = a12;
        _matA1(0, 2) = a13;
        _matA1(1, 0) = a12;
        _matA1(1, 1) = a22;
        _matA1(1, 2) = a23;
        _matA1(2, 0) = a13;
        _matA1(2, 1) = a23;
        _matA1(2, 2) = a33;

        /* 求协方差矩阵的特征向量和特征值 */
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(_matA1);
        Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);

        /* 确保主方向远大于次主方向，即五个点呈直线排列 */
        if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1]) {
          debug_num12 ++;
          /* 沿最大特征向量方向，以质心为中心，在质心两侧构造a、b两个点 */
          float x1 = cx + 0.1 * unit_direction[0];
          float y1 = cy + 0.1 * unit_direction[1];
          float z1 = cz + 0.1 * unit_direction[2];
          float x2 = cx - 0.1 * unit_direction[0];
          float y2 = cy - 0.1 * unit_direction[1];
          float z2 = cz - 0.1 * unit_direction[2];

          /* 将当前点p和a、b两个点加入优化序列，最优的结果是a、p、b三点一线，p在正中间 */
          Eigen::Vector3d tripod1(x1, y1, z1);
          Eigen::Vector3d tripod2(x2, y2, z2);

          /* 构造角点特征点到Map的代价函数 */
          /* 点p到直线ab的距离即是残差 */
          /* 注意下面用于构造代价函数的p点用的是原始点，而不是转换到Map坐标系的点_pointSel */
          /* 因为_pointSel不包含IMU到Lidar的外参变换，精度不够 */
          /* FIXME:lidar_m = 1.5e-3，是定义在IMUIntergrator.h文件中的常数，是什么含义？ */
          auto* e = Cost_NavState_IMU_Line::Create(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z), //点p
                                                   tripod1, //点a
                                                   tripod2, //点b
                                                   Tbl,	  //外参矩阵
                                                   // 1/lidar_m 具体的含义还不清楚
                                                   Eigen::Matrix<double, 1, 1>(1/IMUIntegrator::lidar_m));
          /* 保存当前代价函数到edges */
          edges.push_back(e);
          /* p、a、b三个点加入vLineFeatures */
          vLineFeatures.emplace_back(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                     tripod1,
                                     tripod2);
          /* 直接计算点p到直线ab的距离作为误差，如果误差小于某个阈值，则该代价函数不需要优化 */
          vLineFeatures.back().ComputeError(m4d);

          continue;
        }
      }
    }

    /* 基于当前点到本地Map的距离构造残差，构造代价函数 */
    /* 如果对应的Map规模大于20个点，则在Map中查找5个最近点，求质心，求协方差矩阵，求特征向量
     * 然后沿主方向在质心两侧构造a、b两个点，求p到ab的距离，当p位于ab中心时残差具有最小值	*/
    if(laserCloudCornerLocal->points.size() > 20 ){
      kdtreeLocal->nearestKSearch(_pointSel, 5, _pointSearchInd2, _pointSearchSqDis2);
      if (_pointSearchSqDis2[4] < thres_dist) {

        debug_num2 ++;
        float cx = 0;
        float cy = 0;
        float cz = 0;
        for (int j = 0; j < 5; j++) {
          cx += laserCloudCornerLocal->points[_pointSearchInd2[j]].x;
          cy += laserCloudCornerLocal->points[_pointSearchInd2[j]].y;
          cz += laserCloudCornerLocal->points[_pointSearchInd2[j]].z;
        }
        cx /= 5;
        cy /= 5;
        cz /= 5;

        float a11 = 0;
        float a12 = 0;
        float a13 = 0;
        float a22 = 0;
        float a23 = 0;
        float a33 = 0;
        for (int j = 0; j < 5; j++) {
          float ax = laserCloudCornerLocal->points[_pointSearchInd2[j]].x - cx;
          float ay = laserCloudCornerLocal->points[_pointSearchInd2[j]].y - cy;
          float az = laserCloudCornerLocal->points[_pointSearchInd2[j]].z - cz;

          a11 += ax * ax;
          a12 += ax * ay;
          a13 += ax * az;
          a22 += ay * ay;
          a23 += ay * az;
          a33 += az * az;
        }
        a11 /= 5;
        a12 /= 5;
        a13 /= 5;
        a22 /= 5;
        a23 /= 5;
        a33 /= 5;

        _matA1(0, 0) = a11;
        _matA1(0, 1) = a12;
        _matA1(0, 2) = a13;
        _matA1(1, 0) = a12;
        _matA1(1, 1) = a22;
        _matA1(1, 2) = a23;
        _matA1(2, 0) = a13;
        _matA1(2, 1) = a23;
        _matA1(2, 2) = a33;

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(_matA1);
      Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);

        if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1]) {
          debug_num22++;
          float x1 = cx + 0.1 * unit_direction[0];
          float y1 = cy + 0.1 * unit_direction[1];
          float z1 = cz + 0.1 * unit_direction[2];
          float x2 = cx - 0.1 * unit_direction[0];
          float y2 = cy - 0.1 * unit_direction[1];
          float z2 = cz - 0.1 * unit_direction[2];

          Eigen::Vector3d tripod1(x1, y1, z1);
          Eigen::Vector3d tripod2(x2, y2, z2);
          auto* e = Cost_NavState_IMU_Line::Create(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                                  tripod1,
                                                  tripod2,
                                                  Tbl,
                                                  Eigen::Matrix<double, 1, 1>(1/IMUIntegrator::lidar_m));
          edges.push_back(e);
          vLineFeatures.emplace_back(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                    tripod1,
                                    tripod2);
          vLineFeatures.back().ComputeError(m4d);
        }
      }
    }
  }
}

/** @brief 该函数定义了但没有任何地方使用
  */
void Estimator::processPointToPlan(std::vector<ceres::CostFunction *>& edges,
                                   std::vector<FeaturePlan>& vPlanFeatures,
                                   const pcl::PointCloud<PointType>::Ptr& laserCloudSurf,
                                   const pcl::PointCloud<PointType>::Ptr& laserCloudSurfLocal,
                                   const pcl::KdTreeFLANN<PointType>::Ptr& kdtreeLocal,
                                   const Eigen::Matrix4d& exTlb,
                                   const Eigen::Matrix4d& m4d){
  Eigen::Matrix4d Tbl = Eigen::Matrix4d::Identity();
  Tbl.topLeftCorner(3,3) = exTlb.topLeftCorner(3,3).transpose();
  Tbl.topRightCorner(3,1) = -1.0 * Tbl.topLeftCorner(3,3) * exTlb.topRightCorner(3,1);
  if(!vPlanFeatures.empty()){
    for(const auto& p : vPlanFeatures){
      auto* e = Cost_NavState_IMU_Plan::Create(p.pointOri,
                                               p.pa,
                                               p.pb,
                                               p.pc,
                                               p.pd,
                                               Tbl,
                                               Eigen::Matrix<double, 1, 1>(1/IMUIntegrator::lidar_m));
      edges.push_back(e);
    }
    return;
  }
  PointType _pointOri, _pointSel, _coeff;
  std::vector<int> _pointSearchInd;
  std::vector<float> _pointSearchSqDis;
  std::vector<int> _pointSearchInd2;
  std::vector<float> _pointSearchSqDis2;

  Eigen::Matrix< double, 5, 3 > _matA0;
  _matA0.setZero();
  Eigen::Matrix< double, 5, 1 > _matB0;
  _matB0.setOnes();
  _matB0 *= -1;
  Eigen::Matrix< double, 3, 1 > _matX0;
  _matX0.setZero();
  int laserCloudSurfStackNum = laserCloudSurf->points.size();

  int debug_num1 = 0;
  int debug_num2 = 0;
  int debug_num12 = 0;
  int debug_num22 = 0;
  for (int i = 0; i < laserCloudSurfStackNum; i++) {
    _pointOri = laserCloudSurf->points[i];
    MAP_MANAGER::pointAssociateToMap(&_pointOri, &_pointSel, m4d);

    int id = map_manager->FindUsedSurfMap(&_pointSel,laserCenWidth_last,laserCenHeight_last,laserCenDepth_last);

    if(id == 5000) continue;

    if(std::isnan(_pointSel.x) || std::isnan(_pointSel.y) ||std::isnan(_pointSel.z)) continue;

    if(GlobalSurfMap[id].points.size() > 50) {
      SurfKdMap[id].nearestKSearch(_pointSel, 5, _pointSearchInd, _pointSearchSqDis);

      if (_pointSearchSqDis[4] < 1.0) {
        debug_num1 ++;
        for (int j = 0; j < 5; j++) {
          _matA0(j, 0) = GlobalSurfMap[id].points[_pointSearchInd[j]].x;
          _matA0(j, 1) = GlobalSurfMap[id].points[_pointSearchInd[j]].y;
          _matA0(j, 2) = GlobalSurfMap[id].points[_pointSearchInd[j]].z;
        }
        _matX0 = _matA0.colPivHouseholderQr().solve(_matB0);

        float pa = _matX0(0, 0);
        float pb = _matX0(1, 0);
        float pc = _matX0(2, 0);
        float pd = 1;

        float ps = std::sqrt(pa * pa + pb * pb + pc * pc);
        pa /= ps;
        pb /= ps;
        pc /= ps;
        pd /= ps;

        bool planeValid = true;
        for (int j = 0; j < 5; j++) {
          if (std::fabs(pa * GlobalSurfMap[id].points[_pointSearchInd[j]].x +
                        pb * GlobalSurfMap[id].points[_pointSearchInd[j]].y +
                        pc * GlobalSurfMap[id].points[_pointSearchInd[j]].z + pd) > 0.2) {
            planeValid = false;
            break;
          }
        }

        if (planeValid) {
          debug_num12 ++;
          auto* e = Cost_NavState_IMU_Plan::Create(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                                  pa,
                                                  pb,
                                                  pc,
                                                  pd,
                                                  Tbl,
                                                  Eigen::Matrix<double, 1, 1>(1/IMUIntegrator::lidar_m));
          edges.push_back(e);
          vPlanFeatures.emplace_back(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                    pa,
                                    pb,
                                    pc,
                                    pd);
          vPlanFeatures.back().ComputeError(m4d);

          continue;
        }
        
      }
    }
    if(laserCloudSurfLocal->points.size() > 20 ){
      kdtreeLocal->nearestKSearch(_pointSel, 5, _pointSearchInd2, _pointSearchSqDis2);
      if (_pointSearchSqDis2[4] < 1.0) {
        debug_num2++;
        for (int j = 0; j < 5; j++) { 
          _matA0(j, 0) = laserCloudSurfLocal->points[_pointSearchInd2[j]].x;
          _matA0(j, 1) = laserCloudSurfLocal->points[_pointSearchInd2[j]].y;
          _matA0(j, 2) = laserCloudSurfLocal->points[_pointSearchInd2[j]].z;
        }
        _matX0 = _matA0.colPivHouseholderQr().solve(_matB0);

        float pa = _matX0(0, 0);
        float pb = _matX0(1, 0);
        float pc = _matX0(2, 0);
        float pd = 1;

        float ps = std::sqrt(pa * pa + pb * pb + pc * pc);
        pa /= ps;
        pb /= ps;
        pc /= ps;
        pd /= ps;

        bool planeValid = true;
        for (int j = 0; j < 5; j++) {
          if (std::fabs(pa * laserCloudSurfLocal->points[_pointSearchInd2[j]].x +
                        pb * laserCloudSurfLocal->points[_pointSearchInd2[j]].y +
                        pc * laserCloudSurfLocal->points[_pointSearchInd2[j]].z + pd) > 0.2) {
            planeValid = false;
            break;
          }
        }
  
        if (planeValid) {
          debug_num22 ++;
          auto* e = Cost_NavState_IMU_Plan::Create(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                                  pa,
                                                  pb,
                                                  pc,
                                                  pd,
                                                  Tbl,
                                                  Eigen::Matrix<double, 1, 1>(1/IMUIntegrator::lidar_m));
          edges.push_back(e);
          vPlanFeatures.emplace_back(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                    pa,
                                    pb,
                                    pc,
                                    pd);
          vPlanFeatures.back().ComputeError(m4d);
        }
      }
    }
  }
}

/** @brief 求待匹配平面特征点云中每个点到Map的距离，为每个点构造一个代价函数
  * @param [out] edges 构造好的代价函数
  * @param [out] vPlanFeatures 平面特征容器
  * @param [in] laserCloudSurf 平面特征点云，即待匹配点云
  * @param [in] laserCloudSurfLocal 本地Map
  * @param [in] kdtreeLocal 用本地Map建立的KDtree
  * @param [in] exTlb Lidar与IMU之间的外参矩阵
  * @param [in] m4d 待匹配点云的估计位姿
  */
void Estimator::processPointToPlanVec(std::vector<ceres::CostFunction *>& edges,
                                   std::vector<FeaturePlanVec>& vPlanFeatures,
                                   const pcl::PointCloud<PointType>::Ptr& laserCloudSurf,
                                   const pcl::PointCloud<PointType>::Ptr& laserCloudSurfLocal,
                                   const pcl::KdTreeFLANN<PointType>::Ptr& kdtreeLocal,
                                   const Eigen::Matrix4d& exTlb,
                                   const Eigen::Matrix4d& m4d){
  Eigen::Matrix4d Tbl = Eigen::Matrix4d::Identity();
  Tbl.topLeftCorner(3,3) = exTlb.topLeftCorner(3,3).transpose();
  Tbl.topRightCorner(3,1) = -1.0 * Tbl.topLeftCorner(3,3) * exTlb.topRightCorner(3,1);
  
  /* 如果vLineFeatures不空，直接从中取得p、j两个点构造代价函数 */
  /* 免去求最近5点，求平面等的过程 */
  if(!vPlanFeatures.empty()){
    for(const auto& p : vPlanFeatures){
      auto* e = Cost_NavState_IMU_Plan_Vec::Create(p.pointOri,
                                                   p.pointProj,
                                                   Tbl,
                                                   p.sqrt_info);
      edges.push_back(e);
    }
    return;
  }
  
  PointType _pointOri, _pointSel, _coeff;
  std::vector<int> _pointSearchInd;
  std::vector<float> _pointSearchSqDis;
  std::vector<int> _pointSearchInd2;
  std::vector<float> _pointSearchSqDis2;

  Eigen::Matrix< double, 5, 3 > _matA0;
  _matA0.setZero();
  Eigen::Matrix< double, 5, 1 > _matB0;
  _matB0.setOnes();
  _matB0 *= -1;
  Eigen::Matrix< double, 3, 1 > _matX0;
  _matX0.setZero();
  int laserCloudSurfStackNum = laserCloudSurf->points.size();

  int debug_num1 = 0;
  int debug_num2 = 0;
  int debug_num12 = 0;
  int debug_num22 = 0;
  
  for (int i = 0; i < laserCloudSurfStackNum; i++) {
    /* 从待匹配点云中提取一点，变换到Map坐标系 */
    /* 注意：这里为了速度快，没有用到外参矩阵 */
    _pointOri = laserCloudSurf->points[i];
    MAP_MANAGER::pointAssociateToMap(&_pointOri, &_pointSel, m4d);
    /* 找到与该点对应的MapID */
    int id = map_manager->FindUsedSurfMap(&_pointSel,laserCenWidth_last,laserCenHeight_last,laserCenDepth_last);

    if(id == 5000) continue;

    if(std::isnan(_pointSel.x) || std::isnan(_pointSel.y) ||std::isnan(_pointSel.z)) continue;

    /* 基于当前点到全局Map的距离构造残差，构造代价函数 */
    /* 如果对应的Map规模大于50个点，则在Map中查找5个最近点，求5个点所在平面
       然后求点p在该平面上的投影点，p点到投影点的距离就是残差*/
    if(GlobalSurfMap[id].points.size() > 50) {
      SurfKdMap[id].nearestKSearch(_pointSel, 5, _pointSearchInd, _pointSearchSqDis);

      if (_pointSearchSqDis[4] < thres_dist) {
		  
        /* 求解三元一次方程组Ax=b，进行平面的法向量估计
         * 三元一次方程Ax+By+Cz+D=0对应于空间平面，向量n=(A,B,C)是其法向量，
         * 这里已知五个点在同一平面，且设定D为1，则一定可以找到唯一的一组法
         * 向量n=(A,B,C)与该平面对应，调用A.colPivHouseholderQr().solve(b)
         * 求解三元一次方程组，获得法向量n=(A,B,C)。*/
        debug_num1 ++;
        for (int j = 0; j < 5; j++) {
          _matA0(j, 0) = GlobalSurfMap[id].points[_pointSearchInd[j]].x;
          _matA0(j, 1) = GlobalSurfMap[id].points[_pointSearchInd[j]].y;
          _matA0(j, 2) = GlobalSurfMap[id].points[_pointSearchInd[j]].z;
        }
        _matX0 = _matA0.colPivHouseholderQr().solve(_matB0);

        /* 下面用法向量检查这五个点是否构成一个严格的平面
         * 在平面上的点p(x,y,z)一定满足方程Ax+By+Cz+1=0,如果法向量(A,B,C)的范数
         * 是n，则方程的两边同时除以n，等式仍然成立：(A/n)x+(B/n)y+(C/n)z+1/n=0。
         * 不满足这个等式的点不在该平面上。*/
        float pa = _matX0(0, 0);
        float pb = _matX0(1, 0);
        float pc = _matX0(2, 0);
        float pd = 1;

        float ps = std::sqrt(pa * pa + pb * pb + pc * pc);
        pa /= ps;
        pb /= ps;
        pc /= ps;
        pd /= ps;

        /* 计算等式(A/n)x+(B/n)y+(C/n)z+1/n的值，并检查是否≤0.2 */
        bool planeValid = true;
        for (int j = 0; j < 5; j++) {
          if (std::fabs(pa * GlobalSurfMap[id].points[_pointSearchInd[j]].x +
                        pb * GlobalSurfMap[id].points[_pointSearchInd[j]].y +
                        pc * GlobalSurfMap[id].points[_pointSearchInd[j]].z + pd) > 0.2) {
            planeValid = false;
            break;
          }
        }

        if (planeValid) {
          
          debug_num12 ++;
          
          /* 求点p在平面上的投影点point_proj */
          /* 求点p到平面的距离dist */
          double dist = pa * _pointSel.x +
                        pb * _pointSel.y +
                        pc * _pointSel.z + pd;
          /* 构造平面单位法向量omega，(dist*omega)即从平面到点p的法向量 */
          Eigen::Vector3d omega(pa, pb, pc);
          /* 构造点p对应的向量Vector3d，(Vector3d-点p法向量)即是点p在平面上的投影点的向量 */
          Eigen::Vector3d point_proj = Eigen::Vector3d(_pointSel.x,_pointSel.y,_pointSel.z) - (dist * omega);
          
          /* 构造信息矩阵 */
          /* FIXME:构造左乘的信息矩阵用到了SVD分解，具体的算法原理还不是很明白 */
          /* 构造J矩阵如下：
           * | A B C |
           * | 0 0 0 |
           * | 0 0 0 | */
          Eigen::Vector3d e1(1, 0, 0);
          Eigen::Matrix3d J = e1 * omega.transpose();
          /* 对矩阵J进行SVD分解：J=UΣV^T， */
          Eigen::JacobiSVD<Eigen::Matrix3d> svd(J, Eigen::ComputeThinU | Eigen::ComputeThinV);
          Eigen::Matrix3d R_svd = svd.matrixV() * svd.matrixU().transpose();
          Eigen::Matrix3d info = (1.0/IMUIntegrator::lidar_m) * Eigen::Matrix3d::Identity();
          info(1, 1) *= plan_weight_tan;
          info(2, 2) *= plan_weight_tan;
          Eigen::Matrix3d sqrt_info = info * R_svd.transpose();

          /* 构造平面特征点到Map的代价函数 */
          /* 点p与其在Map平面投影点的差就是残差 */
          /* 注意下面用于构造代价函数的p点用的是原始点，而不是转换到Map坐标系的点_pointSel */
          /* 因为_pointSel不包含IMU到Lidar的外参变换，精度不够 */
          auto* e = Cost_NavState_IMU_Plan_Vec::Create(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                                       point_proj,
                                                       Tbl,
                                                       sqrt_info);
          edges.push_back(e);
          vPlanFeatures.emplace_back(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                     point_proj,
                                     sqrt_info);
          vPlanFeatures.back().ComputeError(m4d);

          continue;
        }
      }
    }

    /* 基于当前点到本地Map的距离构造残差，构造代价函数 */
    /* 如果对应的Map规模大于20个点，则在Map中查找5个最近点，求5个点所在平面
       然后求点p在该平面上的投影点，p点到投影点的距离就是残差*/
    if(laserCloudSurfLocal->points.size() > 20 ){
      kdtreeLocal->nearestKSearch(_pointSel, 5, _pointSearchInd2, _pointSearchSqDis2);
      if (_pointSearchSqDis2[4] < thres_dist) {
        debug_num2++;
        for (int j = 0; j < 5; j++) { 
          _matA0(j, 0) = laserCloudSurfLocal->points[_pointSearchInd2[j]].x;
          _matA0(j, 1) = laserCloudSurfLocal->points[_pointSearchInd2[j]].y;
          _matA0(j, 2) = laserCloudSurfLocal->points[_pointSearchInd2[j]].z;
        }
        _matX0 = _matA0.colPivHouseholderQr().solve(_matB0);

        float pa = _matX0(0, 0);
        float pb = _matX0(1, 0);
        float pc = _matX0(2, 0);
        float pd = 1;

        float ps = std::sqrt(pa * pa + pb * pb + pc * pc);
        pa /= ps;
        pb /= ps;
        pc /= ps;
        pd /= ps;

        bool planeValid = true;
        for (int j = 0; j < 5; j++) {
          if (std::fabs(pa * laserCloudSurfLocal->points[_pointSearchInd2[j]].x +
                        pb * laserCloudSurfLocal->points[_pointSearchInd2[j]].y +
                        pc * laserCloudSurfLocal->points[_pointSearchInd2[j]].z + pd) > 0.2) {
            planeValid = false;
            break;
          }
        }

        if (planeValid) {
          debug_num22 ++;
          double dist = pa * _pointSel.x +
                        pb * _pointSel.y +
                        pc * _pointSel.z + pd;
          Eigen::Vector3d omega(pa, pb, pc);
          Eigen::Vector3d point_proj = Eigen::Vector3d(_pointSel.x,_pointSel.y,_pointSel.z) - (dist * omega);
          Eigen::Vector3d e1(1, 0, 0);
          Eigen::Matrix3d J = e1 * omega.transpose();
          Eigen::JacobiSVD<Eigen::Matrix3d> svd(J, Eigen::ComputeThinU | Eigen::ComputeThinV);
          Eigen::Matrix3d R_svd = svd.matrixV() * svd.matrixU().transpose();
          Eigen::Matrix3d info = (1.0/IMUIntegrator::lidar_m) * Eigen::Matrix3d::Identity();
          info(1, 1) *= plan_weight_tan;
          info(2, 2) *= plan_weight_tan;
          Eigen::Matrix3d sqrt_info = info * R_svd.transpose();
  
          auto* e = Cost_NavState_IMU_Plan_Vec::Create(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                                        point_proj,
                                                        Tbl,
                                                        sqrt_info);
          edges.push_back(e);
          vPlanFeatures.emplace_back(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                      point_proj,
                                      sqrt_info);
          vPlanFeatures.back().ComputeError(m4d);
        }
      }
    }
  }
}

/** @brief 求待匹配不规则特征点云中每个点到Map的距离，为每个点构造一个代价函数
  * @param [out] edges 构造好的代价函数
  * @param [out] vNonFeatures 不规则特征容器
  * @param [in] laserCloudNonFeature 不规则特征点云，即待匹配点云
  * @param [in] laserCloudNonFeatureLocal 本地Map
  * @param [in] kdtreeLocal 用本地Map建立的KDtree
  * @param [in] exTlb Lidar与IMU之间的外参矩阵
  * @param [in] m4d 待匹配点云的估计位姿
  */
void Estimator::processNonFeatureICP(std::vector<ceres::CostFunction *>& edges,
                                     std::vector<FeatureNon>& vNonFeatures,
                                     const pcl::PointCloud<PointType>::Ptr& laserCloudNonFeature,
                                     const pcl::PointCloud<PointType>::Ptr& laserCloudNonFeatureLocal,
                                     const pcl::KdTreeFLANN<PointType>::Ptr& kdtreeLocal,
                                     const Eigen::Matrix4d& exTlb,
                                     const Eigen::Matrix4d& m4d){
  Eigen::Matrix4d Tbl = Eigen::Matrix4d::Identity();
  Tbl.topLeftCorner(3,3) = exTlb.topLeftCorner(3,3).transpose();
  Tbl.topRightCorner(3,1) = -1.0 * Tbl.topLeftCorner(3,3) * exTlb.topRightCorner(3,1);

  /* 如果vNonFeatures不空，直接从中取得p、a、b、c、d五个点构造代价函数 */
  /* 免去求最近5点，求平面等的过程 */
  if(!vNonFeatures.empty()){
    for(const auto& p : vNonFeatures){
      auto* e = Cost_NonFeature_ICP::Create(p.pointOri,
                                            p.pa,
                                            p.pb,
                                            p.pc,
                                            p.pd,
                                            Tbl,
                                            Eigen::Matrix<double, 1, 1>(1/IMUIntegrator::lidar_m));
      edges.push_back(e);
    }
    return;
  }

  PointType _pointOri, _pointSel, _coeff;
  std::vector<int> _pointSearchInd;
  std::vector<float> _pointSearchSqDis;
  std::vector<int> _pointSearchInd2;
  std::vector<float> _pointSearchSqDis2;

  Eigen::Matrix< double, 5, 3 > _matA0;
  _matA0.setZero();
  Eigen::Matrix< double, 5, 1 > _matB0;
  _matB0.setOnes();
  _matB0 *= -1;
  Eigen::Matrix< double, 3, 1 > _matX0;
  _matX0.setZero();

  int laserCloudNonFeatureStackNum = laserCloudNonFeature->points.size();
  for (int i = 0; i < laserCloudNonFeatureStackNum; i++) {
    /* 从待匹配点云中提取一点，变换到Map坐标系 */
    /* 注意：这里为了速度快，没有用到外参矩阵 */
    _pointOri = laserCloudNonFeature->points[i];
    MAP_MANAGER::pointAssociateToMap(&_pointOri, &_pointSel, m4d);
    /* 找到与该点对应的MapID */
    int id = map_manager->FindUsedNonFeatureMap(&_pointSel,laserCenWidth_last,laserCenHeight_last,laserCenDepth_last);

    if(id == 5000) continue;

    if(std::isnan(_pointSel.x) || std::isnan(_pointSel.y) ||std::isnan(_pointSel.z)) continue;

    /* 基于当前点到全局Map的距离构造残差，构造代价函数 */
    /* 如果对应的Map规模大于100个点，则在Map中查找5个最近点，求5个点所在平面
       p点到平面的距离就是残差*/
    if(GlobalNonFeatureMap[id].points.size() > 100) {
      NonFeatureKdMap[id].nearestKSearch(_pointSel, 5, _pointSearchInd, _pointSearchSqDis);
      if (_pointSearchSqDis[4] < 1 * thres_dist) {

        /* 求解三元一次方程组Ax=b，进行平面的法向量估计
         * 三元一次方程Ax+By+Cz+D=0对应于空间平面，向量n=(A,B,C)是其法向量，
         * 这里已知五个点在同一平面，且设定D为1，则一定可以找到唯一的一组法
         * 向量n=(A,B,C)与该平面对应，调用A.colPivHouseholderQr().solve(b)
         * 求解三元一次方程组，获得法向量n=(A,B,C)。*/
        for (int j = 0; j < 5; j++) {
          _matA0(j, 0) = GlobalNonFeatureMap[id].points[_pointSearchInd[j]].x;
          _matA0(j, 1) = GlobalNonFeatureMap[id].points[_pointSearchInd[j]].y;
          _matA0(j, 2) = GlobalNonFeatureMap[id].points[_pointSearchInd[j]].z;
        }
        _matX0 = _matA0.colPivHouseholderQr().solve(_matB0);

        /* 下面用法向量检查这五个点是否构成一个严格的平面
         * 在平面上的点p(x,y,z)一定满足方程Ax+By+Cz+1=0,如果法向量(A,B,C)的范数
         * 是n，则方程的两边同时除以n，等式仍然成立：(A/n)x+(B/n)y+(C/n)z+1/n=0。
         * 不满足这个等式的点不在该平面上。*/    
        float pa = _matX0(0, 0);
        float pb = _matX0(1, 0);
        float pc = _matX0(2, 0);
        float pd = 1;

        float ps = std::sqrt(pa * pa + pb * pb + pc * pc);
        pa /= ps;
        pb /= ps;
        pc /= ps;
        pd /= ps;

        /* 计算等式(A/n)x+(B/n)y+(C/n)z+1/n的值，并检查是否≤0.2 */
        bool planeValid = true;
        for (int j = 0; j < 5; j++) {
          if (std::fabs(pa * GlobalNonFeatureMap[id].points[_pointSearchInd[j]].x +
                        pb * GlobalNonFeatureMap[id].points[_pointSearchInd[j]].y +
                        pc * GlobalNonFeatureMap[id].points[_pointSearchInd[j]].z + pd) > 0.2) {
            planeValid = false;
            break;
          }
        }

        if(planeValid) {

          /* 构造不规则特征点到Map的代价函数 */
          /* 点p到平面的距离即是残差 */
          /* 注意下面用于构造代价函数的p点用的是原始点，而不是转换到Map坐标系的点_pointSel */
          /* 因为_pointSel不包含IMU到Lidar的外参变换，精度不够 */
          auto* e = Cost_NonFeature_ICP::Create(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                                pa,
                                                pb,
                                                pc,
                                                pd,
                                                Tbl,
                                                Eigen::Matrix<double, 1, 1>(1/IMUIntegrator::lidar_m));
          edges.push_back(e);
          vNonFeatures.emplace_back(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                    pa,
                                    pb,
                                    pc,
                                    pd);
          vNonFeatures.back().ComputeError(m4d);

          continue;
        }
      }
    }

    /* 基于当前点到本地Map的距离构造残差，构造代价函数 */
    /* 如果对应的Map规模大于20个点，则在Map中查找5个最近点，求5个点所在平面
       p点到平面的距离就是残差*/
    if(laserCloudNonFeatureLocal->points.size() > 20 ){
      kdtreeLocal->nearestKSearch(_pointSel, 5, _pointSearchInd2, _pointSearchSqDis2);
      if (_pointSearchSqDis2[4] < 1 * thres_dist) {
        for (int j = 0; j < 5; j++) { 
          _matA0(j, 0) = laserCloudNonFeatureLocal->points[_pointSearchInd2[j]].x;
          _matA0(j, 1) = laserCloudNonFeatureLocal->points[_pointSearchInd2[j]].y;
          _matA0(j, 2) = laserCloudNonFeatureLocal->points[_pointSearchInd2[j]].z;
        }
        _matX0 = _matA0.colPivHouseholderQr().solve(_matB0);

        float pa = _matX0(0, 0);
        float pb = _matX0(1, 0);
        float pc = _matX0(2, 0);
        float pd = 1;

        float ps = std::sqrt(pa * pa + pb * pb + pc * pc);
        pa /= ps;
        pb /= ps;
        pc /= ps;
        pd /= ps;

        bool planeValid = true;
        for (int j = 0; j < 5; j++) {
          if (std::fabs(pa * laserCloudNonFeatureLocal->points[_pointSearchInd2[j]].x +
                        pb * laserCloudNonFeatureLocal->points[_pointSearchInd2[j]].y +
                        pc * laserCloudNonFeatureLocal->points[_pointSearchInd2[j]].z + pd) > 0.2) {
            planeValid = false;
            break;
          }
        }

        if(planeValid) {

          auto* e = Cost_NonFeature_ICP::Create(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                                pa,
                                                pb,
                                                pc,
                                                pd,
                                                Tbl,
                                                Eigen::Matrix<double, 1, 1>(1/IMUIntegrator::lidar_m));
          edges.push_back(e);
          vNonFeatures.emplace_back(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                    pa,
                                    pb,
                                    pc,
                                    pd);
          vNonFeatures.back().ComputeError(m4d);
        }
      }
    }
  }
}

/** @brief 将lidarFrameList中的位姿转存到para_PR数组中
  *   通过对数映射将旋转四元数转成李代数
  *   函数中通过Eigen::Map模板类PR和VBias以矩阵的方式来访问C++数组para_PR和para_VBias
  * @param [inout] lidarFrameList 点云帧列表
  */
void Estimator::vector2double(const std::list<LidarFrame>& lidarFrameList){
  int i = 0;
  /* PR和VBias是Eigen::Map类型的模板类，达到以矩阵的方式访问C++数组的效果 */
  /* 最终的效果是将l.P和l.Q填到para_PR数组中，以矩阵的方式访问C++数组更便捷 */
  for(const auto& l : lidarFrameList){
    Eigen::Map<Eigen::Matrix<double, 6, 1>> PR(para_PR[i]);
    PR.segment<3>(0) = l.P;
    PR.segment<3>(3) = Sophus::SO3d(l.Q).log();

    Eigen::Map<Eigen::Matrix<double, 9, 1>> VBias(para_VBias[i]);
    VBias.segment<3>(0) = l.V;
    VBias.segment<3>(3) = l.bg;
    VBias.segment<3>(6) = l.ba;
    i++;
  }
}

/** @brief 将para_PR数组中的位姿转存到lidarFrameList中
  *   通过指数映射将李代数转成旋转四元数
  *   函数中通过Eigen::Map模板类PR和VBias以矩阵的方式来访问C++数组para_PR和para_VBias
  * @param [inout] lidarFrameList 点云帧列表
  */
void Estimator::double2vector(std::list<LidarFrame>& lidarFrameList){
  int i = 0;
  for(auto& l : lidarFrameList){
    Eigen::Map<const Eigen::Matrix<double, 6, 1>> PR(para_PR[i]);
    Eigen::Map<const Eigen::Matrix<double, 9, 1>> VBias(para_VBias[i]);
    l.P = PR.segment<3>(0);
    l.Q = Sophus::SO3d::exp(PR.segment<3>(3)).unit_quaternion();
    l.V = VBias.segment<3>(0);
    l.bg = VBias.segment<3>(3);
    l.ba = VBias.segment<3>(6);
    i++;
  }
}

/** @brief 位姿优化
  *   1、将三种特征点分开，并降采样
  *   2、进行点云匹配，获得优化后的位姿
  *   3、将匹配后的特征点云添加到Map
  * @param [in] lidarFrameList: 点云帧列表
  * @param [in] exTlb: 从Lidar坐标系到IMU坐标系的外参
  * @param [in] gravity: 重力加速度向量
  * @param [in] debugInfo: 
  */
void Estimator::EstimateLidarPose(std::list<LidarFrame>& lidarFrameList,
                           const Eigen::Matrix4d& exTlb,
                           const Eigen::Vector3d& gravity,
                           nav_msgs::Odometry& debugInfo){
  
  /* 获得从IMU到Lidar坐标系的旋转和平移外参 */
  Eigen::Matrix3d exRbl = exTlb.topLeftCorner(3,3).transpose();
  Eigen::Vector3d exPbl = -1.0 * exRbl * exTlb.topRightCorner(3,1);
  
  /* 从点云帧列表中取得最新的待优化帧的位姿 */
  /* FIXME:取得待优化的位姿之后并没有使用？ */
  Eigen::Matrix4d transformTobeMapped = Eigen::Matrix4d::Identity();
  transformTobeMapped.topLeftCorner(3,3) = lidarFrameList.back().Q * exRbl;
  transformTobeMapped.topRightCorner(3,1) = lidarFrameList.back().Q * exPbl + lidarFrameList.back().P;

  /* 取得Map中角点特征点和平面特征点的数量 */
  int laserCloudCornerFromMapNum = map_manager->get_corner_map()->points.size();
  int laserCloudSurfFromMapNum = map_manager->get_surf_map()->points.size();
  /* 取得本地Map中角点特征点和平面特征点的数量 */
  int laserCloudCornerFromLocalNum = laserCloudCornerFromLocal->points.size();
  int laserCloudSurfFromLocalNum = laserCloudSurfFromLocal->points.size();

  /* 准备待匹配特征点云，分成三类特征后存放到Stack容器中，不同的帧放在不同的层中 */
  /* 当IMU_Mode=1时，lidarFrameList的长度＝1，Stack容器深度=1 */
  /* 当IMU_Mode=2时，lidarFrameList的长度=20，Stack容器深度>1 */
  /* FIXME:Stack容器的最大深度只有2，但是lidarFrameList的最大长度似乎不止*/
  int stack_count = 0; //Stack容器深度
  for(const auto& l : lidarFrameList){
    /* 将点云中的角点特征点挪到laserCloudCornerLast中 */
    laserCloudCornerLast[stack_count]->clear();
    for(const auto& p : l.laserCloud->points){
      if(std::fabs(p.normal_z - 1.0) < 1e-5)
        laserCloudCornerLast[stack_count]->push_back(p);
    }
    /* 将点云中的平面特征点挪到laserCloudSurfLast中 */
    laserCloudSurfLast[stack_count]->clear();
    for(const auto& p : l.laserCloud->points){
      if(std::fabs(p.normal_z - 2.0) < 1e-5)
        laserCloudSurfLast[stack_count]->push_back(p);
    }
    /* 将点云中的不规则特征点挪到laserCloudNonFeatureLast中 */
    laserCloudNonFeatureLast[stack_count]->clear();
    for(const auto& p : l.laserCloud->points){
      if(std::fabs(p.normal_z - 3.0) < 1e-5)
        laserCloudNonFeatureLast[stack_count]->push_back(p);
    }
    /* 对角点特征点进行降采样 */
    laserCloudCornerStack[stack_count]->clear();
    downSizeFilterCorner.setInputCloud(laserCloudCornerLast[stack_count]);
    downSizeFilterCorner.filter(*laserCloudCornerStack[stack_count]);
    /* 对平面特征点进行降采样 */
    laserCloudSurfStack[stack_count]->clear();
    downSizeFilterSurf.setInputCloud(laserCloudSurfLast[stack_count]);
    downSizeFilterSurf.filter(*laserCloudSurfStack[stack_count]);
    /* 对不规则特征点进行降采样 */
    laserCloudNonFeatureStack[stack_count]->clear();
    downSizeFilterNonFeature.setInputCloud(laserCloudNonFeatureLast[stack_count]);
    downSizeFilterNonFeature.filter(*laserCloudNonFeatureStack[stack_count]);
    stack_count++;
  }
  
  /* 进行位姿优化，即点云匹配 */
  /* 匹配前确保Map以及本地Map中的特征点数量＞0 */
  if ( ((laserCloudCornerFromMapNum > 0 && laserCloudSurfFromMapNum > 100) || 
       (laserCloudCornerFromLocalNum > 0 && laserCloudSurfFromLocalNum > 100))) {
    Estimate(lidarFrameList, exTlb, gravity);
  }

  /* 取得优化后的位姿变换 */
  transformTobeMapped = Eigen::Matrix4d::Identity();
  transformTobeMapped.topLeftCorner(3,3) = lidarFrameList.front().Q * exRbl;
  transformTobeMapped.topRightCorner(3,1) = lidarFrameList.front().Q * exPbl + lidarFrameList.front().P;

  /* 将完成匹配的特征点云分别添加到Map和本地Map */
  
  /* 将降采样后的特征点云添加到ForMap容器，threadMapIncrement线程会从该容器取出点云叠加到Map中 */
  std::unique_lock<std::mutex> locker(mtx_Map);
  *laserCloudCornerForMap = *laserCloudCornerStack[0];
  *laserCloudSurfForMap = *laserCloudSurfStack[0];
  *laserCloudNonFeatureForMap = *laserCloudNonFeatureStack[0];
  /* threadMapIncrement线程用transformForMap完成点云的变换后叠加到Map */
  transformForMap = transformTobeMapped;
  
  /* 清空FromLocal容器，准备更新本地Map */
  /* FromLocal容器存放了最近的若干帧完成坐标变换、叠加和降采样的特征点云，相当于本地地图 */
  /* 与Map的匹配就是和FromLocal中的特征点云匹配 */
  laserCloudCornerFromLocal->clear();
  laserCloudSurfFromLocal->clear();
  laserCloudNonFeatureFromLocal->clear();
  /* 生成用于点云匹配的本地Map，存放在FromLocal容器中 */
  /* 即将最近的若干帧特征点云变换到Map坐标系，叠加在一起，降采样后存放在FromLocal容器中 */
  MapIncrementLocal(laserCloudCornerForMap,laserCloudSurfForMap,laserCloudNonFeatureForMap,transformTobeMapped);
  locker.unlock();
}

/** @brief 位姿优化
  *   每次优化使用两帧点云，分别是i和j，i是上一帧，j是当前帧
  *   优化的状态变量有六个：Pi，Vi，Ri，Pj，Vj，Rj，δbg，δba
  *   首先对i和j之间的IMU数据进行预积，获得第j帧的位姿估计值，同时获得i、j之间的预积分
  *   将预积分代价函数添加到优化问题，以实现对偏差δbg，δba的优化
  *   将上一轮的边缘优化添加到优化问题，以获得该帧更高精度的位姿
  *   将第i和第j帧特征点云到Map的Ceres代价函数添加到优化问题，以实现对位姿Pi，Vi，Ri，Pj，Vj，Rj的优化
  *     -角点特征点使用点p到直线ab的距离作为残差
  *     -平面特征点使用点p到平面投影点的距离作为残差
  *     -不规则特征点使用点p到平面的距离作为残差
  *     -在残差的后处理上使用了信息矩阵，具体的原理还不是很清楚
  *   优化完成后将第i帧的状态变量以及代价函数添加到边缘优化，参与下一轮的优化
  * @param [in] lidarFrameList: 点云帧列表
  * @param [in] exTlb: 从Lidar坐标系到IMU坐标系的外参
  * @param [in] gravity: 重力加速度向量
  */
void Estimator::Estimate(std::list<LidarFrame>& lidarFrameList,
                         const Eigen::Matrix4d& exTlb,
                         const Eigen::Vector3d& gravity){

  int num_corner_map = 0;
  int num_surf_map = 0;

  static uint32_t frame_count = 0;
  /* lidarFrameList中的帧数就是窗口的Size，在IMU紧耦合模式下等于2 */
  /* 这里应用了IMU预积分算法，IMU预积分的残差每次需要用到两帧点云对应位姿，分别称作第i和第j帧 */
  int windowSize = lidarFrameList.size();
  Eigen::Matrix4d transformTobeMapped = Eigen::Matrix4d::Identity();
  /* 获得从IMU到Lidar坐标系的旋转和平移外参 */
  Eigen::Matrix3d exRbl = exTlb.topLeftCorner(3,3).transpose();
  Eigen::Vector3d exPbl = -1.0 * exRbl * exTlb.topRightCorner(3,1);

  /* 建立本地Map的KDtree */
  kdtreeCornerFromLocal->setInputCloud(laserCloudCornerFromLocal);
  kdtreeSurfFromLocal->setInputCloud(laserCloudSurfFromLocal);
  kdtreeNonFeatureFromLocal->setInputCloud(laserCloudNonFeatureFromLocal);

  /* ??? */
  std::unique_lock<std::mutex> locker3(map_manager->mtx_MapManager);
  for(int i = 0; i < 4851; i++){
    CornerKdMap[i] = map_manager->getCornerKdMap(i);
    SurfKdMap[i] = map_manager->getSurfKdMap(i);
    NonFeatureKdMap[i] = map_manager->getNonFeatureKdMap(i);

    GlobalSurfMap[i] = map_manager->laserCloudSurf_for_match[i];
    GlobalCornerMap[i] = map_manager->laserCloudCorner_for_match[i];
    GlobalNonFeatureMap[i] = map_manager->laserCloudNonFeature_for_match[i];
  }
  laserCenWidth_last = map_manager->get_laserCloudCenWidth_last();
  laserCenHeight_last = map_manager->get_laserCloudCenHeight_last();
  laserCenDepth_last = map_manager->get_laserCloudCenDepth_last();

  locker3.unlock();

  // store point to line features
  std::vector<std::vector<FeatureLine>> vLineFeatures(windowSize);
  for(auto& v : vLineFeatures){
    v.reserve(2000);
  }

  // store point to plan features
  std::vector<std::vector<FeaturePlanVec>> vPlanFeatures(windowSize);
  for(auto& v : vPlanFeatures){
    v.reserve(2000);
  }

  std::vector<std::vector<FeatureNon>> vNonFeatures(windowSize);
  for(auto& v : vNonFeatures){
    v.reserve(2000);
  }

  if(windowSize == SLIDEWINDOWSIZE) {
    plan_weight_tan = 0.0003;
    thres_dist = 1.0;
  } else {
    plan_weight_tan = 0.0;
    thres_dist = 25.0;
  }

  /**
   * @brief 开始迭代优化，最多迭代5次，每迭代一次iterOpt加一
   */
  // excute optimize process
  const int max_iters = 5;
  for(int iterOpt=0; iterOpt<max_iters; ++iterOpt){

    /**
     * @brief 优化第一步：准备待优化的状态变量para_PR和para_VBias
     * 待优化的状态变量保存在para_PR和para_VBias数组中
     * 将lidarFrameList中的位姿转存到para_PR和para_VBias数组中，作为初始值
     * para_PR中保存位置和姿态，其中姿态以李代数的形式保存
     * para_VBias中保存的是速度和偏差
     * 转存为C++数组的原因是ceres库只接受数组形式的参数
     */
    vector2double(lidarFrameList);

    /* 创建huber损失函数 */
    /** FIXME:为什么windowSize等于2的时候要清空损失函数，而且还存在内存泄露的嫌疑 */
    //create huber loss function
    /* 创建鲁棒核函数，防止个别错误数据把整个优化方向带偏 */
    ceres::LossFunction* loss_function = NULL;
    loss_function = new ceres::HuberLoss(0.1 / IMUIntegrator::lidar_m);
    if(windowSize == SLIDEWINDOWSIZE) {
      loss_function = NULL;
    } else {
      loss_function = new ceres::HuberLoss(0.1 / IMUIntegrator::lidar_m);
    }

    /* 定义待优化的问题 */
    ceres::Problem::Options problem_options;
    ceres::Problem problem(problem_options);

    /**
     * @brief 优化第二步：将IMU预积分相关的残差添加到优化问题
     * 注意这里只添加第j帧一帧的IMU预积分代价函数到优化问题。
     * IMU预积分是从i到j之间所有数据的预积分，因此只需要添加一帧即可。
     * IMU预积分的结果就是在这里发挥作用。
     */

    /* 将位置和姿态以参数块的方式添加到优化问题 */
    for(int i=0; i<windowSize; ++i) {
      problem.AddParameterBlock(para_PR[i], 6);
    }

    /* 将速度和偏差以参数块的方式添加到优化问题 */
    for(int i=0; i<windowSize; ++i)
      problem.AddParameterBlock(para_VBias[i], 9);

    // add IMU CostFunction
    for(int f=1; f<windowSize; ++f){
      /* 取得指向第一帧的迭代器 */
      auto frame_curr = lidarFrameList.begin();
      /* 设置迭代指针指向第2帧，即IMU预积分中的第j帧 */
      std::advance(frame_curr, f);
      /* Eigen::LLT表示对矩阵进行Cholesky分解，然后通过matrixL()方法获得分解后的下三角矩阵L */
      /* 这里是对IMU预积分测量噪声的协方差矩阵的逆矩阵进行Cholesky分解，然后获得L矩阵的转置 */
      /* Cholesky分解本质上是对矩阵进行开方，下三角矩阵L即是原矩阵的平方根 */
      /* 也就是将IMU预积分测量噪声协方差矩阵的平方根传入代价函数 */
      problem.AddResidualBlock(Cost_NavState_PRV_Bias::Create(frame_curr->imuIntegrator,
                                                              const_cast<Eigen::Vector3d&>(gravity),
                                                              Eigen::LLT<Eigen::Matrix<double, 15, 15>>
                                                                      (frame_curr->imuIntegrator.GetCovariance().inverse())
                                                                      .matrixL().transpose()),
                               nullptr,         //损失函数为空
                               para_PR[f-1],    //参数pri_
                               para_VBias[f-1], //参数velobiasi_
                               para_PR[f],      //参数prj_
                               para_VBias[f]);  //参数velobiasj_
    }

    /**
     * @brief 优化第三步：防止某节点（点云）的当前优化结果与上一周期的优化结果脱节
     * 
     * 由于使用了滑动窗口的机制，每一帧点云都要参与多个周期的优化，具体的优化次数取决于滑动窗口的长度，
     * 当滑动窗口的长度为2时，每帧点云参与优化的次数就是2。
     * 
     * 为了防止同一个节点（点云）状态（位姿）的多次优化结果相互脱节，需要设计一种代价函数（因子）把它们拉住，
     * MarginalizationFactor就是这个防止脱节的因子。
     * 
     * MarginalizationFactor是一个重载的代价函数，它将某个节点当前周期的优化结果与上一周期优化结果之差作
     * 为残差进行迭代优化。
     * 
     * last_marginalization_info中保存着上一周期第j帧参数块的优化结果，在创建因子的时候就作为参数传进去。
     * 
     * last_marginalization_parameter_blocks中是上一周期第j帧参数块在本周期即第i帧对应的参数块地址，即
     * para_PR[0]和para_VBias[0]的地址，存放着本周期的优化结果，并且在持续迭代更新中
     */
    if (last_marginalization_info){
      // construct new marginlization_factor
      auto *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
      problem.AddResidualBlock(marginalization_factor, nullptr,
                               last_marginalization_parameter_blocks);
    }

    /* 取得第j帧的姿态q和位置t */
    Eigen::Quaterniond q_before_opti = lidarFrameList.back().Q;
    Eigen::Vector3d t_before_opti = lidarFrameList.back().P;

    /**
     * @brief 优化第四步：以当前扫描点云到Map的距离为优化目标构造代价函数，添加到优化问题。这部分
     * 内容与livox_horizon_loam中构造代价函数的方法几乎完全一致，唯一的区别是增加了不规则特征点对
     * 应的代价函数，在这个过程中并未使用IMU预积分的结果。
     *   以角点特征点为例，构造的方法是为当前扫描点云中的每一个点p在map中找到最近的点a和b，然后
     * 以点p、a和b为已知数据，待优化位姿para_PR为未知数据，构造一个代价函数。优化过程中Ceres将找到
     * 最优的位姿para_PR，对点p进行变换，使得p到直线ab的距离最小。
     *   由于点的数量众多，需要逐一构造代价函数，因此创建三个线程来提高并行度，创建好的代价函数存放在
     * edgesLine、edgesPlan、edgesNon中，然后再连同待优化位姿para_PR添加到优化问题。
     */
    
    /* 定义存放代价函数的的容器，几乎每个点都有一个代价函数 */
    std::vector<std::vector<ceres::CostFunction *>> edgesLine(windowSize);
    std::vector<std::vector<ceres::CostFunction *>> edgesPlan(windowSize);
    std::vector<std::vector<ceres::CostFunction *>> edgesNon(windowSize);
    
    /* 为点云列表lidarFrameList中的每一帧点云构造代价函数 */
    std::thread threads[3];
    for(int f=0; f<windowSize; ++f) {
      /* 取得指向点云列表第一个元素的迭代器 */
      auto frame_curr = lidarFrameList.begin();
      /* 取得点云列表中的第f个元素 */
      /* advance是迭代器辅助函数，配合for循环中的f变量自增，实现对点云列表的遍历 */
      std::advance(frame_curr, f);
      /* 将第f帧点云的姿态Q和位置P从IMU坐标系转成Lidar坐标系 */
      transformTobeMapped = Eigen::Matrix4d::Identity();
      transformTobeMapped.topLeftCorner(3,3) = frame_curr->Q * exRbl;
      transformTobeMapped.topRightCorner(3,1) = frame_curr->Q * exPbl + frame_curr->P;

      /* 启动独立线程，构造角点特征点云与Map之间的Ceres代价函数 */
      threads[0] = std::thread(&Estimator::processPointToLine, this,
                               std::ref(edgesLine[f]),
                               std::ref(vLineFeatures[f]),
                               std::ref(laserCloudCornerStack[f]),
                               std::ref(laserCloudCornerFromLocal),
                               std::ref(kdtreeCornerFromLocal),
                               std::ref(exTlb),
                               std::ref(transformTobeMapped));

      /* 启动独立线程，构造平面特征点云与Map之间的Ceres代价函数 */
      threads[1] = std::thread(&Estimator::processPointToPlanVec, this,
                               std::ref(edgesPlan[f]),
                               std::ref(vPlanFeatures[f]),
                               std::ref(laserCloudSurfStack[f]),
                               std::ref(laserCloudSurfFromLocal),
                               std::ref(kdtreeSurfFromLocal),
                               std::ref(exTlb),
                               std::ref(transformTobeMapped));

      /* 启动独立线程，构造不规则特征点云与Map之间的Ceres代价函数 */
      threads[2] = std::thread(&Estimator::processNonFeatureICP, this,
                               std::ref(edgesNon[f]),
                               std::ref(vNonFeatures[f]),
                               std::ref(laserCloudNonFeatureStack[f]),
                               std::ref(laserCloudNonFeatureFromLocal),
                               std::ref(kdtreeNonFeatureFromLocal),
                               std::ref(exTlb),
                               std::ref(transformTobeMapped));
      /* 等待构造完成 */
      threads[0].join();
      threads[1].join();
      threads[2].join();
    }

    /* 将构造好的代价函数添加到优化问题 */
    int cntSurf = 0;
    int cntCorner = 0;
    int cntNon = 0;
    /* 如果windowSize=2，则使用1.0的阈值 */
    /* FIXME: 修改阈值似乎并没有任何意义，多此一举 */
    if(windowSize == SLIDEWINDOWSIZE) {
      thres_dist = 1.0;
      /* 如果是第一次迭代，则对代价函数的有效性进行检查，仅有残差大于0.00001的才是有效的 */
      if(iterOpt == 0){
        for(int f=0; f<windowSize; ++f){
          int cntFtu = 0;
          for (auto &e : edgesLine[f]) {
            if(std::fabs(vLineFeatures[f][cntFtu].error) > 1e-5){
              /* 将代价函数e和待优化位姿para_PR添加到优化问题 */
              problem.AddResidualBlock(e, loss_function, para_PR[f]);
              vLineFeatures[f][cntFtu].valid = true;
            }else{
              vLineFeatures[f][cntFtu].valid = false;
            }
            cntFtu++;
            cntCorner++;
          }

          cntFtu = 0;
          for (auto &e : edgesPlan[f]) {
            if(std::fabs(vPlanFeatures[f][cntFtu].error) > 1e-5){
              problem.AddResidualBlock(e, loss_function, para_PR[f]);
              vPlanFeatures[f][cntFtu].valid = true;
            }else{
              vPlanFeatures[f][cntFtu].valid = false;
            }
            cntFtu++;
            cntSurf++;
          }

          cntFtu = 0;
          for (auto &e : edgesNon[f]) {
            if(std::fabs(vNonFeatures[f][cntFtu].error) > 1e-5){
              problem.AddResidualBlock(e, loss_function, para_PR[f]);
              vNonFeatures[f][cntFtu].valid = true;
            }else{
              vNonFeatures[f][cntFtu].valid = false;
            }
            cntFtu++;
            cntNon++;
          }
        }
      }else{ /* 如果不是第一次迭代，则直接使用使用第一次迭代的结果，不再检查代价函数的有效性 */
        for(int f=0; f<windowSize; ++f){
          int cntFtu = 0;
          for (auto &e : edgesLine[f]) {
            if(vLineFeatures[f][cntFtu].valid) {
              problem.AddResidualBlock(e, loss_function, para_PR[f]);
            }
            cntFtu++;
            cntCorner++;
          }
          cntFtu = 0;
          for (auto &e : edgesPlan[f]) {
            if(vPlanFeatures[f][cntFtu].valid){
              problem.AddResidualBlock(e, loss_function, para_PR[f]);
            }
            cntFtu++;
            cntSurf++;
          }

          cntFtu = 0;
          for (auto &e : edgesNon[f]) {
            if(vNonFeatures[f][cntFtu].valid){
              problem.AddResidualBlock(e, loss_function, para_PR[f]);
            }
            cntFtu++;
            cntNon++;
          }
        }
      }
    } else { /* 如果windowSize=2，则使用1.0的阈值 */
        /* FIXME: 修改阈值似乎并没有任何意义，下面整段代码都是多此一举 */
        if(iterOpt == 0) {
          thres_dist = 10.0;
        } else {
          thres_dist = 1.0;
        }
        for(int f=0; f<windowSize; ++f){
          int cntFtu = 0;
          for (auto &e : edgesLine[f]) {
            if(std::fabs(vLineFeatures[f][cntFtu].error) > 1e-5){
              problem.AddResidualBlock(e, loss_function, para_PR[f]);
              vLineFeatures[f][cntFtu].valid = true;
            }else{
              vLineFeatures[f][cntFtu].valid = false;
            }
            cntFtu++;
            cntCorner++;
          }
          cntFtu = 0;
          for (auto &e : edgesPlan[f]) {
            if(std::fabs(vPlanFeatures[f][cntFtu].error) > 1e-5){
              problem.AddResidualBlock(e, loss_function, para_PR[f]);
              vPlanFeatures[f][cntFtu].valid = true;
            }else{
              vPlanFeatures[f][cntFtu].valid = false;
            }
            cntFtu++;
            cntSurf++;
          }

          cntFtu = 0;
          for (auto &e : edgesNon[f]) {
            if(std::fabs(vNonFeatures[f][cntFtu].error) > 1e-5){
              problem.AddResidualBlock(e, loss_function, para_PR[f]);
              vNonFeatures[f][cntFtu].valid = true;
            }else{
              vNonFeatures[f][cntFtu].valid = false;
            }
            cntFtu++;
            cntNon++;
          }
        }
    }

    /**
     * @brief 优化第五步：开始优化
     */

    /* 开始优化 */
    ceres::Solver::Options options;
    /* 采用类似BA问题那样的Schur消元法求解线性方程组 */
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 10;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 6;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    /* 将完成优化的状态变量值更新到lidarFrameList中 */
    /* 待优化的状态变量保存在para_PR和para_VBias数组中 */
    /* 将para_PR和para_VBias数组中的位姿转存到lidarFrameList中 */
    double2vector(lidarFrameList);

    /* 取得优化后的位姿，获得优化前后的位姿增量 */
    Eigen::Quaterniond q_after_opti = lidarFrameList.back().Q;
    Eigen::Vector3d t_after_opti = lidarFrameList.back().P;
    Eigen::Vector3d V_after_opti = lidarFrameList.back().V;
    double deltaR = (q_before_opti.angularDistance(q_after_opti)) * 180.0 / M_PI;
    double deltaT = (t_before_opti - t_after_opti).norm();

    /**
     * @brief 优化第六步：开展边缘化
     * 边缘化的内容是全新的内容，总共四个步骤，理解有一定的难度，尤其是第一步和第四步。
     * FIXME: 还需要进一步的深入研究
     */

    /* 位姿增量小于阈值或达到最大迭代次数，停止 */
    if (deltaR < 0.05 && deltaT < 0.05 || (iterOpt+1) == max_iters){
      ROS_INFO("Frame: %d\n",frame_count++);
      if(windowSize != SLIDEWINDOWSIZE) break;
      
      /**
       * @brief 开展边缘化。
       * 
       * 这里的边缘化和
       * 所谓边缘化是指，将此次优化后的状态变量，以及代价函数添加到下一次优化中继续优化，以获得更高的精度 */
      /* 也就是说，每次实际上是有多帧点云参与优化 */

      // apply marginalization
      /* 更新本轮的边缘信息，准备存储待优化的状态变量和代价函数 */
      auto *marginalization_info = new MarginalizationInfo();

      /**
       * @brief 边缘化第一步：将防止脱节的因子添加到边缘化，防止某节点（点云）的边缘化结果与本周期前面的优化结
       * 果脱节。同时将滑动窗口中最老的一个节点所对应的参数块标识为丢弃状态，阻止该节点的状态进入下一周期的优化。
       */
      if (last_marginalization_info){
        std::vector<int> drop_set;
        for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
        {
          /**
           * 将滑动窗口中最早的节点（即第i帧）所对应的参数块标识为丢弃状态，阻止该参数块进入下一周期的优化。
           * last_marginalization_parameter_blocks中是参与上一周期优化的节点在本周期对应参数块的地址，在
           * 滑动窗口长度等于2的情况下，其中只有1个节点即上周期第j帧本周期第i帧对应参数块的地址，即para_PR[0]
           * 和para_VBias[0]的地址，这两个参数块被被标识为丢弃，标识将不会参与下一周期的优化。
           */
          if (last_marginalization_parameter_blocks[i] == para_PR[0] ||
              last_marginalization_parameter_blocks[i] == para_VBias[0])
            drop_set.push_back(i);
            /* last_marginalization_parameter_blocks中与para_PR[0]和para_VBias[0]对应的参数块被丢弃 */
        }

        /**
         * @brief 防止某节点（点云）的当前边缘化结果与前面的优化结果脱节
         * 
         * 由于使用了滑动窗口的机制，每一帧点云都要参与多个周期的优化，具体的优化次数取决于滑动窗口的长度，
         * 当滑动窗口的长度为2时，每帧点云参与优化的次数就是2。
         * 
         * 之外，每周期除了正常优化之外还有边缘化，因此每一帧点云在一个周期内就要进行多次优化。
         * 
         * 为了防止同一个节点（点云）状态（位姿）的多次优化结果相互脱节，需要设计一种代价函数（因子）把它们拉住，
         * MarginalizationFactor就是这个防止脱节的因子。
         * 
         * MarginalizationFactor是一个重载的代价函数，它将某个节点当前周期的优化结果与上一周期优化结果之差作
         * 为残差进行迭代优化。
         * 
         * last_marginalization_info中保存着上一周期第j帧（滑动窗口长度为2）参数块的优化结果，在创建因子的时
         * 候就作为参数传进去。如果滑动窗口长度＞n，则其中应该保存n-1帧参数块的优化结果。
         * 
         * last_marginalization_parameter_blocks中是上一周期第j帧参数块在本周期即第i帧对应的参数块地址，即
         * para_PR[0]和para_VBias[0]的地址，存放着本周期的优化结果，并且在持续迭代更新中。如果滑动窗口长度＞n，
         * 则其中应该保存n-1帧参数块在本周期对应的参数块地址。
         */
        auto *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        auto *residual_block_info = new ResidualBlockInfo(marginalization_factor, nullptr,
                                                          last_marginalization_parameter_blocks,
                                                          drop_set);
        marginalization_info->addResidualBlockInfo(residual_block_info);
      }
      
      /**
       * @brief 边缘化第二步：将IMU预积分代价函数添加到边缘化，对滑动窗口中最老的两帧之间的IMU数据进行边缘化
       * 注意这里只添加第2帧的IMU预积分代价函数到优化问题，IMU预积分是从i到j之间所有数据的预积分，因此只需要
       * 添加一帧即可。下面取得指向点云帧的迭代器，并指向第2帧，即第j帧。
       */
      auto frame_curr = lidarFrameList.begin();
      std::advance(frame_curr, 1);

      /**
       * 添加IMU预积分的代价函数到边缘化，对滑动窗口中最老的两帧进行边缘化。
       * 这里添加的IMU预积分代价函数与前面添加的完全一致，主要的区别有两点：
       * 1. 只添加了滑动窗口的前两帧（0、1）到边缘化；
       * 2. 前面的代价函数采用ceres::Solve进行优化，而这里由marginalization_info->preMarginalize()和
       * marginalization_info->marginalize()负责边缘化。
       */
      ceres::CostFunction* IMU_Cost = Cost_NavState_PRV_Bias::Create(frame_curr->imuIntegrator,
                                                                     const_cast<Eigen::Vector3d&>(gravity),
                                                                     Eigen::LLT<Eigen::Matrix<double, 15, 15>>
                                                                             (frame_curr->imuIntegrator.GetCovariance().inverse())
                                                                             .matrixL().transpose());
      auto *residual_block_info = new ResidualBlockInfo(IMU_Cost, nullptr,
                                                        std::vector<double *>{para_PR[0], para_VBias[0], para_PR[1], para_VBias[1]},
                                                        std::vector<int>{0, 1}); //将para_PR[0], para_VBias[0]添加到drop_set
      marginalization_info->addResidualBlockInfo(residual_block_info);

      /**
       * @brief 边缘化第三步：将点云到map的代价函数添加到边缘化
       * 对点云列表lidarFrameList中的第一帧点云进行边缘化
       * lidarFrameList中的第一帧点云(即此次匹配的第i帧)在此次优化后即将被删除，边缘化的就是这一帧
       * lidarFrameList中的第二帧点云(即此次匹配的第j帧)则会被保留下来作为下次匹配的第i帧
       * 注意f=0，即只取滑动窗口中的第1帧进行边缘化
       */
      int f = 0;
      transformTobeMapped = Eigen::Matrix4d::Identity();
      transformTobeMapped.topLeftCorner(3,3) = frame_curr->Q * exRbl;
      transformTobeMapped.topRightCorner(3,1) = frame_curr->Q * exPbl + frame_curr->P;
      
      /* 分别求第0帧的角点、平面点、不规则特征点云到Map的代价函数 */
      /* 与前面不同的是，此时已经完成主体优化，状态变量的估计值已经变成了优化值 */
      edgesLine[f].clear();
      edgesPlan[f].clear();
      edgesNon[f].clear();
      threads[0] = std::thread(&Estimator::processPointToLine, this,
                               std::ref(edgesLine[f]),
                               std::ref(vLineFeatures[f]),
                               std::ref(laserCloudCornerStack[f]),
                               std::ref(laserCloudCornerFromLocal),
                               std::ref(kdtreeCornerFromLocal),
                               std::ref(exTlb),
                               std::ref(transformTobeMapped));

      threads[1] = std::thread(&Estimator::processPointToPlanVec, this,
                               std::ref(edgesPlan[f]),
                               std::ref(vPlanFeatures[f]),
                               std::ref(laserCloudSurfStack[f]),
                               std::ref(laserCloudSurfFromLocal),
                               std::ref(kdtreeSurfFromLocal),
                               std::ref(exTlb),
                               std::ref(transformTobeMapped));

      threads[2] = std::thread(&Estimator::processNonFeatureICP, this,
                               std::ref(edgesNon[f]),
                               std::ref(vNonFeatures[f]),
                               std::ref(laserCloudNonFeatureStack[f]),
                               std::ref(laserCloudNonFeatureFromLocal),
                               std::ref(kdtreeNonFeatureFromLocal),
                               std::ref(exTlb),
                               std::ref(transformTobeMapped));      
                      
      threads[0].join();
      threads[1].join();
      threads[2].join();

      /* 将状态变量及代价函数添加到边缘化 */
      int cntFtu = 0;
      for (auto &e : edgesLine[f]) {
        if(vLineFeatures[f][cntFtu].valid){
          auto *residual_block_info = new ResidualBlockInfo(e, nullptr,
                                                            std::vector<double *>{para_PR[0]},
                                                            std::vector<int>{0}); //将para_PR[0]添加到drop_set
          marginalization_info->addResidualBlockInfo(residual_block_info);
        }
        cntFtu++;
      }
      cntFtu = 0;
      for (auto &e : edgesPlan[f]) {
        if(vPlanFeatures[f][cntFtu].valid){
          auto *residual_block_info = new ResidualBlockInfo(e, nullptr,
                                                            std::vector<double *>{para_PR[0]},
                                                            std::vector<int>{0}); //将para_PR[0]添加到drop_set
          marginalization_info->addResidualBlockInfo(residual_block_info);
        }
        cntFtu++;
      }

      cntFtu = 0;
      for (auto &e : edgesNon[f]) {
        if(vNonFeatures[f][cntFtu].valid){
          auto *residual_block_info = new ResidualBlockInfo(e, nullptr,
                                                            std::vector<double *>{para_PR[0]},
                                                            std::vector<int>{0}); //将para_PR[0]添加到drop_set
          marginalization_info->addResidualBlockInfo(residual_block_info);
        }
        cntFtu++;
      }

      /**
       * @brief 边缘化第四步：开展预边缘化和边缘化：
       * 这里的边缘化和BA的边缘化似乎不一样，这里的边缘化仅仅是对滑动窗口中的第0帧进行优化，该帧在本周期边缘化之后就永远
       * 的退出优化序列，下一周期不再对其进行优化，不论是正常优化还是边缘化都仅限于滑动窗口中的帧。
       */
      marginalization_info->preMarginalize();
      marginalization_info->marginalize();

      /**
       * @brief 边缘化第五步：更新边缘变量，用于下一周期防止优化脱节，主要是下面两个：
       * last_marginalization_info保存本周期滑动窗口中除第0帧之外所有帧对应参数块的优化结果，如果滑动窗口长度是2，则
       * 保存第1帧（即第j帧）对应参数块的优化结果。
       * last_marginalization_parameter_blocks保存本周期第j帧参数块在下一周期即第i帧对应的参数块地址，即
       * para_PR[0]和para_VBias[0]的地址。如果滑动窗口长度是n，则应该保存n-1帧参数块在下一周期对应的参数块地址。
       */

      /** 计算本周期第1帧~第n-1帧（滑动窗口长度为n）参数块在下一周期对应参数块的地址，即第0帧~第n-2帧参数快的地址 */
      std::unordered_map<long, double *> addr_shift;
      for (int i = 1; i < SLIDEWINDOWSIZE; i++)
      {
        /* 将para_PR[i]的地址转成整形作为map的index，将para_PR[i - 1]的地址作为map的value */
        /* 即用第j帧的地址去索引第i帧的地址 */
        addr_shift[reinterpret_cast<long>(para_PR[i])] = para_PR[i - 1];
        addr_shift[reinterpret_cast<long>(para_VBias[i])] = para_VBias[i - 1];
      }
      
      /** 
       * getParameterBlocks返回本周期第1帧~第n-1帧参数块在下一周期对应的参数块地址，即第0帧~第n-2帧参数块的地址，
       * 并将这些帧参数块在本周期优化的结果保存在marginalization_info->keep_block中，用于下一周期构造防止优化脱
       * 节的代价函数或因子。
       */
      std::vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
      delete last_marginalization_info;
      last_marginalization_info = marginalization_info;
      /* 将第i帧参数块para_PR[0]和para_VBias[0]的地址添加到last_marginalization_parameter_blocks中 */
      last_marginalization_parameter_blocks = parameter_blocks;
      break;
    }

    if(windowSize != SLIDEWINDOWSIZE) {
      for(int f=0; f<windowSize; ++f){
        edgesLine[f].clear();
        edgesPlan[f].clear();
        edgesNon[f].clear();
        vLineFeatures[f].clear();
        vPlanFeatures[f].clear();
        vNonFeatures[f].clear();
      }
    }
  }

}

/** @brief 将特征点云变换到地图坐标系，添加到FromLocal容器
  *   本地Map每次都重新生成，只保存最近的50帧点云数据
  * @param [in] laserCloudCornerStack: 角点特征点云
  * @param [in] laserCloudSurfStack: 平面特征点云
  * @param [in] laserCloudNonFeatureStack: 不规则特征点云
  * @param [in] transformTobeMapped: 变换矩阵
  */
void Estimator::MapIncrementLocal(const pcl::PointCloud<PointType>::Ptr& laserCloudCornerStack,
                                  const pcl::PointCloud<PointType>::Ptr& laserCloudSurfStack,
                                  const pcl::PointCloud<PointType>::Ptr& laserCloudNonFeatureStack,
                                  const Eigen::Matrix4d& transformTobeMapped){
  int laserCloudCornerStackNum = laserCloudCornerStack->points.size();
  int laserCloudSurfStackNum = laserCloudSurfStack->points.size();
  int laserCloudNonFeatureStackNum = laserCloudNonFeatureStack->points.size();
  PointType pointSel;
  PointType pointSel2;
  size_t Id = localMapID % localMapWindowSize;
  
  /* 将特征点云变换到地图坐标系 */
  localCornerMap[Id]->clear();
  localSurfMap[Id]->clear();
  localNonFeatureMap[Id]->clear();
  for (int i = 0; i < laserCloudCornerStackNum; i++) {
    MAP_MANAGER::pointAssociateToMap(&laserCloudCornerStack->points[i], &pointSel, transformTobeMapped);
    localCornerMap[Id]->push_back(pointSel);
  }
  for (int i = 0; i < laserCloudSurfStackNum; i++) {
    MAP_MANAGER::pointAssociateToMap(&laserCloudSurfStack->points[i], &pointSel2, transformTobeMapped);
    localSurfMap[Id]->push_back(pointSel2);
  }
  for (int i = 0; i < laserCloudNonFeatureStackNum; i++) {
    MAP_MANAGER::pointAssociateToMap(&laserCloudNonFeatureStack->points[i], &pointSel2, transformTobeMapped);
    localNonFeatureMap[Id]->push_back(pointSel2);
  }
  /* 将变换到Map坐标系的特征点云叠加到FromLocal容器 */
  for (int i = 0; i < localMapWindowSize; i++) {
    *laserCloudCornerFromLocal += *localCornerMap[i];
    *laserCloudSurfFromLocal += *localSurfMap[i];
    *laserCloudNonFeatureFromLocal += *localNonFeatureMap[i];
  }
  /* 因为叠加了多帧，对FromLocal容器再次进行降采样 */
  pcl::PointCloud<PointType>::Ptr temp(new pcl::PointCloud<PointType>());
  downSizeFilterCorner.setInputCloud(laserCloudCornerFromLocal);
  downSizeFilterCorner.filter(*temp);
  laserCloudCornerFromLocal = temp;
  pcl::PointCloud<PointType>::Ptr temp2(new pcl::PointCloud<PointType>());
  downSizeFilterSurf.setInputCloud(laserCloudSurfFromLocal);
  downSizeFilterSurf.filter(*temp2);
  laserCloudSurfFromLocal = temp2;
  pcl::PointCloud<PointType>::Ptr temp3(new pcl::PointCloud<PointType>());
  downSizeFilterNonFeature.setInputCloud(laserCloudNonFeatureFromLocal);
  downSizeFilterNonFeature.filter(*temp3);
  laserCloudNonFeatureFromLocal = temp3;
  localMapID ++;
}