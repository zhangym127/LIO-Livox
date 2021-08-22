#include "LidarFeatureExtractor/LidarFeatureExtractor.h"

LidarFeatureExtractor::LidarFeatureExtractor(int n_scans,int NumCurvSize,float DistanceFaraway,int NumFlat,
                                             int PartNum,float FlatThreshold,float BreakCornerDis,float LidarNearestDis,float KdTreeCornerOutlierDis)
                                             :N_SCANS(n_scans),
                                              thNumCurvSize(NumCurvSize),
                                              thDistanceFaraway(DistanceFaraway),
                                              thNumFlat(NumFlat),
                                              thPartNum(PartNum),
                                              thFlatThreshold(FlatThreshold),
                                              thBreakCornerDis(BreakCornerDis),
                                              thLidarNearestDis(LidarNearestDis){
  vlines.resize(N_SCANS);
  for(auto & ptr : vlines){
    ptr.reset(new pcl::PointCloud<PointType>());
  }
  vcorner.resize(N_SCANS);
  vsurf.resize(N_SCANS);
}

bool LidarFeatureExtractor::plane_judge(const std::vector<PointType>& point_list,const int plane_threshold)
{
  int num = point_list.size();
  float cx = 0;
  float cy = 0;
  float cz = 0;
  for (int j = 0; j < num; j++) {
    cx += point_list[j].x;
    cy += point_list[j].y;
    cz += point_list[j].z;
  }
  cx /= num;
  cy /= num;
  cz /= num;
  //mean square error
  float a11 = 0;
  float a12 = 0;
  float a13 = 0;
  float a22 = 0;
  float a23 = 0;
  float a33 = 0;
  for (int j = 0; j < num; j++) {
    float ax = point_list[j].x - cx;
    float ay = point_list[j].y - cy;
    float az = point_list[j].z - cz;

    a11 += ax * ax;
    a12 += ax * ay;
    a13 += ax * az;
    a22 += ay * ay;
    a23 += ay * az;
    a33 += az * az;
  }
  a11 /= num;
  a12 /= num;
  a13 /= num;
  a22 /= num;
  a23 /= num;
  a33 /= num;

  Eigen::Matrix< double, 3, 3 > _matA1;
  _matA1.setZero();
  Eigen::Matrix< double, 3, 1 > _matD1;
  _matD1.setZero();
  Eigen::Matrix< double, 3, 3 > _matV1;
  _matV1.setZero();

  _matA1(0, 0) = a11;
  _matA1(0, 1) = a12;
  _matA1(0, 2) = a13;
  _matA1(1, 0) = a12;
  _matA1(1, 1) = a22;
  _matA1(1, 2) = a23;
  _matA1(2, 0) = a13;
  _matA1(2, 1) = a23;
  _matA1(2, 2) = a33;

  Eigen::JacobiSVD<Eigen::Matrix3d> svd(_matA1, Eigen::ComputeThinU | Eigen::ComputeThinV);
  _matD1 = svd.singularValues();
  _matV1 = svd.matrixU();
  if (_matD1(0, 0) < plane_threshold * _matD1(1, 0)) {
    return true;
  }
  else{
    return false;
  }
}

/**
 * @brief 对同一条扫描线上的点进行特征提取
 * @param cloud 输入参数，同一条扫描线上的点
 * @param pointsLessSharp 输出参数，角点特征点云
 * @param pointsLessFlat 输出参数，平面点特征点云
 */
void LidarFeatureExtractor::detectFeaturePoint(pcl::PointCloud<PointType>::Ptr& cloud,
                                                std::vector<int>& pointsLessSharp,
                                                std::vector<int>& pointsLessFlat){
  int CloudFeatureFlag[20000];
  float cloudCurvature[20000];
  float cloudDepth[20000];
  int cloudSortInd[20000];
  float cloudReflect[20000];
  int reflectSortInd[20000];
  int cloudAngle[20000];

  pcl::PointCloud<PointType>::Ptr& laserCloudIn = cloud;

  int cloudSize = laserCloudIn->points.size();

  /* 将输入点云腾挪到_laserCloud中 */
  PointType point;
  pcl::PointCloud<PointType>::Ptr _laserCloud(new pcl::PointCloud<PointType>());
  _laserCloud->reserve(cloudSize);

  for (int i = 0; i < cloudSize; i++) {
    point.x = laserCloudIn->points[i].x;
    point.y = laserCloudIn->points[i].y;
    point.z = laserCloudIn->points[i].z;
#ifdef UNDISTORT
    point.normal_x = laserCloudIn.points[i].normal_x;
#else
    point.normal_x = 1.0;
#endif
    point.intensity = laserCloudIn->points[i].intensity;

    if (!pcl_isfinite(point.x) ||
        !pcl_isfinite(point.y) ||
        !pcl_isfinite(point.z)) {
      continue;
    }

    _laserCloud->push_back(point);
    CloudFeatureFlag[i] = 0;
  }

  cloudSize = _laserCloud->size();

  int debugnum1 = 0;
  int debugnum2 = 0;
  int debugnum3 = 0;
  int debugnum4 = 0;
  int debugnum5 = 0;

  int count_num = 1;
  bool left_surf_flag = false;
  bool right_surf_flag = false;

  //---------------------------------------- surf feature extract ---------------------------------------------
  int scanStartInd = 5;
  int scanEndInd = cloudSize - 6;

  int thDistanceFaraway_fea = 0;

  for (int i = 5; i < cloudSize - 5; i ++ ) {

    float diffX = 0;
    float diffY = 0;
    float diffZ = 0;

    /* 求该点到坐标原点的距离 */
    float dis = sqrt(_laserCloud->points[i].x * _laserCloud->points[i].x +
                     _laserCloud->points[i].y * _laserCloud->points[i].y +
                     _laserCloud->points[i].z * _laserCloud->points[i].z);

    /* 取当前点的相邻两点L和N */
    Eigen::Vector3d pt_last(_laserCloud->points[i-1].x, _laserCloud->points[i-1].y, _laserCloud->points[i-1].z);
    Eigen::Vector3d pt_cur(_laserCloud->points[i].x, _laserCloud->points[i].y, _laserCloud->points[i].z);
    Eigen::Vector3d pt_next(_laserCloud->points[i+1].x, _laserCloud->points[i+1].y, _laserCloud->points[i+1].z);

    /* 设坐标原点是O，当前点是C，相邻两点是L和N，则分别求向量OC与CL的夹角α的余弦，向量OC与CN的夹角β的余弦 */
    double angle_last = (pt_last-pt_cur).dot(pt_cur) / ((pt_last-pt_cur).norm()*pt_cur.norm());
    double angle_next = (pt_next-pt_cur).dot(pt_cur) / ((pt_next-pt_cur).norm()*pt_cur.norm());
 
    /* 如果距离大于100米，或者夹角α和β都小于15°或大于165° */
	/* 夹角α和β都小于15°或大于165°说明相邻两点L和N的深度都比C更大 */
	/* 这里说的L和N比C更远指的是从坐标原点O算起L和N比C更远，或者说OL和ON比OC更长，下面注释中的远近都指的是这个意思 */
    if (dis > thDistanceFaraway || (fabs(angle_last) > 0.966 && fabs(angle_next) > 0.966)) {
      thNumCurvSize = 2;
    } else {
      thNumCurvSize = 3;
    }

    /* 夹角α和β都小于15°或大于165°，说明当前点C是角点 */
    if(fabs(angle_last) > 0.966 && fabs(angle_next) > 0.966) {
      cloudAngle[i] = 1;
    }

	/* 
	 * 求当前点的曲率： 
	 *  -取当前点与两侧相邻4个或6个点的距离差作为曲率，距离差越大，则曲率越大。
	 *  -如果相邻两点比当前点显著更远，则只需要取两侧5个点，否则需要7个点。
	 *  -如果当前点距离大于100米，则只需要取两侧5个点，否则取7个点。
	 *
	 * 求当前点的反射率差异：
	 *  -取当前点与两侧相邻4个或6个点反射率的总差值作为当前点的反射率特征
	 */
    float diffR = -2 * thNumCurvSize * _laserCloud->points[i].intensity;
    for (int j = 1; j <= thNumCurvSize; ++j) {
      diffX += _laserCloud->points[i - j].x + _laserCloud->points[i + j].x;
      diffY += _laserCloud->points[i - j].y + _laserCloud->points[i + j].y;
      diffZ += _laserCloud->points[i - j].z + _laserCloud->points[i + j].z;
      diffR += _laserCloud->points[i - j].intensity + _laserCloud->points[i + j].intensity;
    }
    diffX -= 2 * thNumCurvSize * _laserCloud->points[i].x;
    diffY -= 2 * thNumCurvSize * _laserCloud->points[i].y;
    diffZ -= 2 * thNumCurvSize * _laserCloud->points[i].z;

    cloudDepth[i] = dis;
    cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;// / (2 * thNumCurvSize * dis + 1e-3);
    cloudSortInd[i] = i;
    cloudReflect[i] = diffR;
    reflectSortInd[i] = i;

  }

  /* 将点云以150个点为限，划分成若干个区段，然后逐一处理每个区段 */
  /* 找出每个区段中的平面特征点 */
  for (int j = 0; j < thPartNum; j++) {
	/* sp指向点云起点，ep指向点云终点， */
    int sp = scanStartInd + (scanEndInd - scanStartInd) * j / thPartNum;
    int ep = scanStartInd + (scanEndInd - scanStartInd) * (j + 1) / thPartNum - 1;

    /* 对当前区段按照曲率大小对点进行排序 */
    // sort the curvatures from small to large
    for (int k = sp + 1; k <= ep; k++) {
      for (int l = k; l >= sp + 1; l--) {
        if (cloudCurvature[cloudSortInd[l]] <
            cloudCurvature[cloudSortInd[l - 1]]) {
          int temp = cloudSortInd[l - 1];
          cloudSortInd[l - 1] = cloudSortInd[l];
          cloudSortInd[l] = temp;
        }
      }
    }
	
    /* 对当前区段按照反射率大小对点进行排序 */
    // sort the reflectivity from small to large
    for (int k = sp + 1; k <= ep; k++) {
      for (int l = k; l >= sp + 1; l--) {
        if (cloudReflect[reflectSortInd[l]] <
            cloudReflect[reflectSortInd[l - 1]]) {
          int temp = reflectSortInd[l - 1];
          reflectSortInd[l - 1] = reflectSortInd[l];
          reflectSortInd[l] = temp;
        }
      }
    }

	/* 提取曲率较小的点作为平面特征点，提取反射率较大的点作为角点特征点 */
	/* 除了曲率之外，算法将＞100米的点优先添加到特征点中 */
    int smallestPickedNum = 1;
    int sharpestPickedNum = 1;
    for (int k = sp; k <= ep; k++) {
      int ind = cloudSortInd[k];

      /* 仅处理那些尚未被标志的点 */
      if (CloudFeatureFlag[ind] != 0) continue;

      if (cloudCurvature[ind] < thFlatThreshold * cloudDepth[ind] * thFlatThreshold * cloudDepth[ind]) {
        
		/* 将曲率小于阈值的点暂列为候选特征点 */
        CloudFeatureFlag[ind] = 3;

		/* 如果当前点的相邻点与其相邻点的距离很近，并且距离小于100米，则丢弃该点 */
        for (int l = 1; l <= thNumCurvSize; l++) {
          float diffX = _laserCloud->points[ind + l].x -
                        _laserCloud->points[ind + l - 1].x;
          float diffY = _laserCloud->points[ind + l].y -
                        _laserCloud->points[ind + l - 1].y;
          float diffZ = _laserCloud->points[ind + l].z -
                        _laserCloud->points[ind + l - 1].z;
          if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.02 || cloudDepth[ind] > thDistanceFaraway) {
            break;
          }
          /* 特征值为1，表明该点将被丢弃 */
          CloudFeatureFlag[ind + l] = 1;
        }
        for (int l = -1; l >= -thNumCurvSize; l--) {
          float diffX = _laserCloud->points[ind + l].x -
                        _laserCloud->points[ind + l + 1].x;
          float diffY = _laserCloud->points[ind + l].y -
                        _laserCloud->points[ind + l + 1].y;
          float diffZ = _laserCloud->points[ind + l].z -
                        _laserCloud->points[ind + l + 1].z;
          if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.02 || cloudDepth[ind] > thDistanceFaraway) {
            break;
          }
          /* 特征值为1，表明该点将被丢弃 */
          CloudFeatureFlag[ind + l] = 1;
        }
      }
    }
    
	/* 每个区段默认只选3个平面特征点，除非该点的距离＞100米，或者该点是角点 */
    for (int k = sp; k <= ep; k++) {
      int ind = cloudSortInd[k];
      if(((CloudFeatureFlag[ind] == 3) && (smallestPickedNum <= thNumFlat)) || 
          ((CloudFeatureFlag[ind] == 3) && (cloudDepth[ind] > thDistanceFaraway)) ||
          cloudAngle[ind] == 1){
        smallestPickedNum ++;
        CloudFeatureFlag[ind] = 2; //特征值为2，表明该点被正式列入平面特征点
        /* 统计特征点中距离大于100米的点数 */
        if(cloudDepth[ind] > thDistanceFaraway) {
          thDistanceFaraway_fea++;
        }
      }
      
	  /* 如果曲率低于平面点阈值的0.7倍，且反射率＞20，则给与该点300的特征值，每个区段最多不超过3个 */
	  /* 可以认为是从曲率较低的点中提取那些反射率特别高的点，作为候选的角点特征点 */
	  /* FIXME：特征值为300的点最后为什么没有用？ */
      int idx = reflectSortInd[k];
      if(cloudCurvature[idx] < 0.7 * thFlatThreshold * cloudDepth[idx] * thFlatThreshold * cloudDepth[idx]
         && sharpestPickedNum <= 3 && cloudReflect[idx] > 20.0){
        sharpestPickedNum ++;
        CloudFeatureFlag[idx] = 300;
      }
    }
    
  }

  //---------------------------------------- line feature where surfaces meet -------------------------------------

  /* 找出角点特征点 */
  /* 1.首先确认当前点两侧的曲率都比较小，满足平面点的要求 */
  /* 2.然后求当前点到两侧的方向向量均值 */
  /* 3.然后求当前点到两侧第4个点的距离 */
  /* 4.当两侧方向向量夹角在60~120度之间，且当前点到两侧第4点的距离＞0.05，则认定为角点 */
  for (int i = 5; i < cloudSize - 5; i += count_num ) {
	  
	/* 求当前点到坐标原点O的距离 */ 
    float depth = sqrt(_laserCloud->points[i].x * _laserCloud->points[i].x +
                       _laserCloud->points[i].y * _laserCloud->points[i].y +
                       _laserCloud->points[i].z * _laserCloud->points[i].z);
    /* 求左侧曲率 */
	/* 去包含当前点C以及左右两侧相邻的各四个点： */
	/* L4 L3 L2 L1 C R1 R2 R3 R4 */
	/* 然后分别求L2和R2的曲率，作为左侧曲率和右侧曲率 */
	//left curvature
    float ldiffX =
            _laserCloud->points[i - 4].x + _laserCloud->points[i - 3].x
            - 4 * _laserCloud->points[i - 2].x
            + _laserCloud->points[i - 1].x + _laserCloud->points[i].x;

    float ldiffY =
            _laserCloud->points[i - 4].y + _laserCloud->points[i - 3].y
            - 4 * _laserCloud->points[i - 2].y
            + _laserCloud->points[i - 1].y + _laserCloud->points[i].y;

    float ldiffZ =
            _laserCloud->points[i - 4].z + _laserCloud->points[i - 3].z
            - 4 * _laserCloud->points[i - 2].z
            + _laserCloud->points[i - 1].z + _laserCloud->points[i].z;

    float left_curvature = ldiffX * ldiffX + ldiffY * ldiffY + ldiffZ * ldiffZ;

	/* 如果左侧曲率小于阈值，则设置左侧是平面的标志 */
    if(left_curvature < thFlatThreshold * depth){

      /* FIXME:下面这个局部vector压入操作似乎没有意义，后面并没有使用 */
      std::vector<PointType> left_list;
      for(int j = -4; j < 0; j++){
        left_list.push_back(_laserCloud->points[i + j]);
      }

      left_surf_flag = true;
    }
    else{
      left_surf_flag = false;
    }

    /* 求右侧曲率 */
    //right curvature
    float rdiffX =
            _laserCloud->points[i + 4].x + _laserCloud->points[i + 3].x
            - 4 * _laserCloud->points[i + 2].x
            + _laserCloud->points[i + 1].x + _laserCloud->points[i].x;

    float rdiffY =
            _laserCloud->points[i + 4].y + _laserCloud->points[i + 3].y
            - 4 * _laserCloud->points[i + 2].y
            + _laserCloud->points[i + 1].y + _laserCloud->points[i].y;

    float rdiffZ =
            _laserCloud->points[i + 4].z + _laserCloud->points[i + 3].z
            - 4 * _laserCloud->points[i + 2].z
            + _laserCloud->points[i + 1].z + _laserCloud->points[i].z;

    float right_curvature = rdiffX * rdiffX + rdiffY * rdiffY + rdiffZ * rdiffZ;

    /* 如果右侧曲率小于阈值，则设置右侧是平面的标志 */
    if(right_curvature < thFlatThreshold * depth){

      /* FIXME:下面这个局部vector压入操作似乎没有意义，后面并没有使用 */
      std::vector<PointType> right_list;
      for(int j = 1; j < 5; j++){
        right_list.push_back(_laserCloud->points[i + j]);
      }
      count_num = 4;
      right_surf_flag = true;
    }
    else{
      count_num = 1;
      right_surf_flag = false;
    }

    /* 如果左右两侧曲率都比较小，即近似直线 */
    //calculate the included angle
    if(left_surf_flag && right_surf_flag){
      debugnum4 ++;

      /* 求从当前点到左侧4个点的方向向量均值 */
      Eigen::Vector3d norm_left(0,0,0);
      Eigen::Vector3d norm_right(0,0,0);
      for(int k = 1;k<5;k++){
        Eigen::Vector3d tmp = Eigen::Vector3d(_laserCloud->points[i - k].x - _laserCloud->points[i].x,
                                              _laserCloud->points[i - k].y - _laserCloud->points[i].y,
                                              _laserCloud->points[i - k].z - _laserCloud->points[i].z);
        tmp.normalize();
        norm_left += (k/10.0)* tmp;
      }
	  /* 求从当前点到右侧4个点的方向向量均值 */
      for(int k = 1;k<5;k++){
        Eigen::Vector3d tmp = Eigen::Vector3d(_laserCloud->points[i + k].x - _laserCloud->points[i].x,
                                              _laserCloud->points[i + k].y - _laserCloud->points[i].y,
                                              _laserCloud->points[i + k].z - _laserCloud->points[i].z);
        tmp.normalize();
        norm_right += (k/10.0)* tmp;
      }
      
	  /* 求左右两个方向向量的夹角的余弦 */
      //calculate the angle between this group and the previous group
      double cc = fabs( norm_left.dot(norm_right) / (norm_left.norm()*norm_right.norm()) );
	  
	  /* 分别求左右两侧第四个点与当前点的距离 */
      //calculate the maximum distance, the distance cannot be too small
      Eigen::Vector3d last_tmp = Eigen::Vector3d(_laserCloud->points[i - 4].x - _laserCloud->points[i].x,
                                                 _laserCloud->points[i - 4].y - _laserCloud->points[i].y,
                                                 _laserCloud->points[i - 4].z - _laserCloud->points[i].z);
      Eigen::Vector3d current_tmp = Eigen::Vector3d(_laserCloud->points[i + 4].x - _laserCloud->points[i].x,
                                                    _laserCloud->points[i + 4].y - _laserCloud->points[i].y,
                                                    _laserCloud->points[i + 4].z - _laserCloud->points[i].z);
      double last_dis = last_tmp.norm();
      double current_dis = current_tmp.norm();

      /* 如果左右两个向量夹角＞60度并＜120度，且左右第四个点与当前点的距离大于0.05，
	   * 则设定当前点的特征值为150，即确定为角点特征点 */
      if(cc < 0.5 && last_dis > 0.05 && current_dis > 0.05 ){ //
        debugnum5 ++;
        CloudFeatureFlag[i] = 150;
      }
    }
  }

  //--------------------------------------------------- break points ---------------------------------------------
  for(int i = 5; i < cloudSize - 5; i ++){
    float diff_left[2];
    float diff_right[2];
	
	/* 求当前点到坐标原点O的距离，即深度 */
	/* FIXME: 之前已经求过，为什么这里重复求？为什么后面没有用？ */
    float depth = sqrt(_laserCloud->points[i].x * _laserCloud->points[i].x +
                       _laserCloud->points[i].y * _laserCloud->points[i].y +
                       _laserCloud->points[i].z * _laserCloud->points[i].z);

	/* 分别求两侧相邻的2个点与当前点的欧氏距离 */
    for(int count = 1; count < 3; count++ ){
      float diffX1 = _laserCloud->points[i + count].x - _laserCloud->points[i].x;
      float diffY1 = _laserCloud->points[i + count].y - _laserCloud->points[i].y;
      float diffZ1 = _laserCloud->points[i + count].z - _laserCloud->points[i].z;
      diff_right[count - 1] = sqrt(diffX1 * diffX1 + diffY1 * diffY1 + diffZ1 * diffZ1);

      float diffX2 = _laserCloud->points[i - count].x - _laserCloud->points[i].x;
      float diffY2 = _laserCloud->points[i - count].y - _laserCloud->points[i].y;
      float diffZ2 = _laserCloud->points[i - count].z - _laserCloud->points[i].z;
      diff_left[count - 1] = sqrt(diffX2 * diffX2 + diffY2 * diffY2 + diffZ2 * diffZ2);
    }

    /* 分别求左右两侧第一个点的深度，即到坐标原点O的距离 */
    float depth_right = sqrt(_laserCloud->points[i + 1].x * _laserCloud->points[i + 1].x +
                             _laserCloud->points[i + 1].y * _laserCloud->points[i + 1].y +
                             _laserCloud->points[i + 1].z * _laserCloud->points[i + 1].z);
    float depth_left = sqrt(_laserCloud->points[i - 1].x * _laserCloud->points[i - 1].x +
                            _laserCloud->points[i - 1].y * _laserCloud->points[i - 1].y +
                            _laserCloud->points[i - 1].z * _laserCloud->points[i - 1].z);
    
	/* 如果左右两侧第一个点与当前点的距离差大于1米 */
    if(fabs(diff_right[0] - diff_left[0]) > thBreakCornerDis){
      /* 如果右侧点的距离＞左侧点的距离 */
      if(diff_right[0] > diff_left[0]){

        /* 当前点是C，左侧相邻点是L，坐标原点是O，求OC与CL夹角∠OCL的余弦cc */
        Eigen::Vector3d surf_vector = Eigen::Vector3d(_laserCloud->points[i - 1].x - _laserCloud->points[i].x,
                                                      _laserCloud->points[i - 1].y - _laserCloud->points[i].y,
                                                      _laserCloud->points[i - 1].z - _laserCloud->points[i].z);
        Eigen::Vector3d lidar_vector = Eigen::Vector3d(_laserCloud->points[i].x,
                                                       _laserCloud->points[i].y,
                                                       _laserCloud->points[i].z);
        double left_surf_dis = surf_vector.norm();
        //calculate the angle between the laser direction and the surface
        double cc = fabs( surf_vector.dot(lidar_vector) / (surf_vector.norm()*lidar_vector.norm()) );

        /* FIXME: 下面这段代码其实没有用 */
        /* 当前点及其左侧三个点分别是：L3,L2,L1,C */
        /* 四个点全部加入到left_list中 */
        /* 依次求C_L1、L1_L2、L2_L3、L3_L4四个向量 */
        /* 找到四个向量中最长和最短的两个 */
        std::vector<PointType> left_list;
        double min_dis = 10000;
        double max_dis = 0;
        for(int j = 0; j < 4; j++){   //TODO: change the plane window size and add thin rod support

          left_list.push_back(_laserCloud->points[i - j]);

          Eigen::Vector3d temp_vector = Eigen::Vector3d(_laserCloud->points[i - j].x - _laserCloud->points[i - j - 1].x,
                                                        _laserCloud->points[i - j].y - _laserCloud->points[i - j - 1].y,
                                                        _laserCloud->points[i - j].z - _laserCloud->points[i - j - 1].z);
          if(j == 3) break;
          double temp_dis = temp_vector.norm();
          if(temp_dis < min_dis) min_dis = temp_dis;
          if(temp_dis > max_dis) max_dis = temp_dis;
        }
        bool left_is_plane = plane_judge(left_list,100);

        /* 当∠OCL在18~162度之间，并且右侧点的深度大于左侧点的深度时，或者右侧点不存在时，
		 * 将当前点的特征值设为100，即角点特征点将当前点的特征值设为100，即角点特征点 */
        if( cc < 0.95 ){//(max_dis < 2*min_dis) && left_surf_dis < 0.05 * depth  && left_is_plane &&
          if(depth_right > depth_left){
            CloudFeatureFlag[i] = 100;
          }
          else{
            if(depth_right == 0) CloudFeatureFlag[i] = 100;
          }
        }
      }
      else{
		
		/* 当前点是C，右侧相邻点是R，坐标原点是O，求OC与CR夹角∠OCR的余弦cc */
        Eigen::Vector3d surf_vector = Eigen::Vector3d(_laserCloud->points[i + 1].x - _laserCloud->points[i].x,
                                                      _laserCloud->points[i + 1].y - _laserCloud->points[i].y,
                                                      _laserCloud->points[i + 1].z - _laserCloud->points[i].z);
        Eigen::Vector3d lidar_vector = Eigen::Vector3d(_laserCloud->points[i].x,
                                                       _laserCloud->points[i].y,
                                                       _laserCloud->points[i].z);
        double right_surf_dis = surf_vector.norm();
        //calculate the angle between the laser direction and the surface
        double cc = fabs( surf_vector.dot(lidar_vector) / (surf_vector.norm()*lidar_vector.norm()) );

        std::vector<PointType> right_list;
        double min_dis = 10000;
        double max_dis = 0;
        for(int j = 0; j < 4; j++){ //TODO: change the plane window size and add thin rod support
          right_list.push_back(_laserCloud->points[i - j]);
          Eigen::Vector3d temp_vector = Eigen::Vector3d(_laserCloud->points[i + j].x - _laserCloud->points[i + j + 1].x,
                                                        _laserCloud->points[i + j].y - _laserCloud->points[i + j + 1].y,
                                                        _laserCloud->points[i + j].z - _laserCloud->points[i + j + 1].z);

          if(j == 3) break;
          double temp_dis = temp_vector.norm();
          if(temp_dis < min_dis) min_dis = temp_dis;
          if(temp_dis > max_dis) max_dis = temp_dis;
        }
        bool right_is_plane = plane_judge(right_list,100);

        /* 当∠OCR在18~162度之间，并且左侧点的深度大于右侧点的深度时，或者左侧点不存在时，
		 * 将当前点的特征值设为100，即角点特征点将当前点的特征值设为100，即角点特征点 */
        if( cc < 0.95){ //right_is_plane  && (max_dis < 2*min_dis) && right_surf_dis < 0.05 * depth &&

          if(depth_right < depth_left){
            CloudFeatureFlag[i] = 100;
          }
          else{
            if(depth_left == 0) CloudFeatureFlag[i] = 100;
          }
        }
      }
    }

    // break points select
    if(CloudFeatureFlag[i] == 100){
      debugnum2++;
      std::vector<Eigen::Vector3d> front_norms;
      Eigen::Vector3d norm_front(0,0,0);
      Eigen::Vector3d norm_back(0,0,0);

      /* 求从当前点到左侧4个点的方向向量均值，仅当左侧点到当前点的距离＞1米的才能入选 */
      for(int k = 1;k<4;k++){

        float temp_depth = sqrt(_laserCloud->points[i - k].x * _laserCloud->points[i - k].x +
                        _laserCloud->points[i - k].y * _laserCloud->points[i - k].y +
                        _laserCloud->points[i - k].z * _laserCloud->points[i - k].z);

        if(temp_depth < 1){
          continue;
        }

        Eigen::Vector3d tmp = Eigen::Vector3d(_laserCloud->points[i - k].x - _laserCloud->points[i].x,
                                              _laserCloud->points[i - k].y - _laserCloud->points[i].y,
                                              _laserCloud->points[i - k].z - _laserCloud->points[i].z);
        tmp.normalize();
        front_norms.push_back(tmp);
        norm_front += (k/6.0)* tmp;
      }
	  /* 求从当前点到右侧4个点的方向向量均值，仅当右侧点到当前点的距离＞1米的才能入选 */
      std::vector<Eigen::Vector3d> back_norms;
      for(int k = 1;k<4;k++){
		/* FIXME: 下面的代码似乎把i+k写成了i-k，存在明显错误 */
        float temp_depth = sqrt(_laserCloud->points[i - k].x * _laserCloud->points[i - k].x +
                        _laserCloud->points[i - k].y * _laserCloud->points[i - k].y +
                        _laserCloud->points[i - k].z * _laserCloud->points[i - k].z);

        if(temp_depth < 1){
          continue;
        }

        Eigen::Vector3d tmp = Eigen::Vector3d(_laserCloud->points[i + k].x - _laserCloud->points[i].x,
                                              _laserCloud->points[i + k].y - _laserCloud->points[i].y,
                                              _laserCloud->points[i + k].z - _laserCloud->points[i].z);
        tmp.normalize();
        back_norms.push_back(tmp);
        norm_back += (k/6.0)* tmp;
      }
	  /* 求左右两个方向向量夹角的余弦cc的绝对值 */
      double cc = fabs( norm_front.dot(norm_back) / (norm_front.norm()*norm_back.norm()) );
      /* 如果夹角不在18~162度之间，则将该点的特征值设置为101，即排除在角点之外 */
      if(cc < 0.95){
        debugnum3++;
      }else{
        CloudFeatureFlag[i] = 101;
      }
    }
  }

  pcl::PointCloud<PointType>::Ptr laserCloudCorner(new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType> cornerPointsSharp;

  std::vector<int> pointsLessSharp_ori;

  int num_surf = 0;
  int num_corner = 0;

  /* 开始将特征值装入输出变量 */
  //push_back feature
  for(int i = 5; i < cloudSize - 5; i ++){
    /* 求当前点的深度的平方 */
    float dis = _laserCloud->points[i].x * _laserCloud->points[i].x
            + _laserCloud->points[i].y * _laserCloud->points[i].y
            + _laserCloud->points[i].z * _laserCloud->points[i].z;
    /* 深度小于1米的点不做为特征点 */
    if(dis < thLidarNearestDis*thLidarNearestDis) continue;

	/* 仅当特征值为2时才被加入平面特征点 */
    if(CloudFeatureFlag[i] == 2){
      pointsLessFlat.push_back(i);
      num_surf++;
      continue;
    }

	/* 仅当特征值为100或150时，才被加入角点特征点 */
    if(CloudFeatureFlag[i] == 100 || CloudFeatureFlag[i] == 150){ //
      pointsLessSharp_ori.push_back(i);
      laserCloudCorner->push_back(_laserCloud->points[i]);
    }

  }

  for(int i = 0; i < laserCloudCorner->points.size();i++){
      pointsLessSharp.push_back(pointsLessSharp_ori[i]);
      num_corner++;
  }

}

/**
 * @brief 包含点云分割的特征提取，分割的主要目的是剔除不是背景的点云
 * @param msg 订阅的消息，包含原始点云
 * @param laserCloud 输出参数，完整点云
 * @param laserConerFeature 输出参数，角点特征点云
 * @param laserSurfFeature 输出参数，平面点特征点云
 * @param laserNonFeature 输出参数，不规则特征点云
 * @param Used_Line 输入参数，使用多少个扫描线
 */
void LidarFeatureExtractor::FeatureExtract_with_segment(const livox_ros_driver::CustomMsgConstPtr &msg,
                                                        pcl::PointCloud<PointType>::Ptr& laserCloud,
                                                        pcl::PointCloud<PointType>::Ptr& laserConerFeature,
                                                        pcl::PointCloud<PointType>::Ptr& laserSurfFeature,
                                                        pcl::PointCloud<PointType>::Ptr& laserNonFeature,
                                                        sensor_msgs::PointCloud2 &msg_seg,
                                                        const int Used_Line){
  /* 对本地变量进行复位清零 */
  laserCloud->clear();
  laserConerFeature->clear();
  laserSurfFeature->clear();
  laserCloud->clear();
  laserCloud->reserve(15000*N_SCANS);
  for(auto & ptr : vlines){
    ptr->clear();
  }
  for(auto & v : vcorner){
    v.clear();
  }
  for(auto & v : vsurf){
    v.clear();
  }

  int dnum = msg->points.size();

  int *idtrans = (int*)calloc(dnum, sizeof(int));
  float *data=(float*)calloc(dnum*4,sizeof(float));
  int point_num = 0;

  /* 将点云转成PCL格式，并记录每个点的反射强度、时间戳、线号 */
  double timeSpan = ros::Time().fromNSec(msg->points.back().offset_time).toSec();
  PointType point;
  for(const auto& p : msg->points){

    int line_num = (int)p.line;
    if(line_num > Used_Line-1) continue;
    if(p.x < 0.01) continue;
    if (!pcl_isfinite(p.x) ||
        !pcl_isfinite(p.y) ||
        !pcl_isfinite(p.z)) {
      continue;
    }
    point.x = p.x;
    point.y = p.y;
    point.z = p.z;
    point.intensity = p.reflectivity;
    /* 记录当前点时间偏移相对于点云帧时长的比值，主要用于帧内校正 */
    point.normal_x = ros::Time().fromNSec(p.offset_time).toSec() /timeSpan;
    point.normal_y = _int_as_float(line_num);
    laserCloud->push_back(point);

    data[point_num*4+0] = point.x;
    data[point_num*4+1] = point.y;
    data[point_num*4+2] = point.z;
    data[point_num*4+3] = point.intensity;

    point_num++;
  }

  /* 对点云进行聚类分割 */
  PCSeg pcseg;
  pcseg.DoSeg(idtrans,data,dnum);

  /* 将点云按线号分类保存在vlines[]中 */
  std::size_t cloud_num = laserCloud->size();
  for(std::size_t i=0; i<cloud_num; ++i){
    int line_idx = _float_as_int(laserCloud->points[i].normal_y);
    laserCloud->points[i].normal_z = _int_as_float(i);
    vlines[line_idx]->push_back(laserCloud->points[i]);
  }

  /* 启动多线程进行特征提取，每个线程负责处理一条线 */
  std::thread threads[N_SCANS];
  for(int i=0; i<N_SCANS; ++i){
    threads[i] = std::thread(&LidarFeatureExtractor::detectFeaturePoint3, this, std::ref(vlines[i]),std::ref(vcorner[i]));
  }

  /* 等待线程执行完毕，join()方法的作用就是阻塞当前线程，直到特征提取线程执行完成 */
  for(int i=0; i<N_SCANS; ++i){
    threads[i].join();
  }

  /* 对角点特征点进行标记 */
  int num_corner = 0;
  for(int i=0; i<N_SCANS; ++i){
    for(int j=0; j<vcorner[i].size(); ++j){
      laserCloud->points[_float_as_int(vlines[i]->points[vcorner[i][j]].normal_z)].normal_z = 1.0; 
      num_corner++;
    }
  }

  /* 进行平面特征点和不规则特征点的提取 */
  detectFeaturePoint2(laserCloud, laserSurfFeature, laserNonFeature);

  /* 将那些深度不足50米，聚类分割后判定是运动物体的特征点剔除掉 */
  for(std::size_t i=0; i<cloud_num; ++i){
    float dis = laserCloud->points[i].x * laserCloud->points[i].x
                + laserCloud->points[i].y * laserCloud->points[i].y
                + laserCloud->points[i].z * laserCloud->points[i].z;
    if( idtrans[i] > 9 && dis < 50*50){
      laserCloud->points[i].normal_z = 0;
    }
  }

  pcl::PointCloud<PointType>::Ptr laserConerFeature_filter;
  laserConerFeature_filter.reset(new pcl::PointCloud<PointType>());
  laserConerFeature.reset(new pcl::PointCloud<PointType>());
  laserSurfFeature.reset(new pcl::PointCloud<PointType>());
  laserNonFeature.reset(new pcl::PointCloud<PointType>());
  
  /* 输出角点特征点 */
  for(const auto& p : laserCloud->points){
    if(std::fabs(p.normal_z - 1.0) < 1e-5)
      laserConerFeature->push_back(p);
  }

  /* 输出平面特征点和不规则特征点 */
  for(const auto& p : laserCloud->points){
    if(std::fabs(p.normal_z - 2.0) < 1e-5)
      laserSurfFeature->push_back(p);
    if(std::fabs(p.normal_z - 3.0) < 1e-5)
      laserNonFeature->push_back(p);
  }

}

/**
 * @brief 提取点云中的平面和不规则特征点。
 *   利用点云的协方差矩阵及其特征向量和特征值来判断当前点及其临近点是否分布在平面上
 *   不规则特征点云的提取方法：
 *     1、距离＞100米的非角点一律归入不规则特征点
 *     2、临近的若干个呈团状分布的点归入不规则特征点
 * @param cloud 输入参数，输入点云，不需是同一个扫描线的点云
 * @param pointsLessFlat 输出参数，平面特征点云
 * @param pointsNonFeature 输出参数，不规则特征点云
 */
void LidarFeatureExtractor::detectFeaturePoint2(pcl::PointCloud<PointType>::Ptr& cloud,
                                                pcl::PointCloud<PointType>::Ptr& pointsLessFlat,
                                                pcl::PointCloud<PointType>::Ptr& pointsNonFeature){

  int cloudSize = cloud->points.size();

  pointsLessFlat.reset(new pcl::PointCloud<PointType>());
  pointsNonFeature.reset(new pcl::PointCloud<PointType>());

  pcl::KdTreeFLANN<PointType>::Ptr KdTreeCloud;
  KdTreeCloud.reset(new pcl::KdTreeFLANN<PointType>);
  KdTreeCloud->setInputCloud(cloud);

  std::vector<int> _pointSearchInd;
  std::vector<float> _pointSearchSqDis;

  int num_near = 10;
  int stride = 1;
  int interval = 4;

  for(int i = 5; i < cloudSize - 5; i = i+stride) {

    /* 如果该点已被识别为角点则略过 */
    if(fabs(cloud->points[i].normal_z - 1.0) < 1e-5) {
      continue;
    }

    double thre1d = 0.5;
    double thre2d = 0.8;
    double thre3d = 0.5;
    double thre3d2 = 0.13;

    /* 求当前点的深度 */
    double disti = sqrt(cloud->points[i].x * cloud->points[i].x + 
                        cloud->points[i].y * cloud->points[i].y + 
                        cloud->points[i].z * cloud->points[i].z);
	/* 根据点的深度设置步长和间隔，深度越大步长和间隔越小 */
    if(disti < 30.0) {
      thre1d = 0.5;
      thre2d = 0.8;
      thre3d2 = 0.07;
      stride = 14;
      interval = 4;
    } else if(disti < 60.0) {
      stride = 10;
      interval = 3;
    } else {
      stride = 1;
      interval = 0;
    }

    /* 如果点的深度＞100米，则直接将该点确定为不规则特征点云 */
	/* 根据点的深度不同，设置不同的KDtree最近点搜索数量 */
    if(disti > 100.0) {
      num_near = 6; //距离＞100米搜索6个
      cloud->points[i].normal_z = 3.0;
      pointsNonFeature->points.push_back(cloud->points[i]);
      continue;
    } else if(disti > 60.0) {
      num_near = 8; //距离＞60米搜索8个
    } else {
      num_near = 10; //距离<60米搜索10个
    }

    /* 在点云中搜索当前点的若干个临近点 */
    KdTreeCloud->nearestKSearch(cloud->points[i], num_near, _pointSearchInd, _pointSearchSqDis);

    /* 如果临近点的距离＞5米，且当前点的深度不足90米，则放弃当前点 */
    if (_pointSearchSqDis[num_near-1] > 5.0 && disti < 90.0) {
      continue;
    }

    Eigen::Matrix< double, 3, 3 > _matA1;
    _matA1.setZero();

    /* 求临近点的质心 */
    float cx = 0;
    float cy = 0;
    float cz = 0;
    for (int j = 0; j < num_near; j++) {
      cx += cloud->points[_pointSearchInd[j]].x;
      cy += cloud->points[_pointSearchInd[j]].y;
      cz += cloud->points[_pointSearchInd[j]].z;
    }
    cx /= num_near;
    cy /= num_near;
    cz /= num_near;

    /* 构造点云的协方差矩阵 */
    float a11 = 0;
    float a12 = 0;
    float a13 = 0;
    float a22 = 0;
    float a23 = 0;
    float a33 = 0;
    for (int j = 0; j < num_near; j++) {
      float ax = cloud->points[_pointSearchInd[j]].x - cx;
      float ay = cloud->points[_pointSearchInd[j]].y - cy;
      float az = cloud->points[_pointSearchInd[j]].z - cz;

      a11 += ax * ax;
      a12 += ax * ay;
      a13 += ax * az;
      a22 += ay * ay;
      a23 += ay * az;
      a33 += az * az;
    }
    a11 /= num_near;
    a12 /= num_near;
    a13 /= num_near;
    a22 /= num_near;
    a23 /= num_near;
    a33 /= num_near;

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
	/* 特征值和特征向量增序排列，第三个是主方向，特征值最大 */
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(_matA1);
	/* 求三个方向特征值的相对强度，a1d是主方向强度，a2d是次主方向强度，a3d是次次方向强度 */
    double a1d = (sqrt(saes.eigenvalues()[2]) - sqrt(saes.eigenvalues()[1])) / sqrt(saes.eigenvalues()[2]);
    double a2d = (sqrt(saes.eigenvalues()[1]) - sqrt(saes.eigenvalues()[0])) / sqrt(saes.eigenvalues()[2]);
    double a3d = sqrt(saes.eigenvalues()[0]) / sqrt(saes.eigenvalues()[2]);

    /* 如果前两个特征值比较大且数值接近、第三个特征值比较小时，说明临近点呈面状分布 */
    if(a2d > thre2d || (a3d < thre3d2 && a1d < thre1d)) {
      /* 将当前点及其临近点添加到平面特征点云 */
      for(int k = 1; k < interval; k++) {
        cloud->points[i-k].normal_z = 2.0;
        pointsLessFlat->points.push_back(cloud->points[i-k]);
        cloud->points[i+k].normal_z = 2.0;
        pointsLessFlat->points.push_back(cloud->points[i+k]);
      }
      cloud->points[i].normal_z = 2.0;
      pointsLessFlat->points.push_back(cloud->points[i]);
    } else if(a3d > thre3d) { // 如果第三个特征向量数值也不小，说明临近点呈团状分布
      /* 将当前点及其临近点添加到不规则特征点云 */
      for(int k = 1; k < interval; k++) {
        cloud->points[i-k].normal_z = 3.0;
        pointsNonFeature->points.push_back(cloud->points[i-k]);
        cloud->points[i+k].normal_z = 3.0;
        pointsNonFeature->points.push_back(cloud->points[i+k]);
      }
      cloud->points[i].normal_z = 3.0;
      pointsNonFeature->points.push_back(cloud->points[i]);
    }
  }  
}

/**
 * @brief 对同一条扫描线上的点进行角点特征提取，
 *  该方法与detectFeaturePoint方法中提取角点的方法几乎完全一致，仅有几个角度参数不一致。
 * @param cloud 输入参数，同一条扫描线上的点
 * @param pointsLessSharp 输出参数，角点特征点云
 */
void LidarFeatureExtractor::detectFeaturePoint3(pcl::PointCloud<PointType>::Ptr& cloud,
                                                std::vector<int>& pointsLessSharp){
  int CloudFeatureFlag[20000];
  float cloudCurvature[20000];
  float cloudDepth[20000];
  int cloudSortInd[20000];
  float cloudReflect[20000];
  int reflectSortInd[20000];
  int cloudAngle[20000];

  pcl::PointCloud<PointType>::Ptr& laserCloudIn = cloud;

  int cloudSize = laserCloudIn->points.size();

  /* 把输入点云腾挪到_laserCloud中，特征值清零 */
  PointType point;
  pcl::PointCloud<PointType>::Ptr _laserCloud(new pcl::PointCloud<PointType>());
  _laserCloud->reserve(cloudSize);

  for (int i = 0; i < cloudSize; i++) {
    point.x = laserCloudIn->points[i].x;
    point.y = laserCloudIn->points[i].y;
    point.z = laserCloudIn->points[i].z;
    point.normal_x = 1.0;
    point.intensity = laserCloudIn->points[i].intensity;

    if (!pcl_isfinite(point.x) ||
        !pcl_isfinite(point.y) ||
        !pcl_isfinite(point.z)) {
      continue;
    }

    _laserCloud->push_back(point);
    CloudFeatureFlag[i] = 0;
  }

  cloudSize = _laserCloud->size();

  int count_num = 1;
  bool left_surf_flag = false;
  bool right_surf_flag = false;

  //--------------------------------------------------- break points ---------------------------------------------
  for(int i = 5; i < cloudSize - 5; i ++){
    float diff_left[2];
    float diff_right[2];
	/* 求当前点到坐标原点O的距离，即深度 */
    float depth = sqrt(_laserCloud->points[i].x * _laserCloud->points[i].x +
                       _laserCloud->points[i].y * _laserCloud->points[i].y +
                       _laserCloud->points[i].z * _laserCloud->points[i].z);

	/* 分别求两侧相邻的2个点与当前点的欧氏距离 */
    for(int count = 1; count < 3; count++ ){
      float diffX1 = _laserCloud->points[i + count].x - _laserCloud->points[i].x;
      float diffY1 = _laserCloud->points[i + count].y - _laserCloud->points[i].y;
      float diffZ1 = _laserCloud->points[i + count].z - _laserCloud->points[i].z;
      diff_right[count - 1] = sqrt(diffX1 * diffX1 + diffY1 * diffY1 + diffZ1 * diffZ1);

      float diffX2 = _laserCloud->points[i - count].x - _laserCloud->points[i].x;
      float diffY2 = _laserCloud->points[i - count].y - _laserCloud->points[i].y;
      float diffZ2 = _laserCloud->points[i - count].z - _laserCloud->points[i].z;
      diff_left[count - 1] = sqrt(diffX2 * diffX2 + diffY2 * diffY2 + diffZ2 * diffZ2);
    }

    /* 分别求左右两侧第一个点的深度，即到坐标原点O的距离 */
    float depth_right = sqrt(_laserCloud->points[i + 1].x * _laserCloud->points[i + 1].x +
                             _laserCloud->points[i + 1].y * _laserCloud->points[i + 1].y +
                             _laserCloud->points[i + 1].z * _laserCloud->points[i + 1].z);
    float depth_left = sqrt(_laserCloud->points[i - 1].x * _laserCloud->points[i - 1].x +
                            _laserCloud->points[i - 1].y * _laserCloud->points[i - 1].y +
                            _laserCloud->points[i - 1].z * _laserCloud->points[i - 1].z);

    /* 如果左右两侧第一个点与当前点的距离差大于1米 */
    if(fabs(diff_right[0] - diff_left[0]) > thBreakCornerDis){
      /* 如果右侧点的距离＞左侧点的距离 */
      if(diff_right[0] > diff_left[0]){
        /* 当前点是C，左侧相邻点是L，坐标原点是O，求OC与CL夹角∠OCL的余弦cc */
        Eigen::Vector3d surf_vector = Eigen::Vector3d(_laserCloud->points[i - 1].x - _laserCloud->points[i].x,
                                                      _laserCloud->points[i - 1].y - _laserCloud->points[i].y,
                                                      _laserCloud->points[i - 1].z - _laserCloud->points[i].z);
        Eigen::Vector3d lidar_vector = Eigen::Vector3d(_laserCloud->points[i].x,
                                                       _laserCloud->points[i].y,
                                                       _laserCloud->points[i].z);
        double left_surf_dis = surf_vector.norm();
        //calculate the angle between the laser direction and the surface
        double cc = fabs( surf_vector.dot(lidar_vector) / (surf_vector.norm()*lidar_vector.norm()) );

        /* FIXME: 下面这段代码其实没有用 */
        /* 当前点及其左侧三个点分别是：L3,L2,L1,C */
        /* 四个点全部加入到left_list中 */
        /* 依次求C_L1、L1_L2、L2_L3、L3_L4四个向量 */
        /* 找到四个向量中最长和最短的两个 */
        std::vector<PointType> left_list;
        double min_dis = 10000;
        double max_dis = 0;
        for(int j = 0; j < 4; j++){   //TODO: change the plane window size and add thin rod support
          left_list.push_back(_laserCloud->points[i - j]);
          Eigen::Vector3d temp_vector = Eigen::Vector3d(_laserCloud->points[i - j].x - _laserCloud->points[i - j - 1].x,
                                                        _laserCloud->points[i - j].y - _laserCloud->points[i - j - 1].y,
                                                        _laserCloud->points[i - j].z - _laserCloud->points[i - j - 1].z);

          if(j == 3) break;
          double temp_dis = temp_vector.norm();
          if(temp_dis < min_dis) min_dis = temp_dis;
          if(temp_dis > max_dis) max_dis = temp_dis;
        }
        // bool left_is_plane = plane_judge(left_list,0.3);

        /* 当∠OCL在21.5~158.5度之间，并且右侧点的深度大于左侧点的深度时，或者右侧点不存在时，
		 * 将当前点的特征值设为100，即角点特征点将当前点的特征值设为100，即角点特征点 */
        if(cc < 0.93){//(max_dis < 2*min_dis) && left_surf_dis < 0.05 * depth  && left_is_plane &&
          if(depth_right > depth_left){
            CloudFeatureFlag[i] = 100;
          }
          else{
            if(depth_right == 0) CloudFeatureFlag[i] = 100;
          }
        }
      }
      else{ /* 如果左侧点的距离＞右侧点的距离 */

		/* 当前点是C，右侧相邻点是R，坐标原点是O，求OC与CR夹角∠OCR的余弦cc */
        Eigen::Vector3d surf_vector = Eigen::Vector3d(_laserCloud->points[i + 1].x - _laserCloud->points[i].x,
                                                      _laserCloud->points[i + 1].y - _laserCloud->points[i].y,
                                                      _laserCloud->points[i + 1].z - _laserCloud->points[i].z);
        Eigen::Vector3d lidar_vector = Eigen::Vector3d(_laserCloud->points[i].x,
                                                       _laserCloud->points[i].y,
                                                       _laserCloud->points[i].z);
        double right_surf_dis = surf_vector.norm();
        //calculate the angle between the laser direction and the surface
        double cc = fabs( surf_vector.dot(lidar_vector) / (surf_vector.norm()*lidar_vector.norm()) );

        std::vector<PointType> right_list;
        double min_dis = 10000;
        double max_dis = 0;
        for(int j = 0; j < 4; j++){ //TODO: change the plane window size and add thin rod support
          right_list.push_back(_laserCloud->points[i - j]);
          Eigen::Vector3d temp_vector = Eigen::Vector3d(_laserCloud->points[i + j].x - _laserCloud->points[i + j + 1].x,
                                                        _laserCloud->points[i + j].y - _laserCloud->points[i + j + 1].y,
                                                        _laserCloud->points[i + j].z - _laserCloud->points[i + j + 1].z);

          if(j == 3) break;
          double temp_dis = temp_vector.norm();
          if(temp_dis < min_dis) min_dis = temp_dis;
          if(temp_dis > max_dis) max_dis = temp_dis;
        }
        // bool right_is_plane = plane_judge(right_list,0.3);

        /* 当∠OCR在21.5~158.5度之间，并且左侧点的深度大于右侧点的深度时，或者左侧点不存在时，
		 * 将当前点的特征值设为100，即角点特征点将当前点的特征值设为100，即角点特征点 */
        if(cc < 0.93){ //right_is_plane  && (max_dis < 2*min_dis) && right_surf_dis < 0.05 * depth &&

          if(depth_right < depth_left){
            CloudFeatureFlag[i] = 100;
          }
          else{
            if(depth_left == 0) CloudFeatureFlag[i] = 100;
          }
        }
      }
    }

    // break points select
    if(CloudFeatureFlag[i] == 100){
      std::vector<Eigen::Vector3d> front_norms;
      Eigen::Vector3d norm_front(0,0,0);
      Eigen::Vector3d norm_back(0,0,0);

      /* 求从当前点到左侧4个邻点的方向向量均值，邻点到当前点的距离必须＞1米 */
      for(int k = 1;k<4;k++){

        float temp_depth = sqrt(_laserCloud->points[i - k].x * _laserCloud->points[i - k].x +
                        _laserCloud->points[i - k].y * _laserCloud->points[i - k].y +
                        _laserCloud->points[i - k].z * _laserCloud->points[i - k].z);

        if(temp_depth < 1){
          continue;
        }

        Eigen::Vector3d tmp = Eigen::Vector3d(_laserCloud->points[i - k].x - _laserCloud->points[i].x,
                                              _laserCloud->points[i - k].y - _laserCloud->points[i].y,
                                              _laserCloud->points[i - k].z - _laserCloud->points[i].z);
        tmp.normalize();
        front_norms.push_back(tmp);
        norm_front += (k/6.0)* tmp;
      }
	  /* 求从当前点到右侧4个邻点的方向向量均值，邻点到当前点的距离必须＞1米 */
      std::vector<Eigen::Vector3d> back_norms;
      for(int k = 1;k<4;k++){

        float temp_depth = sqrt(_laserCloud->points[i - k].x * _laserCloud->points[i - k].x +
                        _laserCloud->points[i - k].y * _laserCloud->points[i - k].y +
                        _laserCloud->points[i - k].z * _laserCloud->points[i - k].z);

        if(temp_depth < 1){
          continue;
        }

        Eigen::Vector3d tmp = Eigen::Vector3d(_laserCloud->points[i + k].x - _laserCloud->points[i].x,
                                              _laserCloud->points[i + k].y - _laserCloud->points[i].y,
                                              _laserCloud->points[i + k].z - _laserCloud->points[i].z);
        tmp.normalize();
        back_norms.push_back(tmp);
        norm_back += (k/6.0)* tmp;
      }
	  /* 求左右两个方向向量夹角的余弦cc的绝对值 */
      double cc = fabs( norm_front.dot(norm_back) / (norm_front.norm()*norm_back.norm()) );
      /* 如果夹角不在21.5~158.5度之间，则将该点的特征值设置为101，即排除在角点之外 */
      if(cc < 0.93){
      }else{
        CloudFeatureFlag[i] = 101;
      }

    }

  }

  pcl::PointCloud<PointType>::Ptr laserCloudCorner(new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType> cornerPointsSharp;

  std::vector<int> pointsLessSharp_ori;

  int num_surf = 0;
  int num_corner = 0;

  for(int i = 5; i < cloudSize - 5; i ++){
	/* 分别构造当前点C、左邻点L、右邻点R向量 */
    Eigen::Vector3d left_pt = Eigen::Vector3d(_laserCloud->points[i - 1].x,
                                              _laserCloud->points[i - 1].y,
                                              _laserCloud->points[i - 1].z);
    Eigen::Vector3d right_pt = Eigen::Vector3d(_laserCloud->points[i + 1].x,
                                               _laserCloud->points[i + 1].y,
                                               _laserCloud->points[i + 1].z);

    Eigen::Vector3d cur_pt = Eigen::Vector3d(_laserCloud->points[i].x,
                                             _laserCloud->points[i].y,
                                             _laserCloud->points[i].z);

    /* 求当前点的深度 */
    float dis = _laserCloud->points[i].x * _laserCloud->points[i].x +
                _laserCloud->points[i].y * _laserCloud->points[i].y +
                _laserCloud->points[i].z * _laserCloud->points[i].z;

    /* 分别求∠LOR、∠LOC、∠COR的余弦 */
	/* FIXME: ∠LOC、∠COR的余弦求了却没有用 */
    double clr = fabs(left_pt.dot(right_pt) / (left_pt.norm()*right_pt.norm()));
    double cl = fabs(left_pt.dot(cur_pt) / (left_pt.norm()*cur_pt.norm()));
    double cr = fabs(right_pt.dot(cur_pt) / (right_pt.norm()*cur_pt.norm()));

    /* 当∠LOR在2.6°~177.4°之间时，则该点也被归入角点特征点 */
	/* FIXME:为什么左右邻点的夹角∠LOR在2.6°~177.4°之间时，当前点可以算作是角点？ */
    if(clr < 0.999){
      CloudFeatureFlag[i] = 200;
    }

    /* 深度小于1米的点不做为特征点 */
    if(dis < thLidarNearestDis*thLidarNearestDis) continue;

	/* 仅当特征值为100或200时，才被加入角点特征点 */
    if(CloudFeatureFlag[i] == 100 || CloudFeatureFlag[i] == 200){ //
      pointsLessSharp_ori.push_back(i);
      laserCloudCorner->push_back(_laserCloud->points[i]);
    }
  }

  for(int i = 0; i < laserCloudCorner->points.size();i++){
      pointsLessSharp.push_back(pointsLessSharp_ori[i]);
      num_corner++;
  }

}

/**
 * @brief 对原始点云进行特征提取
 * @param msg 订阅的消息，包含原始点云
 * @param laserCloud 输出参数，完整点云
 * @param laserConerFeature 输出参数，角点特征点云
 * @param laserSurfFeature 输出参数，平面点特征点云
 * @param Used_Line 输入参数，使用多少个扫描线
 */
void LidarFeatureExtractor::FeatureExtract(const livox_ros_driver::CustomMsgConstPtr &msg,
                                           pcl::PointCloud<PointType>::Ptr& laserCloud,
                                           pcl::PointCloud<PointType>::Ptr& laserConerFeature,
                                           pcl::PointCloud<PointType>::Ptr& laserSurfFeature,
                                           const int Used_Line){
  /* 清空缓存特征点云的各种模块变量 */
  laserCloud->clear();
  laserConerFeature->clear();
  laserSurfFeature->clear();
  laserCloud->reserve(15000*N_SCANS);
  for(auto & ptr : vlines){
    ptr->clear();
  }
  for(auto & v : vcorner){
    v.clear();
  }
  for(auto & v : vsurf){
    v.clear();
  }
  
  /* 将点云转成PCL格式，并记录每个点的反射强度、时间戳、线号 */
  /* 获得当前点云帧的时长 */
  double timeSpan = ros::Time().fromNSec(msg->points.back().offset_time).toSec();
  PointType point;
  for(const auto& p : msg->points){
    int line_num = (int)p.line;
    if(line_num > Used_Line-1) continue;
    if(p.x < 0.01) continue;
    point.x = p.x;
    point.y = p.y;
    point.z = p.z;
    point.intensity = p.reflectivity;
    /* 记录当前点时间偏移相对于点云帧时长的比值，主要用于帧内校正 */
    point.normal_x = ros::Time().fromNSec(p.offset_time).toSec() /timeSpan;
    point.normal_y = _int_as_float(line_num);
    laserCloud->push_back(point);
  }
  
  /* 将点云按线号分类保存在vlines[]中 */
  std::size_t cloud_num = laserCloud->size();
  for(std::size_t i=0; i<cloud_num; ++i){
    int line_idx = _float_as_int(laserCloud->points[i].normal_y);
    laserCloud->points[i].normal_z = _int_as_float(i);
    vlines[line_idx]->push_back(laserCloud->points[i]);
    laserCloud->points[i].normal_z = 0;
  }
  
  /* 启动多线程进行特征提取，每个线程负责处理一条线 */
  std::thread threads[N_SCANS];
  for(int i=0; i<N_SCANS; ++i){
    threads[i] = std::thread(&LidarFeatureExtractor::detectFeaturePoint, this, std::ref(vlines[i]),
					  std::ref(vcorner[i]), std::ref(vsurf[i]));
  }
  
  /* 等待线程执行完毕，join()方法的作用就是阻塞当前线程，直到特征提取线程执行完成 */
  for(int i=0; i<N_SCANS; ++i){
    threads[i].join();
  }
  
  for(int i=0; i<N_SCANS; ++i){
    for(int j=0; j<vcorner[i].size(); ++j){
	  laserCloud->points[_float_as_int(vlines[i]->points[vcorner[i][j]].normal_z)].normal_z = 1.0;
    }
    for(int j=0; j<vsurf[i].size(); ++j){
	  laserCloud->points[_float_as_int(vlines[i]->points[vsurf[i][j]].normal_z)].normal_z = 2.0;
    }
  }

  for(const auto& p : laserCloud->points){
    if(std::fabs(p.normal_z - 1.0) < 1e-5)
	  laserConerFeature->push_back(p);
  }
  for(const auto& p : laserCloud->points){
    if(std::fabs(p.normal_z - 2.0) < 1e-5)
      laserSurfFeature->push_back(p);
    }
}