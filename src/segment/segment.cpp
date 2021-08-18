
/* 下面所有双斜杠//起头的注释都是代码原作者添加的 */
/* 代码原作者添加的部分注释不是很明确，看的时候需要特别注意 */

#include "segment/segment.hpp"

#define N_FRAME 5

float tmp_gnd_pose[100*6];
int tem_gnd_num = 0;

PCSeg::PCSeg()
{
    this->posFlag=0;
    this->pVImg=(unsigned char*)calloc(DN_SAMPLE_IMG_NX*DN_SAMPLE_IMG_NY*DN_SAMPLE_IMG_NZ,sizeof(unsigned char));
    this->corPoints=NULL;
}
PCSeg::~PCSeg()
{
    if(this->pVImg!=NULL)
    {
        free(this->pVImg);
    }
    if(this->corPoints!=NULL)
    {
        free(this->corPoints);
    }
}

/**
 * @brief 对点云进行分割，为每个点赋予一个类型，主要目的是区别背景和前景
 * @param pLabel1 输出参数，点云中每个点的类型
 * @param fPoints1 输入参数，完整点云
 * @param pointNum 输入参数，点云点数
 */
int PCSeg::DoSeg(int *pLabel1, float* fPoints1, int pointNum)
{

	/* 将雷达的扫描空间划分成600×200×100个网格 */
	/* 每个网格的尺寸是40×40×20cm */
	/* 将点云映射到这个网格空间，每个网格取一个点，进行降采样 */
    // 1 down sampling
    float *fPoints2=(float*)calloc(pointNum*4,sizeof(float));
    int *idtrans1=(int*)calloc(pointNum,sizeof(int));
    int *idtrans2=(int*)calloc(pointNum,sizeof(int));
    int pntNum=0;
    if (this->pVImg == NULL)
    {
        this->pVImg=(unsigned char*)calloc(DN_SAMPLE_IMG_NX*DN_SAMPLE_IMG_NY*DN_SAMPLE_IMG_NZ,sizeof(unsigned char));
    }
    memset(pVImg,0,sizeof(unsigned char)*DN_SAMPLE_IMG_NX*DN_SAMPLE_IMG_NY*DN_SAMPLE_IMG_NZ);//600*200*30  
    
	/* x是雷达光轴方向，y是水平左右方向，z是垂直方向 */
	/* x方向取-40~200m，每40cm一格，划分成600格 */
	/* y方向取-40~40m，每40cm一格，划分成200格 */
	/* z方向取-2.5~17.5m，每20cm一格，划分成100格 */
	/* 这个区间之外的点全部丢弃，不做处理 */
	/* 将原始点云映射到网格，每个网格取一个点，实现降采样 */
	/* 降采样后的点放在fPoints2中 */
    for(int pid=0;pid<pointNum;pid++)
    {
		/* 求当前点在网格中的坐标 */
        int ix=(fPoints1[pid*4]+DN_SAMPLE_IMG_OFFX)/DN_SAMPLE_IMG_DX; //0-240m -> -40-190m
        int iy=(fPoints1[pid*4+1]+DN_SAMPLE_IMG_OFFY)/DN_SAMPLE_IMG_DY; //-40-40m
        int iz=(fPoints1[pid*4+2]+DN_SAMPLE_IMG_OFFZ)/DN_SAMPLE_IMG_DZ;//认为地面为-1.8？ -2.5~17.5

        idtrans1[pid]=-1;
        if((ix>=0)&&(ix<DN_SAMPLE_IMG_NX)&&(iy>=0)&&(iy<DN_SAMPLE_IMG_NY)&&(iz>=0)&&(iz<DN_SAMPLE_IMG_NZ)) //DN_SAMPLE_IMG_OFFX = 0 因此只保留前半块
        {
			/* 记录当前点在网格中的索引 */
            idtrans1[pid]=iz*DN_SAMPLE_IMG_NX*DN_SAMPLE_IMG_NY+iy*DN_SAMPLE_IMG_NX+ix; //记录这个点对应的索引
            /* 如果当前网格没有访问过 */
			if(pVImg[idtrans1[pid]]==0)//没有访问过，肯定栅格内会有重复的，所以fPoints2只取第一个
            {
				/* 每个网格内取一个点，放到fPoints2中，fPoints2是fPoints1的降采样 */
				/* 每个网格内肯定有多个点，但是fPoints2只记录每个网格内的第一个点 */
                fPoints2[pntNum*4]=fPoints1[pid*4];
                fPoints2[pntNum*4+1]=fPoints1[pid*4+1];
                fPoints2[pntNum*4+2]=fPoints1[pid*4+2];
                fPoints2[pntNum*4+3]=fPoints1[pid*4+3];
				/* 建立当前点的索引副本idtrans2 */
                idtrans2[pntNum]=idtrans1[pid];

                pntNum++;
            }
			/* 标记该网格已经访问过 */
            pVImg[idtrans1[pid]]=1;
        }
    }

    /* 下面tmpPos的预设值没有任何意义，GetGndPos方法中并不使用 */ 
	//先进行地面校正
    float tmpPos[6];
    tmpPos[0]=-0.15;
    tmpPos[1]=0;
    tmpPos[2]=1;
    tmpPos[3]=0;
    tmpPos[4]=0;
    tmpPos[5]=-2.04;
    /* 求地面法向量和地面点云的中心坐标 */
    GetGndPos(tmpPos,fPoints2,pntNum); //tempPos是更新后的地面搜索点 & 平均法向量 ys
    memcpy(this->gndPos,tmpPos,6*sizeof(float));
    
    this->posFlag=1;//(this->posFlag+1)%SELF_CALI_FRAMES;

    // 3 点云矫正
    /* 对点云进行变换，将地面转到以(0,0,1)为法向量的平面，并使地面高度为0 */
    this->CorrectPoints(fPoints2,pntNum,this->gndPos);
    if(this->corPoints!=NULL)
        free(this->corPoints);
    this->corPoints=(float*)calloc(pntNum*4,sizeof(float));
    this->corNum=pntNum;
    memcpy(this->corPoints,fPoints2,4*pntNum*sizeof(float));

    // 4 粗略地面分割for地上分割
    /* 对点云进行分割，地面点标识为1，非地面点标识为0 */
    int *pLabelGnd=(int*)calloc(pntNum,sizeof(int));
    int gnum=GndSeg(pLabelGnd,fPoints2,pntNum,1.0);

    // 5 地上分割
    /* 提取非地面点，保存在fPoints3中，丢弃地面点 */
    int agnum = pntNum-gnum;
    float *fPoints3=(float*)calloc(agnum*4,sizeof(float));
    int *idtrans3=(int*)calloc(agnum,sizeof(int));
    int *pLabel2=(int*)calloc(pntNum,sizeof(int));
    int agcnt=0;//非地面点数量
    for(int ii=0;ii<pntNum;ii++)
    {
		/* fPoints3保存非地面点 */
        if(pLabelGnd[ii]==0) //上一步地面标记为1，非地面标记为0
        {
            fPoints3[agcnt*4]=fPoints2[ii*4];
            fPoints3[agcnt*4+1]=fPoints2[ii*4+1];
            fPoints3[agcnt*4+2]=fPoints2[ii*4+2];
            pLabel2[ii] = 2;
            idtrans3[agcnt]=ii; //记录fp3在fp2里面的位置
            agcnt++;
        }
        else
        {
            pLabel2[ii] = 0; //地面为1
        }
        
    }
  
    /* 对非地面点做进一步的分割 */
    /* 分割出背景和前景，前景进一步分割成各种目标
     *   背景label=1 前景0
     *   汽车、行人等运动目标，label>=10，且id唯一
     *   电线杆、花坛、交通指示牌等目标，Label在2~6之间 */
    int *pLabelAg=(int*)calloc(agnum,sizeof(int));
    if (agnum != 0)
    {
        AbvGndSeg(pLabelAg,fPoints3,agnum);  
    }
    else
    {
        std::cout << "0 above ground points!\n";
    }

    /* 分类后，背景为1，车辆等物体≥10 */
    for(int ii=0;ii<agcnt;ii++)
    {   
        if (pLabelAg[ii] >= 10)//前景为0 背景为1 物体分类后 >=10
            pLabel2[idtrans3[ii]] = 200;//pLabelAg[ii]; //前景
        else if (pLabelAg[ii] > 0)
            pLabel2[idtrans3[ii]] = 1; //背景1 -> 0
        else
        {
            pLabel2[idtrans3[ii]] = -1;
        }
    }
    
	/* 映射到pLabel2之后，背景为1，需要滤除的车辆等物体为200，未分类的-1 */
    for(int pid=0;pid<pntNum;pid++)
    {
        pVImg[idtrans2[pid]] = pLabel2[pid];
    }

    for(int pid = 0; pid < pointNum; pid++)
    {
        if(idtrans1[pid]>=0)
        {
            pLabel1[pid]= pVImg[idtrans1[pid]];
        }
        else
        {
            pLabel1[pid] = -1;//未分类
        }
    }

    free(fPoints2);
    free(idtrans1);
    free(idtrans2);
    free(idtrans3);
    free(fPoints3);
    
    free(pLabelAg);
    free(pLabelGnd);


    free(pLabel2);
}

int PCSeg::GetMainVectors(float*fPoints, int* pLabel, int pointNum)
{
    float *pClusterPoints=(float*)calloc(4*pointNum,sizeof(float));
    for(int ii=0;ii<256;ii++)
    {
        this->pnum[ii]=0;
	this->objClass[ii]=-1;
    }
    this->clusterNum=0;

    int clabel=10;
    for(int ii=10;ii<256;ii++)
    {
        int pnum=0;
        for(int pid=0;pid<pointNum;pid++)
        {
            if(pLabel[pid]==ii)
            {
                pClusterPoints[pnum*4]=fPoints[4*pid];
                pClusterPoints[pnum*4+1]=fPoints[4*pid+1];
                pClusterPoints[pnum*4+2]=fPoints[4*pid+2];
                pnum++;
            }
        }

        if(pnum>0)
        {
            SClusterFeature cf = CalOBB(pClusterPoints,pnum);

            if(cf.cls<1)
            {
                this->pVectors[clabel*3]=cf.d0[0];
                this->pVectors[clabel*3+1]=cf.d0[1];
                this->pVectors[clabel*3+2]=cf.d0[2];//朝向向量

                this->pCenters[clabel*3]=cf.center[0];
                this->pCenters[clabel*3+1]=cf.center[1];
                this->pCenters[clabel*3+2]=cf.center[2];

                this->pnum[clabel]=cf.pnum;
                for(int jj=0;jj<8;jj++)
                    this->pOBBs[clabel*8+jj]=cf.obb[jj];

                this->objClass[clabel]=cf.cls;//0 or 1 背景

                this->zs[clabel]=cf.zmax;//-cf.zmin;

                if(clabel!=ii)
                {
                    for(int pid=0;pid<pointNum;pid++)
                    {
                        if(pLabel[pid]==ii)
                        {
                            pLabel[pid]=clabel;
                        }
                    }
                }

                this->clusterNum++;
                clabel++;
            }
            else
            {
                for(int pid=0;pid<pointNum;pid++)
                {
                    if(pLabel[pid]==ii)
                    {
                        pLabel[pid]=1;
                    }
                }
            }


        }
    }

    free(pClusterPoints);

    return 0;
}

int PCSeg::EncodeFeatures(float *pFeas)
{
    int nLen=1+256+256+256*4;
    // pFeas:(1+256*6)*3
    // 0        3 ~ 3+356*3 = 257*3        275*3 ~ 257*3+256*3             3+256*6 ~ 3+256*6 + 256*4*3+12+1 
    // number    center                  每个障碍物的点数、类别、zmax           OBB 12个点
    // clusterNum
    pFeas[0]=this->clusterNum;

    memcpy(pFeas+3,this->pCenters,sizeof(float)*3*256);//重心

    for(int ii=0;ii<256;ii++)//最大容纳256个障碍物
    {
        pFeas[257*3+ii*3]=this->pnum[ii];
        pFeas[257*3+ii*3+1]=this->objClass[ii];
        pFeas[257*3+ii*3+2]=this->zs[ii];
    }

    for(int ii=0;ii<256;ii++)
    {
        for(int jj=0;jj<4;jj++)
        {
            pFeas[257*3+256*3+(ii*4+jj)*3]=this->pOBBs[ii*8+jj*2];
            pFeas[257*3+256*3+(ii*4+jj)*3+1]=this->pOBBs[ii*8+jj*2+1];
        }
    }


}
int FilterGndForPos(float* outPoints,float*inPoints,int inNum)
{
    int outNum=0;
    float dx=2;
    float dy=2;
    int x_len = 40;
    int y_len = 10;
    int nx=x_len/dx;
    int ny=2*y_len/dy;
    float offx=0,offy=-10;
    float THR=0.2;
    

    float *imgMinZ=(float*)calloc(nx*ny,sizeof(float));
    float *imgMaxZ=(float*)calloc(nx*ny,sizeof(float));
    float *imgSumZ=(float*)calloc(nx*ny,sizeof(float));
    float *imgMeanZ=(float*)calloc(nx*ny,sizeof(float));
    int *imgNumZ=(int*)calloc(nx*ny,sizeof(int));
    int *idtemp = (int*)calloc(inNum,sizeof(int));
    for(int ii=0;ii<nx*ny;ii++)
    {
        imgMinZ[ii]=10;
        imgMaxZ[ii]=-10;
        imgSumZ[ii]=0;
        imgNumZ[ii]=0;
    }

    for(int pid=0;pid<inNum;pid++)
    {
        idtemp[pid] = -1;
        if((inPoints[pid*4]<x_len)&&(inPoints[pid*4+1]>-y_len)&&(inPoints[pid*4+1]<y_len))
        {
            int idx=(inPoints[pid*4]-offx)/dx;
            int idy=(inPoints[pid*4+1]-offy)/dy;
            idtemp[pid] = idx+idy*nx;
            imgSumZ[idx+idy*nx] += inPoints[pid*4+2];
            imgNumZ[idx+idy*nx] +=1;
            if(inPoints[pid*4+2]<imgMinZ[idx+idy*nx])//寻找最小点，感觉不对。最小点有误差。
            {
                imgMinZ[idx+idy*nx]=inPoints[pid*4+2];
            }
            if(inPoints[pid*4+2]>imgMaxZ[idx+idy*nx]){
                imgMaxZ[idx+idy*nx]=inPoints[pid*4+2];
            }
        }
    }
    for(int pid=0;pid<inNum;pid++)
    {
        if(idtemp[pid]>0){
            imgMeanZ[idtemp[pid]] = float(imgSumZ[idtemp[pid]]/imgNumZ[idtemp[pid]]);
            if((imgMaxZ[idtemp[pid]]-imgMinZ[idtemp[pid]])<THR && imgMeanZ[idtemp[pid]]<-1.5&&imgNumZ[idtemp[pid]]>3){
                outPoints[outNum*4]=inPoints[pid*4];
                outPoints[outNum*4+1]=inPoints[pid*4+1];
                outPoints[outNum*4+2]=inPoints[pid*4+2];
                outPoints[outNum*4+3]=inPoints[pid*4+3];
                outNum++;
            }
        }
    }
    free(imgMinZ);
    free(imgMaxZ);
    free(imgSumZ);
    free(imgMeanZ);
    free(imgNumZ);
    free(idtemp);
    return outNum;
}

int CalNomarls(float *nvects, float *fPoints,int pointNum,float fSearchRadius)
{
    // 转换点云到pcl的格式
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    cloud->width=pointNum;
    cloud->height=1;
    cloud->points.resize(cloud->width*cloud->height);

    for (int pid=0;pid<cloud->points.size();pid++)
    {
        cloud->points[pid].x=fPoints[pid*4];
        cloud->points[pid].y=fPoints[pid*4+1];
        cloud->points[pid].z=fPoints[pid*4+2];
    }

    //调用pcl的kdtree生成方法
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud (cloud);

    //遍历每个点，获得候选地面点
    int nNum=0;
    unsigned char* pLabel = (unsigned char*)calloc(pointNum,sizeof(unsigned char));
    for(int pid=0;pid<pointNum;pid++)
    {
        if ((nNum<10)&&(pLabel[pid]==0))
        {
            SNeiborPCA npca;
            pcl::PointXYZ searchPoint;
            searchPoint.x=cloud->points[pid].x;
            searchPoint.y=cloud->points[pid].y;
            searchPoint.z=cloud->points[pid].z;

            if(GetNeiborPCA(npca,cloud,kdtree,searchPoint,fSearchRadius)>0)
            {
                for(int ii=0;ii<npca.neibors.size();ii++)
                {
                    pLabel[npca.neibors[ii]]=1;
                }

                // 地面向量筛选
                if((npca.eigenValuesPCA[1]>0.4) && (npca.eigenValuesPCA[0]<0.01))
                {
                    if((npca.eigenVectorsPCA(2,0)>0.9)|(npca.eigenVectorsPCA(2,0)<-0.9))
                    {
                        if(npca.eigenVectorsPCA(2,0)>0)
                        {
                            nvects[nNum*6]=npca.eigenVectorsPCA(0,0);
                            nvects[nNum*6+1]=npca.eigenVectorsPCA(1,0);
                            nvects[nNum*6+2]=npca.eigenVectorsPCA(2,0);

                            nvects[nNum*6+3]=searchPoint.x;
                            nvects[nNum*6+4]=searchPoint.y;
                            nvects[nNum*6+5]=searchPoint.z;
                        }
                        else
                        {
                            nvects[nNum*6]=-npca.eigenVectorsPCA(0,0);
                            nvects[nNum*6+1]=-npca.eigenVectorsPCA(1,0);
                            nvects[nNum*6+2]=-npca.eigenVectorsPCA(2,0);

                            nvects[nNum*6+3]=searchPoint.x;
                            nvects[nNum*6+4]=searchPoint.y;
                            nvects[nNum*6+5]=searchPoint.z;
                        }
                        nNum++;
                    }
                }
            }
        }
    }

    free(pLabel);


    return nNum;
}
int CalGndPos(float *gnd, float *fPoints,int pointNum,float fSearchRadius)
{
    // 初始化gnd
    //float gnd[6]={0};//(vx,vy,vz,x,y,z)
    for(int ii=0;ii<6;ii++)
    {
        gnd[ii]=0;
    }
    // 转换点云到pcl的格式
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    //去除异常点
    float height = 0;
    for(int cur = 0;cur<pointNum;cur++){
        height+=fPoints[cur*4+2];
    }
    float height_mean = height/(pointNum+0.000000001);

    for(int cur = 0;cur<pointNum;cur++){
        if(fPoints[cur*4+2]>height_mean+0.5||fPoints[cur*4+2]<height_mean-0.5){
            pointNum--;
        }
    }

    cloud->width=pointNum;
    cloud->height=1;
    cloud->points.resize(cloud->width*cloud->height);

    for (int pid=0;pid<cloud->points.size();pid++)
    {
        if(fPoints[pid*4+2]<=height_mean+0.5&&fPoints[pid*4+2]>=height_mean-0.5){
            cloud->points[pid].x=fPoints[pid*4];
            cloud->points[pid].y=fPoints[pid*4+1];
            cloud->points[pid].z=fPoints[pid*4+2];
        }
    }

    //调用pcl的kdtree生成方法
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud (cloud);

    //遍历每个点，获得候选地面点
    // float *nvects=(float*)calloc(1000*6,sizeof(float));//最多支持1000个点
    int nNum=0;
    unsigned char* pLabel = (unsigned char*)calloc(pointNum,sizeof(unsigned char));
    for(int pid=0;pid<pointNum;pid++)
    {
        if ((nNum<1000)&&(pLabel[pid]==0))
        {
            SNeiborPCA npca;
            pcl::PointXYZ searchPoint;
            searchPoint.x=cloud->points[pid].x;
            searchPoint.y=cloud->points[pid].y;
            searchPoint.z=cloud->points[pid].z;

            if(GetNeiborPCA(npca,cloud,kdtree,searchPoint,fSearchRadius)>0)
            {
                for(int ii=0;ii<npca.neibors.size();ii++)
                {
                    pLabel[npca.neibors[ii]]=1;
                }
                // 地面向量筛选
                if((npca.eigenValuesPCA[1]>0.2) && (npca.eigenValuesPCA[0]<0.1) && searchPoint.x>29.9 && searchPoint.x<40)//&&searchPoint.x>19.9
                {
                    if((npca.eigenVectorsPCA(2,0)>0.9)|(npca.eigenVectorsPCA(2,0)<-0.9))
                    {
                        if(npca.eigenVectorsPCA(2,0)>0)
                        {
                            gnd[0]+=npca.eigenVectorsPCA(0,0);
                            gnd[1]+=npca.eigenVectorsPCA(1,0);
                            gnd[2]+=npca.eigenVectorsPCA(2,0);

                            gnd[3]+=searchPoint.x;
                            gnd[4]+=searchPoint.y;
                            gnd[5]+=searchPoint.z;
                        }
                        else
                        {
                            gnd[0]+=-npca.eigenVectorsPCA(0,0);
                            gnd[1]+=-npca.eigenVectorsPCA(1,0);
                            gnd[2]+=-npca.eigenVectorsPCA(2,0);

                            gnd[3]+=searchPoint.x;
                            gnd[4]+=searchPoint.y;
                            gnd[5]+=searchPoint.z;
                        }
                        nNum++;
                    }
                }
            }
        }
    }

    free(pLabel);

    // 估计地面的高度和法相量
    if(nNum>0)
    {
        for(int ii=0;ii<6;ii++)
        {
            gnd[ii]/=nNum;
        }
        // gnd[0] *=5.0;
    }
    return nNum;
}

int GetNeiborPCA(SNeiborPCA &npca, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::KdTreeFLANN<pcl::PointXYZ> kdtree, pcl::PointXYZ searchPoint, float fSearchRadius)
{
    std::vector<float> k_dis;
    pcl::PointCloud<pcl::PointXYZ>::Ptr subCloud(new pcl::PointCloud<pcl::PointXYZ>);

    if(kdtree.radiusSearch(searchPoint,fSearchRadius,npca.neibors,k_dis)>5)
    {
        subCloud->width=npca.neibors.size();
        subCloud->height=1;
        subCloud->points.resize(subCloud->width*subCloud->height);

        for (int pid=0;pid<subCloud->points.size();pid++)
        {
            subCloud->points[pid].x=cloud->points[npca.neibors[pid]].x;
            subCloud->points[pid].y=cloud->points[npca.neibors[pid]].y;
            subCloud->points[pid].z=cloud->points[npca.neibors[pid]].z;

        }

        Eigen::Vector4f pcaCentroid;
    	pcl::compute3DCentroid(*subCloud, pcaCentroid);
	    Eigen::Matrix3f covariance;
	    pcl::computeCovarianceMatrixNormalized(*subCloud, pcaCentroid, covariance);
	    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
	    npca.eigenVectorsPCA = eigen_solver.eigenvectors();
	    npca.eigenValuesPCA = eigen_solver.eigenvalues();
        float vsum=npca.eigenValuesPCA(0)+npca.eigenValuesPCA(1)+npca.eigenValuesPCA(2);
        npca.eigenValuesPCA(0)=npca.eigenValuesPCA(0)/(vsum+0.000001);
        npca.eigenValuesPCA(1)=npca.eigenValuesPCA(1)/(vsum+0.000001);
        npca.eigenValuesPCA(2)=npca.eigenValuesPCA(2)/(vsum+0.000001);
    }
    else
    {
        npca.neibors.clear();
    }


    return npca.neibors.size();
}

/**
 * @brief 求从向量v0变换到v1的旋转矩阵
 * @param RTM 输出参数，旋转矩阵
 * @param v0 输入参数，起始向量
 * @param v1 输入参数，目标向量
 */
int GetRTMatrix(float *RTM, float *v0, float *v1) // v0 gndpos v1:001垂直向上
{
    // 归一化
    float nv0=sqrt(v0[0]*v0[0]+v0[1]*v0[1]+v0[2]*v0[2]);
    v0[0]/=(nv0+0.000001);
    v0[1]/=(nv0+0.000001);
    v0[2]/=(nv0+0.000001);

    float nv1=sqrt(v1[0]*v1[0]+v1[1]*v1[1]+v1[2]*v1[2]);
    v1[0]/=(nv1+0.000001);
    v1[1]/=(nv1+0.000001);
    v1[2]/=(nv1+0.000001);

    // 叉乘
    //  ^ v2  ^ v0
    //  |   /
    //  |  /
    //  |-----> v1
    //
    float v2[3]; // 叉乘的向量代表了v0旋转到v1的旋转轴，因为叉乘结果垂直于v0、v1构成的平面
    v2[0]=v0[1]*v1[2]-v0[2]*v1[1];
    v2[1]=v0[2]*v1[0]-v0[0]*v1[2];
    v2[2]=v0[0]*v1[1]-v0[1]*v1[0];

    // 正余弦
    float cosAng=0,sinAng=0;
    cosAng=v0[0]*v1[0]+v0[1]*v1[1]+v0[2]*v1[2]; //a.b = |a||b|cos theta
    sinAng=sqrt(1-cosAng*cosAng);

    // 计算旋转矩阵
    RTM[0]=v2[0]*v2[0]*(1-cosAng)+cosAng;
    RTM[4]=v2[1]*v2[1]*(1-cosAng)+cosAng;
    RTM[8]=v2[2]*v2[2]*(1-cosAng)+cosAng;

    RTM[1]=RTM[3]=v2[0]*v2[1]*(1-cosAng);
    RTM[2]=RTM[6]=v2[0]*v2[2]*(1-cosAng);
    RTM[5]=RTM[7]=v2[1]*v2[2]*(1-cosAng);

    RTM[1]+=(v2[2])*sinAng;
    RTM[2]+=(-v2[1])*sinAng;
    RTM[3]+=(-v2[2])*sinAng;

    RTM[5]+=(v2[0])*sinAng;
    RTM[6]+=(v2[1])*sinAng;
    RTM[7]+=(-v2[0])*sinAng;

    return 0;
}

/**
 * @brief 对点云进行变换，将地面转到以(0,0,1)为法向量的平面，并使地面高度为0
 * @param fPoints 输入输出参数，完整点云
 * @param pointNum 入参数，点云点数
 * @param gndPos 输入参数，地面法向量和地面点云中心坐标
 */
int PCSeg::CorrectPoints(float *fPoints,int pointNum,float *gndPos)
{
    float RTM[9];
    float gndHeight=0;
    float znorm[3]={0,0,1};
    float tmp[3];

    /* 求从向量gndPos变换到znorm的旋转矩阵 */
    GetRTMatrix(RTM,gndPos,znorm); // RTM 由当前地面法向量gnd，将平面转到以(0,0,1)为法向量的平面 sy
    /* RTM矩阵的序号排列如下：
    /* | 0 3 6 |
    /* | 1 4 7 |
    /* | 2 5 8 | */

    /* 求变换后地面中心点的Z坐标 */
    gndHeight = RTM[2]*gndPos[3]+RTM[5]*gndPos[4]+RTM[8]*gndPos[5];

    /* 对点云进行整体变换使，将地面转到以(0,0,1)为法向量的平面，并使地面高度为0 */
    for(int pid=0;pid<pointNum;pid++)
    {
        tmp[0]=RTM[0]*fPoints[pid*4]+RTM[3]*fPoints[pid*4+1]+RTM[6]*fPoints[pid*4+2];
        tmp[1]=RTM[1]*fPoints[pid*4]+RTM[4]*fPoints[pid*4+1]+RTM[7]*fPoints[pid*4+2];
        tmp[2]=RTM[2]*fPoints[pid*4]+RTM[5]*fPoints[pid*4+1]+RTM[8]*fPoints[pid*4+2]-gndHeight; //将地面高度变换到0 sy

        fPoints[pid*4]=tmp[0];
        fPoints[pid*4+1]=tmp[1];
        fPoints[pid*4+2]=tmp[2];
    }

    return 0;
}

/**
 * @brief 对点云进行分割，区分出背景和车辆等运动目标
 *   背景label=1 前景0
 *   汽车、行人等运动目标，label>=10，且id唯一
 *   电线杆、花坛、交通指示牌等目标，Label在2~6之间
 * @param pLabel 输出参数，点云中每个点的类型
 * @param fPoints 输入参数，地上点云，已经剔除掉地面点云
 * @param pointNum 输入参数，点云点数
 */
int AbvGndSeg(int *pLabel, float *fPoints, int pointNum)
{
    // 转换点云到pcl的格式
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    cloud->width=pointNum;
    cloud->height=1;
    cloud->points.resize(cloud->width*cloud->height);

    for (int pid=0;pid<cloud->points.size();pid++)
    {
        cloud->points[pid].x=fPoints[pid*4];
        cloud->points[pid].y=fPoints[pid*4+1];
        cloud->points[pid].z=fPoints[pid*4+2];
    }

    //调用pcl的kdtree生成方法
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud (cloud);

    /* 进行背景和前景的分割 */
    SegBG(pLabel,cloud,kdtree,0.5); //背景label=1 前景0

    /* 进行前景目标的分割 */
    /* 对于汽车、行人等运动目标，label>=10，且id唯一 */
    /* 对于电线杆、花坛、交通指示牌等目标，Label在2~6之间*/
    SegObjects(pLabel,cloud,kdtree,0.7);

    /* 将位于背景之后的点强制标识为背景 */
    FreeSeg(fPoints,pLabel,pointNum);
    return 0;
}

/**
 * @brief 对点云进行分割，区分出背景和前景，所谓前景是指车辆等近距离和运动目标
 *   选取高度大于4米的点作为背景种子点，通过迭代搜索临近点的方式生长出背景点云，剩下的就是前景
 * @param pLabel 输出参数，点云中每个点的类型，背景为1，前景为0
 * @param cloud 输入参数，待分割点云，不包含地面
 * @param kdtree 输入参数，待分割点云的KDtree
 * @param fSearchRadius 输入参数，搜索半径
 */
int SegBG(int *pLabel, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::KdTreeFLANN<pcl::PointXYZ> &kdtree, float fSearchRadius)
{
    //从较高的一些点开始从上往下增长，这样能找到一些类似树木、建筑物这样的高大背景
    // clock_t t0,t1,tsum=0;
    int pnum=cloud->points.size();
    pcl::PointXYZ searchPoint;

    /* 取高度4~6米之间的点作为种子点 */
    /* 所有高度超过4米的点都被设定为背景 */
    // 初始化种子点
    std::vector<int> seeds;
    for (int pid=0;pid<pnum;pid++)
    {
        if(cloud->points[pid].z>4)
        {
            pLabel[pid]=1;
            if (cloud->points[pid].z<6)
            {
                seeds.push_back(pid);
            }
        }
        else
        {
            pLabel[pid]=0;
        }

    }

    // 区域增长
    while(seeds.size()>0)
    {
        /* 以种子点为起始，搜索临近点 */
        int sid = seeds[seeds.size()-1];
        seeds.pop_back();

        std::vector<float> k_dis;
        std::vector<int> k_inds;
        // t0=clock();
        if (cloud->points[sid].x<44.8)
            kdtree.radiusSearch(sid,fSearchRadius,k_inds,k_dis);
        else
            kdtree.radiusSearch(sid,1.5*fSearchRadius,k_inds,k_dis);

        /* 将搜索到的临近点也标识为背景，并添加到种子点中，进行背景生长过程 */
        for(int ii=0;ii<k_inds.size();ii++)
        {
            if(pLabel[k_inds[ii]]==0)
            {
                pLabel[k_inds[ii]]=1;
                if(cloud->points[k_inds[ii]].z>0.2)//地面20cm以下不参与背景分割，以防止错误的地面点导致过度分割
                {
                    seeds.push_back(k_inds[ii]);
                }
            }
        }

    }
    return 0;

}

/**
 * @brief 以给定的种子点为起点，通过迭代搜索临近点的方式找到属于同一个目标的点，即同属一个簇的点
 *   并赋予一个唯一的标签ID，最后返回目标的尺寸特征
 * @param pLabel 输出参数，点云中每个点的类型，背景为1，前景为0，前进目标＞10
 * @param seedId 输入参数，种子点的id
 * @param labelId 输入参数，标签ID
 * @param cloud 输入参数，待分割点云，不包含地面
 * @param kdtree 输入参数，待分割点云的KDtree
 * @param fSearchRadius 输入参数，搜索半径
 * @param thrHeight 输入参数，高度阈值，低于该阈值的点不参与分割
 * @return 返回目标的尺寸特征，即xMin/xMax，yMin/yMax，zMin/zMax
 */
SClusterFeature FindACluster(int *pLabel, int seedId, int labelId, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::KdTreeFLANN<pcl::PointXYZ> &kdtree, float fSearchRadius, float thrHeight)
{
    // 初始化种子
    std::vector<int> seeds;
    seeds.push_back(seedId);
    pLabel[seedId]=labelId;
    // int cnum=1;//当前簇里的点数

    SClusterFeature cf;
    cf.pnum=1;
    cf.xmax=-2000;
    cf.xmin=2000;
    cf.ymax=-2000;
    cf.ymin=2000;
    cf.zmax=-2000;
    cf.zmin=2000;
    cf.zmean=0;

    // 区域增长
    while(seeds.size()>0) //实际上是种子点找了下k近邻，然后k近邻再纳入种子点 sy
    {
        int sid=seeds[seeds.size()-1];
        seeds.pop_back();

        pcl::PointXYZ searchPoint;
        searchPoint.x=cloud->points[sid].x;
        searchPoint.y=cloud->points[sid].y;
        searchPoint.z=cloud->points[sid].z;

        // 特征统计
        {
            if(searchPoint.x>cf.xmax) cf.xmax=searchPoint.x;
            if(searchPoint.x<cf.xmin) cf.xmin=searchPoint.x;

            if(searchPoint.y>cf.ymax) cf.ymax=searchPoint.y;
            if(searchPoint.y<cf.ymin) cf.ymin=searchPoint.y;

            if(searchPoint.z>cf.zmax) cf.zmax=searchPoint.z;
            if(searchPoint.z<cf.zmin) cf.zmin=searchPoint.z;

            cf.zmean+=searchPoint.z;
        }

        std::vector<float> k_dis;
        std::vector<int> k_inds;

        if(searchPoint.x<44.8)
            kdtree.radiusSearch(searchPoint,fSearchRadius,k_inds,k_dis);
        else
            kdtree.radiusSearch(searchPoint,2*fSearchRadius,k_inds,k_dis);

        for(int ii=0;ii<k_inds.size();ii++)
        {
            if(pLabel[k_inds[ii]]==0)
            {
                pLabel[k_inds[ii]]=labelId;
                // cnum++;
                cf.pnum++;
                if(cloud->points[k_inds[ii]].z>thrHeight)//地面60cm以下不参与分割
                {
                    seeds.push_back(k_inds[ii]);
                }
            }
        }
    }
    cf.zmean/=(cf.pnum+0.000001);
    return cf;
}

/**
 * @brief 对前景点云进行分割，所谓前景是指车辆等近距离和运动目标
 *   确定是车辆等运动前景目标的，为每个独立的目标赋予一个独立的id，且id≥10
 *   对于尺寸特征不满足运动目标的前景点，将其类型设置为2~6的区间
 * @param pLabel 输出参数，点云中每个点的类型，背景为1，前景为0
 * @param cloud 输入参数，待分割点云，不包含地面
 * @param kdtree 输入参数，待分割点云的KDtree
 * @param fSearchRadius 输入参数，搜索半径
 * @param 返回找到的运动目标总数，即id≥10的目标总数
 */
int SegObjects(int *pLabel, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::KdTreeFLANN<pcl::PointXYZ> &kdtree, float fSearchRadius)
{
    int pnum=cloud->points.size();
    int labelId=10;//object编号从10开始

    //遍历每个非背景点，若高度大于0.4则寻找一个簇，并给一个编号(>10)
    for(int pid=0;pid<pnum;pid++)
    {
        if(pLabel[pid]==0)
        {
            /* 每找到一个目标，赋予一个新的id，id从10开始编号 */
            if(cloud->points[pid].z>0.4)//高度阈值
            {
                /* 对于高度超过40cm的前景点，通过迭代搜索临近点的方式找到属于同一个目标的点，实现分割 */
                SClusterFeature cf = FindACluster(pLabel,pid,labelId,cloud,kdtree,fSearchRadius,0);
                int isBg=0;

                /* 根据目标的尺寸特征进行分类 */
                // cluster 分类
                float dx=cf.xmax-cf.xmin;
                float dy=cf.ymax-cf.ymin;
                float dz=cf.zmax-cf.zmin;
                
                /* 求x坐标的最小值，即最近目标 */
                float cx=10000;
                for(int ii=0;ii<pnum;ii++)
                {
                    if (cx>cloud->points[pid].x)
                    {
                        cx=cloud->points[pid].x;
                    }
                }

                /* 通过目标的尺寸特征排除不应该是汽车等运动目标的点 */
                if((dx>15)||(dy>15)||((dx>10)&&(dy>10)))// 太大
                {
                    isBg=2;
                }
                else if(((dx>6)||(dy>6))&&(cf.zmean < 1.5))//长而过低，例如花坛
                {
                    isBg = 3;//1;//2;
                }
                else if(((dx<1.5)&&(dy<1.5))&&(cf.zmax>2.5))//小而过高，例如电线杆
                {
                    isBg = 4;
                }
                else if(cf.pnum<5 || (cf.pnum<10 && cx<50)) //点太少
                {
                    isBg=5;
                }
                else if((cf.zmean>3)||(cf.zmean<0.3))//太高或太低，例如交通指示牌
                {
                    isBg = 6;//1;
                }
                /* 对于尺寸特征上不满足运动目标的前景点，将其类型设置为2~6的区间 */
                if(isBg>0)
                {
                    for(int ii=0;ii<pnum;ii++)
                    {
                        if(pLabel[ii]==labelId)
                        {
                            pLabel[ii]=isBg;
                        }
                    }
                }
                else /* 为每个目标赋予独一无二的id */
		        {
                    labelId++;
                }
            }
        }
    }
    return labelId-10;
}

int CompleteObjects(int *pLabel, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::KdTreeFLANN<pcl::PointXYZ> &kdtree, float fSearchRadius)
{
    int pnum=cloud->points.size();
    int *tmpLabel = (int*)calloc(pnum,sizeof(int));
    for(int pid=0;pid<pnum;pid++)
    {
        if(pLabel[pid]!=2)
        {
            tmpLabel[pid]=1;
        }
    }

    //遍历每个非背景点，若高度大于1则寻找一个簇，并给一个编号(>10)
    for(int pid=0;pid<pnum;pid++)
    {
        if(pLabel[pid]>=10)
        {
            FindACluster(tmpLabel,pid,pLabel[pid],cloud,kdtree,fSearchRadius,0);
        }
    }
    for(int pid=0;pid<pnum;pid++)
    {
        if((pLabel[pid]==2)&&(tmpLabel[pid]>=10))
        {
            pLabel[pid]=tmpLabel[pid];
        }
    }

    free(tmpLabel);

    return 0;
}
int ExpandObjects(int *pLabel, float* fPoints, int pointNum, float fSearchRadius)
{
    int *tmpLabel = (int*)calloc(pointNum,sizeof(int));
    for(int pid=0;pid<pointNum;pid++)
    {
        if(pLabel[pid]<10)
        {
            tmpLabel[pid]=1;
        }
    }

    // 转换点云到pcl的格式
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    cloud->width=pointNum;
    cloud->height=1;
    cloud->points.resize(cloud->width*cloud->height);

    for (int pid=0;pid<cloud->points.size();pid++)
    {
        cloud->points[pid].x=fPoints[pid*4];
        cloud->points[pid].y=fPoints[pid*4+1];
        cloud->points[pid].z=fPoints[pid*4+2];
    }


    //调用pcl的kdtree生成方法
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud (cloud);
    std::vector<float> k_dis;
    std::vector<int> k_inds;

    for(int pid=0;pid<cloud->points.size();pid++)
    {
        if((pLabel[pid]>=10)&&(tmpLabel[pid]==0))
        {
            kdtree.radiusSearch(pid,fSearchRadius,k_inds,k_dis);

            for(int ii=0;ii<k_inds.size();ii++)
            {
                if((pLabel[k_inds[ii]]<10)&&(cloud->points[k_inds[ii]].z>0.2))
                {
                    pLabel[k_inds[ii]]=pLabel[pid];
                }
            }
        }
    }

    free(tmpLabel);

    return 0;
}
int ExpandBG(int *pLabel, float* fPoints, int pointNum, float fSearchRadius)
{
    int *tmpLabel = (int*)calloc(pointNum,sizeof(int));
    for(int pid=0;pid<pointNum;pid++)
    {
        if((pLabel[pid]>0)&&(pLabel[pid]<10))
        {
            tmpLabel[pid]=1;
        }
    }

    // 转换点云到pcl的格式
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    cloud->width=pointNum;
    cloud->height=1;
    cloud->points.resize(cloud->width*cloud->height);

    for (int pid=0;pid<cloud->points.size();pid++)
    {
        cloud->points[pid].x=fPoints[pid*4];
        cloud->points[pid].y=fPoints[pid*4+1];
        cloud->points[pid].z=fPoints[pid*4+2];
    }


    //调用pcl的kdtree生成方法
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud (cloud);
    std::vector<float> k_dis;
    std::vector<int> k_inds;

    for(int pid=0;pid<cloud->points.size();pid++)
    {
        if((tmpLabel[pid]==1))
        {
            kdtree.radiusSearch(pid,fSearchRadius,k_inds,k_dis);

            for(int ii=0;ii<k_inds.size();ii++)
            {
                if((pLabel[k_inds[ii]]==0)&&(cloud->points[k_inds[ii]].z>0.4))
                {
                    pLabel[k_inds[ii]]=pLabel[pid];
                }
            }
        }
    }

    free(tmpLabel);

    return 0;
}

/**
 * @brief 求极坐标下360度网格，每一格中最近的背景点
 * @param pFreeDis 输出参数，待分割点云，不包含地面
 * @param fPoints 输入参数，待分割点云，不包含地面
 * @param pLabel 输出参数，点云中每个点的类型，背景为1，前景为0
 * @param pointNum 输入参数，点云中的点数
 * @return 总是返回0
 */
int CalFreeRegion(float *pFreeDis, float *fPoints,int *pLabel,int pointNum)
{
    int da=FREE_DELTA_ANG;
    int thetaId;
    float dis;

    /* 在极坐标下，将X-Y平面划分成360个网格，每个网格一度 */
    for(int ii=0;ii<FREE_ANG_NUM;ii++)
    {
        pFreeDis[ii]=20000;
    }
    /* 找到每一度中的最近点，记录在pFreeDis中 */
    for(int pid=0;pid<pointNum;pid++)
    {
        /* 从高度低于4.5米的背景点中找出每一度网格中的最近点 */
        if((pLabel[pid]==1)&&(fPoints[pid*4+2]<4.5))
        {
            /* 计算弧长 */
            dis=fPoints[pid*4]*fPoints[pid*4]+fPoints[pid*4+1]*fPoints[pid*4+1];
            /* 计算度数，以及所在网格 */
            thetaId=((atan2f(fPoints[pid*4+1],fPoints[pid*4])+FREE_PI)/FREE_DELTA_ANG);
            thetaId=thetaId%FREE_ANG_NUM;

            if(pFreeDis[thetaId]>dis)
            {
                pFreeDis[thetaId]=dis;
            }
        }
    }

    return 0;
}

/**
 * @brief 将位于背景之后的点强制标识为背景
 * @param fPoints 输入参数，待分割点云，不包含地面
 * @param pLabel 输出参数，点云中每个点的类型，背景为1，前景为0
 * @param pointNum 输入参数，点云中的点数
 * @return 
 */
int FreeSeg(float *fPoints,int *pLabel,int pointNum)
{
    float *pFreeDis = (float*)calloc(FREE_ANG_NUM,sizeof(float));
    int thetaId;
    float dis;
    
    /* FREE_ANG_NUM 360
     * FREE_PI (3.14159265)
     * FREE_DELTA_ANG (FREE_PI*2/FREE_ANG_NUM) */

    /* 计算极坐标下360度网格，记录每一度网格中的最近背景点的距离 */
    CalFreeRegion(pFreeDis,fPoints,pLabel,pointNum);

    /* 如果当前点位于背景的背后，则强制标识为背景 */
    for(int pid=0;pid<pointNum;pid++)
    {
        //! 需要考虑的是，被前景遮住的背景和被背景点遮住的前景，那个需要处理？
        {
            dis=fPoints[pid*4]*fPoints[pid*4]+fPoints[pid*4+1]*fPoints[pid*4+1];
            thetaId=((atan2f(fPoints[pid*4+1],fPoints[pid*4])+FREE_PI)/FREE_DELTA_ANG);
            thetaId=thetaId%FREE_ANG_NUM;

            if(pFreeDis[thetaId]<dis) //表示在背后
            {
                pLabel[pid]=1;
            }
        }
    }
    if (pFreeDis != NULL)
        free(pFreeDis);
}

/**
 * @brief 对点云进行地面分割，将地面点标识为1，非地面点标识为0
 * @param fPoints 输入输出参数，完整点云
 * @param pointNum 入参数，点云点数
 * @param gndPos 输入参数，地面法向量和地面点云中心坐标
 */
int GndSeg(int* pLabel,float *fPoints,int pointNum,float fSearchRadius)
{
    int gnum=0;
    
    /* GND_IMG_NX1 24
     * GND_IMG_NY1 20
     * GND_IMG_DX1 4
     * GND_IMG_DY1 4
     * GND_IMG_OFFX1 40
     * GND_IMG_OFFY1 40 */
     
    /* x方向取-40~56m，每4m一格，划分成24格 */
	/* y方向取-40~40m，每4m一格，划分成20格 */
	/* 这个区间之外的点全部丢弃，不做处理 */
    // 根据BV最低点来缩小地面点搜索范围----2
    float *pGndImg1 = (float*)calloc(GND_IMG_NX1*GND_IMG_NY1,sizeof(float));
    int *tmpLabel1 = (int*)calloc(pointNum,sizeof(int));
    
    /* 将点云映射到网格，找到每个网格的最低高度 */
    for(int ii=0;ii<GND_IMG_NX1*GND_IMG_NY1;ii++)
    {
        pGndImg1[ii]=100;
    }
    for(int pid=0;pid<pointNum;pid++)//求最低点图像
    {
        /* 将点云中的点映射到网格中，获得网格坐标 */
        int ix= (fPoints[pid*4]+GND_IMG_OFFX1)/(GND_IMG_DX1+0.000001);
        int iy= (fPoints[pid*4+1]+GND_IMG_OFFY1)/(GND_IMG_DY1+0.000001);
        if(ix<0 || ix>=GND_IMG_NX1 || iy<0 || iy>=GND_IMG_NY1)
        {
            tmpLabel1[pid]=-1;
            continue;
        }

        /* 建立点的序号到网格坐标的索引 */
        int iid=ix+iy*GND_IMG_NX1;
        tmpLabel1[pid]=iid;

        /* 找到每个网格的最低高度 */
        if(pGndImg1[iid]>fPoints[pid*4+2])//找最小高度
        {
            pGndImg1[iid]=fPoints[pid*4+2];
        }
    }

    /* 每个格子中，相对于最低高度不超过0.5米的点初步标识为地面点 */
    int pnum=0;
    for(int pid=0;pid<pointNum;pid++)
    {
        if(tmpLabel1[pid]>=0)
        {
            if(pGndImg1[tmpLabel1[pid]]+0.5>fPoints[pid*4+2])// 相对高度限制 即pid所在点对应的格子里面高度不超过0.5，标记为1
            {
                {pLabel[pid]=1;
                pnum++;}
            }
        }
    }

    free(pGndImg1);
    free(tmpLabel1);


    // 绝对高度限制
    for(int pid=0;pid<pointNum;pid++)
    {
        /* 对于相对高度不超过0.5米的点 */
        if(pLabel[pid]==1)
        {
            /* 剔除绝对高度超过1米的点 */
            if(fPoints[pid*4+2]>1)//for all cases 即使这个点所在格子高度差小于0.5，但绝对高度大于1，那也不行
            {
                pLabel[pid]=0;
            }
            /* 在深度15米的范围内，绝对高度不能超过0.5米 */
            else if(fPoints[pid*4]*fPoints[pid*4]+fPoints[pid*4+1]*fPoints[pid*4+1]<225)// 15m内，强制要求高度小于0.5
            {
                if(fPoints[pid*4+2]>0.5)
                {
                    pLabel[pid]=0;

                }
            }
        }
        else /* 在深度20米的范围内，相对高度高度超过0.5米，但是绝对高度低于0.2米的也强制标识为地面 */
        {   
            if(fPoints[pid*4]*fPoints[pid*4]+fPoints[pid*4+1]*fPoints[pid*4+1]<400)// 20m内，0.2以下强制设为地面
            {
                if(fPoints[pid*4+2]<0.2)
                {
                    pLabel[pid]=1;
                }
            }
        }
    }

    // 平面拟合与剔除
    float zMean=0;
    gnum=0;
    /* 找到深度20米范围内的地面高度均值 */
    for(int pid=0;pid<pointNum;pid++)
    {
        if(pLabel[pid]==1 && fPoints[pid*4]*fPoints[pid*4]+fPoints[pid*4+1]*fPoints[pid*4+1]<400)
        {
            zMean+=fPoints[pid*4+2];
            gnum++;
        }
    }
    zMean/=(gnum+0.0001);
    /* 剔除高出地面均值0.4米的点 */
    for(int pid=0;pid<pointNum;pid++)
    {
        if(pLabel[pid]==1)
        {
            if(fPoints[pid*4]*fPoints[pid*4]+fPoints[pid*4+1]*fPoints[pid*4+1]<400 && fPoints[pid*4+2]>zMean+0.4)
                pLabel[pid]=0;
        }
    }

    // 个数统计
    gnum=0;
    for(int pid=0;pid<pointNum;pid++)
    {
        if(pLabel[pid]==1)
        {
            gnum++;
        }
    }

    return gnum;
}

SClusterFeature CalBBox(float *fPoints,int pointNum)
{
    SClusterFeature cf;
     // 转换点云到pcl的格式
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    cloud->width=pointNum;
    cloud->height=1;
    cloud->points.resize(cloud->width*cloud->height);

    for (int pid=0;pid<cloud->points.size();pid++)
    {
        cloud->points[pid].x=fPoints[pid*4];
        cloud->points[pid].y=fPoints[pid*4+1];
        cloud->points[pid].z=0;//fPoints[pid*4+2];
    }

    //调用pcl的kdtree生成方法
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud (cloud);

    //计算特征向量和特征值
    Eigen::Vector4f pcaCentroid;
    pcl::compute3DCentroid(*cloud, pcaCentroid);
    Eigen::Matrix3f covariance;
    pcl::computeCovarianceMatrixNormalized(*cloud, pcaCentroid, covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);

    Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
    Eigen::Vector3f eigenValuesPCA = eigen_solver.eigenvalues();
    float vsum = eigenValuesPCA(0)+eigenValuesPCA(1)+eigenValuesPCA(2);
    eigenValuesPCA(0) = eigenValuesPCA(0)/(vsum+0.000001);
    eigenValuesPCA(1) = eigenValuesPCA(1)/(vsum+0.000001);
    eigenValuesPCA(2) = eigenValuesPCA(2)/(vsum+0.000001);

    cf.d0[0]=eigenVectorsPCA(0,2);
    cf.d0[1]=eigenVectorsPCA(1,2);
    cf.d0[2]=eigenVectorsPCA(2,2);

    cf.d1[0]=eigenVectorsPCA(0,1);
    cf.d1[1]=eigenVectorsPCA(1,1);
    cf.d1[2]=eigenVectorsPCA(2,1);

    cf.center[0]=pcaCentroid(0);
    cf.center[1]=pcaCentroid(1);
    cf.center[2]=pcaCentroid(2);

    cf.pnum=pointNum;

    return cf;
}

SClusterFeature CalOBB(float *fPoints,int pointNum)
{
    SClusterFeature cf;
    cf.pnum=pointNum;

    float *hoff=(float*)calloc(20000,sizeof(float));
    int hnum=0;

    float coef_as[180],coef_bs[180];
    for(int ii=0;ii<180;ii++)
    {
        coef_as[ii]=cos(ii*0.5/180*FREE_PI);
        coef_bs[ii]=sin(ii*0.5/180*FREE_PI);
    }

    float *rxy=(float*)calloc(pointNum*2,sizeof(float));

    float area=-1000;//1000000;
    int ori=-1;
    int angid=0;
    for(int ii=0;ii<180;ii++)
    {

        float a_min=10000,a_max=-10000;
        float b_min=10000,b_max=-10000;
        for(int pid=0;pid<pointNum;pid++)
        {
            float val=fPoints[pid*4]*coef_as[ii]+fPoints[pid*4+1]*coef_bs[ii];// x*cos + y*sin 这是把每个点旋转一下，然后求旋转后点云的xy最大最小值 sy
            if(a_min>val)
                a_min=val;
            if(a_max<val)
                a_max=val;
            rxy[pid*2]=val;

            val=fPoints[pid*4+1]*coef_as[ii]-fPoints[pid*4]*coef_bs[ii];
            if(b_min>val)
                b_min=val;
            if(b_max<val)
                b_max=val;
            rxy[pid*2+1]=val;
        }

        float weights=0;

        hnum=(a_max-a_min)/0.05;
        for(int ih=0;ih<hnum;ih++)
        {
            hoff[ih]=0;
        }
        for(int pid=0;pid<pointNum;pid++)
        {
            int ix0=(rxy[pid*2]-a_min)*20;
            hoff[ix0]+=1;
        }

        int mh=-1;
        for(int ih=0;ih<hnum;ih++)
        {
            if(hoff[ih]>weights)
            {
              weights=hoff[ih];
              mh=ih;
            }
        }

        if (mh>0)
        {
            if(mh*3<hnum)
            {
                a_min=a_min+mh*0.05/2;
            }
            else if((hnum-mh)*3<hnum)
            {
                a_max=a_max-(hnum-mh)*0.05/2;
            }
        }

        // --y
        float weights1=0;
        hnum=(b_max-b_min)/0.05;
        for(int ih=0;ih<hnum;ih++)
        {
            hoff[ih]=0;
        }
        for(int pid=0;pid<pointNum;pid++)
        {
            int iy0=(rxy[pid*2+1]-b_min)*20;
            hoff[iy0]+=1;
        }
        int mh1=-1;
        for(int ih=0;ih<hnum;ih++)
        {
            if(hoff[ih]>weights1)
            {
                weights1=hoff[ih];
                mh1=ih;
            }
        }
        if(mh1>0)
        {
            if(mh1*3<hnum)
            {
                b_min=b_min+mh1*0.05/2;
            }
            else if((hnum-mh1)*3<hnum)
            {
                b_max=b_max-(hnum-mh1)*0.05/2;
            }
        }

        if(weights < weights1)
        {
            weights=weights1;
        }

        if(weights>area)//(b_max-b_min)*(a_max-a_min)<area)
        {
            area=weights;//(b_max-b_min)*(a_max-a_min);
            ori=ii;
            angid=ii;

            cf.obb[0]=a_max*coef_as[ii]-b_max*coef_bs[ii];
            cf.obb[1]=a_max*coef_bs[ii]+b_max*coef_as[ii];

            cf.obb[2]=a_max*coef_as[ii]-b_min*coef_bs[ii];
            cf.obb[3]=a_max*coef_bs[ii]+b_min*coef_as[ii];

            cf.obb[4]=a_min*coef_as[ii]-b_min*coef_bs[ii];
            cf.obb[5]=a_min*coef_bs[ii]+b_min*coef_as[ii];

            cf.obb[6]=a_min*coef_as[ii]-b_max*coef_bs[ii];
            cf.obb[7]=a_min*coef_bs[ii]+b_max*coef_as[ii];

            cf.center[0]=(a_max+a_min)/2*coef_as[ii]-(b_max+b_min)/2*coef_bs[ii];
            cf.center[1]=(a_max+a_min)/2*coef_bs[ii]+(b_max+b_min)/2*coef_as[ii];
            cf.center[2]=0;

            cf.d0[0]=coef_as[ii]; //相当于朝向角
            cf.d0[1]=coef_bs[ii];
            cf.d0[2]=0;

            cf.xmin=a_min;
            cf.xmax=a_max;
            cf.ymin=b_min;
            cf.ymax=b_max;
        }
    }

    //center
    cf.center[0]=0;
    cf.center[1]=0;
    cf.center[2]=0;
    for(int pid=0;pid<pointNum;pid++)
    {
        cf.center[0]+=fPoints[pid*4];
        cf.center[1]+=fPoints[pid*4+1];
        cf.center[2]+=fPoints[pid*4+2];
    }

    cf.center[0]/=(pointNum+0.0001);
    cf.center[1]/=(pointNum+0.0001);
    cf.center[2]/=(pointNum+0.0001);

    //z
    float z_min=10000,z_max=-10000;
    for(int pid=0;pid<pointNum;pid++)
    {
        if(fPoints[pid*4+2]>z_max)
            z_max=fPoints[pid*4+2];
        if(fPoints[pid*4+2]<z_min)
            z_min=fPoints[pid*4+2];
    }

    cf.zmin=z_min;
    cf.zmax=z_max;

    // 分类
    cf.cls=0;
    float dx=cf.xmax-cf.xmin;
    float dy=cf.ymax-cf.ymin;
    float dz=cf.zmax-cf.zmin;
    if((dx>15)||(dy>15))// 太大
    {
        cf.cls=1;//bkg
    }
    else if((dx>4)&&(dy>4))//太大
    {
        cf.cls=1;
    }
    else if((dz<0.5||cf.zmax<1) && (dx>3 || dy>3))//too large
    {
        cf.cls=1;
    }
    else if(cf.zmax>3 && (dx<0.5 || dy<0.5))//too small
    {
        cf.cls=1;
    }
    else if(dx<0.5 && dy<0.5)//too small
    {
        cf.cls=1;
    }
    else if(dz<2&&(dx>6||dy>6||dx/dy>5||dy/dx>5))//too small
    {
        //cf.cls=1;
    }
    else if((dz<4)&&(dx>3)&&(dy>3))//太大
    {
        //cf.cls=1;
    }
    
    else if(cf.center[0]>45 && (dx>=3 || dy>=3 || dx*dy>2.3))//太da
    {
        //cf.cls=1;
    }
    else if(dx/dy>5||dy/dx>5) //太窄
    {
        //cf.cls=0;
    }
    else if((dz<2.5)&&((dx/dy>3)||(dy/dx>3)))
    {
        //cf.cls=0;
    }
    else if((dz<2.5)&&(dx>2.5)&&(dy>2.5))
    {
        //cf.cls=1;
    }


    free(rxy);
    free(hoff);
    return cf;
}
