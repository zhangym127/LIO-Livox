#ifndef LIO_LIVOX_CERESFUNC_H
#define LIO_LIVOX_CERESFUNC_H
#include <ceres/ceres.h>
#include <glog/logging.h>
#include <utility>
#include <pthread.h>
#include <unordered_map>
#include "sophus/so3.hpp"
#include "IMUIntegrator/IMUIntegrator.h"

const int NUM_THREADS = 4;

/** \brief Residual Block Used for marginalization
 */
struct ResidualBlockInfo
{
	/**
	 * @brief 构造用于边缘化的残差块
	 * @param _cost_function 代价函数，或说边缘化因子
	 * @param _loss_function 鲁邦核函数指针
	 * @param _parameter_blocks 参数块，即para_PR或para_VBias
	 * @param _drop_set 需要丢弃的参数块序号
	 */
	ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
					: cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(std::move(_parameter_blocks)), drop_set(std::move(_drop_set)) {}

	/**
	 * @brief 对残差块进行优化
	 */
	void Evaluate(){

		/* 初始化残差 */
		residuals.resize(cost_function->num_residuals());

		/* 根据参数块的数量初始化雅可比长度 */
		std::vector<int> block_sizes = cost_function->parameter_block_sizes();
		raw_jacobians = new double *[block_sizes.size()];
		jacobians.resize(block_sizes.size());

		/* 用每个参数块的size初始化每个雅可比的size */
		for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
		{
			jacobians[i].resize(cost_function->num_residuals(), block_sizes[i]);
			raw_jacobians[i] = jacobians[i].data();
		}

		/* 调用代价函数进行优化 */
		cost_function->Evaluate(parameter_blocks.data(), residuals.data(), raw_jacobians);

		/* 使用鲁棒核函数对残差的比例进行调节，防止错误的测量值带偏优化结果 */
		if (loss_function)
		{
			double residual_scaling_, alpha_sq_norm_;

			double sq_norm, rho[3];

			/* 求残差序列的平方和 */
			sq_norm = residuals.squaredNorm();
			
			/** 
			 * 鲁棒核函数对于输入的非负标量s，返回三个值：ρ(s)、ρ'(s)、ρ"(s)，其中ρ()就是鲁棒核函数，第2、3项分别是其一阶和二阶导数。
			 * 下面Evaluate函数的第一个参数sq_norm就是非负标量s，rho[]就是ρ(s)、ρ'(s)、ρ"(s)
			 * Ceres官方手册讲：如果二阶导数小于0，则“in the outlier region”，具体含义还不是很清楚。
			 */
			loss_function->Evaluate(sq_norm, rho);

			/**
			 * FIXME: 下面通过鲁棒核函数返回的ρ(s)、ρ'(s)、ρ"(s)来计算残差比例的算法不是很明白，也没有找到出处
			 */
			double sqrt_rho1_ = sqrt(rho[1]);

			if ((sq_norm == 0.0) || (rho[2] <= 0.0))
			{
				residual_scaling_ = sqrt_rho1_;
				alpha_sq_norm_ = 0.0;
			}
			else
			{
				const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
				const double alpha = 1.0 - sqrt(D);
				residual_scaling_ = sqrt_rho1_ / (1 - alpha);
				alpha_sq_norm_ = alpha / sq_norm;
			}

			/* 用计算好的残差比例对雅可比进行修正 */
			for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
			{
				jacobians[i] = sqrt_rho1_ * (jacobians[i] - alpha_sq_norm_ * residuals * (residuals.transpose() * jacobians[i]));
			}

			/* 用计算好的残差比例对残差进行修正，防止异常值带偏优化结果 */
			residuals *= residual_scaling_;
		}
	}

	ceres::CostFunction *cost_function;
	ceres::LossFunction *loss_function;
	std::vector<double *> parameter_blocks; /* 参数块 */
	std::vector<int> drop_set; /* 要丢弃的参数块 */

	double **raw_jacobians{};
	std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
	Eigen::VectorXd residuals;

};

struct ThreadsStruct
{
	std::vector<ResidualBlockInfo *> sub_factors;
	Eigen::MatrixXd A;
	Eigen::VectorXd b;
	std::unordered_map<long, int> parameter_block_size;
	std::unordered_map<long, int> parameter_block_idx;
};

/** \brief Multi-thread to process marginalization
 */
void* ThreadsConstructA(void* threadsstruct);

/** \brief marginalization infomation
 *         边缘化信息
 */
class MarginalizationInfo
{
public:
	~MarginalizationInfo(){
//			ROS_WARN("release marginlizationinfo");

		for (auto it = parameter_block_data.begin(); it != parameter_block_data.end(); ++it)
			delete[] it->second;

		for (int i = 0; i < (int)factors.size(); i++)
		{
			delete[] factors[i]->raw_jacobians;
			delete factors[i]->cost_function;
			delete factors[i];
		}
	}

	/**
	 * @brief 添加残差块信息（即因子）到边缘化中
	 *        将新增参数块的size添加到parameter_block_sizes中
	 *        将丢弃参数块的索引从parameter_block_idx中清除
	 * @param residual_block_info 待添加的残差块信息，即因子
	 */
	void addResidualBlockInfo(ResidualBlockInfo *residual_block_info){

		/* 添加因子到factors中 */
		factors.emplace_back(residual_block_info);

		std::vector<double *> &parameter_blocks = residual_block_info->parameter_blocks;
		std::vector<int> parameter_block_sizes = residual_block_info->cost_function->parameter_block_sizes();

		for (int i = 0; i < static_cast<int>(residual_block_info->parameter_blocks.size()); i++)
		{
			/**
			 * 以参数块的地址为key，参数块的大小为value，初始化parameter_block_size。
			 * 由于使用参数块的地址作为key，因此即使本函数被调用很多次，相同的参数块也不会被重复添加。
			 * 通过多次调用本函数，最终添加的参数块只有本周期的第i帧和第j帧参数块，总共四个参数块，即：
			 * para_PR[0]、para_VBias[0]、para_PR[1]、para_VBias[1]。
			 * 
			 * 为什么这里添加参数块的时候只添加到了parameter_block_sizes，而没有同步添加到parameter_block_idx？
			 * 因为提前添加到parameter_block_idx中表示该参数块被将被丢弃，将不会进入下一周期优化。
			 * 
			 * 在marginalize()中会将所有参数块的索引更新到parameter_block_idx中，并区分提前添加和后来添加。
			 */
			double *addr = parameter_blocks[i];
			int size = parameter_block_sizes[i];
			parameter_block_size[reinterpret_cast<long>(addr)] = size;
		}

		/* 如果参数块中有需要丢弃的，则进行丢弃操作 */
		for (int i = 0; i < static_cast<int>(residual_block_info->drop_set.size()); i++)
		{
			/** 将需要丢弃的参数块提前添加到parameter_block_idx中，这些参数块将不会进入下一周期优化。
			 * drop_set中记录了parameter_blocks中要丢弃的参数块的序号，主要是本周期的第i帧，即
			 * para_PR[0]和para_VBias[0]，在这里被提前添加到parameter_block_idx中。
			 * 在marginalize()中将首先遍历提前添加到parameter_block_idx中的参数块，建立索引，并用m
			 * 变量记录提前添加参数块的结束位置，然后把parameter_block_size中其余的参数块，主要是本
			 * 周期的第j帧，即para_PR[1]和para_VBias[1]也添加到parameter_block_idx中。
			 * 在getParameterBlocks()中将只提取m变量之后的添加的参数块，从而达到丢弃drop_set中指定参
			 * 数块的目的。
			 */
			/* parameter_block_idx是map类型，存储是乱序的，索引是parameter_blocks中参数块的地址 */
			double *addr = parameter_blocks[residual_block_info->drop_set[i]];
			parameter_block_idx[reinterpret_cast<long>(addr)] = 0;
		}
	}

	/**
	 * @brief 预边缘化
	 *        遍历所有因子及其参数块，对因子进行优化，计算残差
	 *        将新增的参数块添加到parameter_block_data中，滑动窗口长度为2的情况下，添加的参数块总数就是4，
	 * 　　　　即：para_PR[0],para_PR[1],para_VBias[0],para_VBias[1]
	 */
	void preMarginalize(){

		/* 遍历所有的因子 */
		for (auto it : factors)
		{
			/* 对该因子进行优化，获得残差 */
			it->Evaluate();

			/* 遍历该因子中的所有参数块 */
			std::vector<int> block_sizes = it->cost_function->parameter_block_sizes();
			for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
			{
				/* 取得当前参数块的地址 */
				long addr = reinterpret_cast<long>(it->parameter_blocks[i]);
				int size = block_sizes[i];
				/* 在parameter_block_data中查找该参数块 */
				if (parameter_block_data.find(addr) == parameter_block_data.end())
				{
					/* 没找到该参数块，则添加该参数块到parameter_block_data中 */
					double *data = new double[size];
					memcpy(data, it->parameter_blocks[i], sizeof(double) * size);
					parameter_block_data[addr] = data;
					/**
					 * @brief 经过打印数据分析：每处理一帧点云，这里会添加四个参数块到parameter_block_data中，即：
					 * para_PR[0],para_PR[1],para_VBias[0],para_VBias[1]，parameter_block_data的总长度最大就是4。
					 */
				}
			}
		}
	}

    /**
     * @brief 边缘化
     *        更新parameter_block_idx中的索引
	 *        将parameter_block_size中新增的参数块的索引添加到parameter_block_idx中
     */
	void marginalize(){

		/** 
		 * parameter_block_idx索引的参数块总是只有四个：para_PR[0]、para_VBias[0]、para_PR[1]、para_VBias[1]，
		 * 存放顺序以及对应的索引分别是：
		 *  序号	参数块			size	索引	索引16进制
		 * 	0		para_VBias[0]	9		0		0x0
		 *  1		para_PR[0]		6		9		0x9
		 *  2		para_VBias[1]	9		15		0xf
		 *  3		para_PR[1]		6		24		0x18
		 * 
		 * 下面首先遍历提前添加到parameter_block_idx中的参数块，也就是para_VBias[0]和para_PR[0]，这两个参数块在本轮
		 * 边缘化后将被丢弃，结束位置用变量m标识。
		 */
		/* 按照parameter_block_idx的参数块顺序，索引从0开始，每个参数块的索引等于parameter_block_idx中位于该参数块之前的所有参数块的size之和 */
		int pos = 0;
		for (auto &it : parameter_block_idx)
		{
			/* 将参数块的索引修改为pos，pos是前面所有参数块的size之和 */
			it.second = pos; //second表示map容器的value，将参数块的索引设置为pos
			pos += parameter_block_size[it.first]; //first表示map容器的key，即参数块的地址，获得当前参数块的size，累加到pos上，变成下一个参数块的索引
			/**
			 * @brief 经过打印数据分析：parameter_block_idx中既有的参数块总是只有para_PR[0]和para_VBias[0]，参数块数量为2
			 */
		}

		/**
		 * m等于parameter_block_idx中既有参数块的size之和，即结束位置，即新增参数块的起始索引
		 * m变量记录了一个位置，在这个位置之前的参数块是需要丢弃的，之后的参数块是需要保留的 
		 */
		m = pos;

		/**
		 * 遍历parameter_block_size中的参数块，然后将parameter_block_size中的其余参数块添加到parameter_block_idx中，
		 * 也就是para_VBias[1]和para_PR[1]，
		 */
		for (const auto &it : parameter_block_size)
		{
			/* 如果在parameter_block_idx中找不到该参数块的索引 */
			if (parameter_block_idx.find(it.first) == parameter_block_idx.end())
			{
				/* 则添加该参数块的索引到parameter_block_idx中 */
				parameter_block_idx[it.first] = pos;
				/* 更新索引 */
				pos += it.second;
				/**
				 * @brief 经过打印数据分析：parameter_block_idx在此处新增的参数块总是只有para_PR[1]和para_VBias[1]，
				 * 新增后的总长度为4。
				 */
			}
		}

		/* n等于此次新增参数块的size之和，即所有参数块的结束位置 */
		n = pos - m;

		/**
		 * 下面启动四个线程开始计算linearized_jacobians和linearized_residuals的值，从表面来看使用了两次
		 * 求解特征值和特征向量。
		 * FIXME: 这里计算linearized_jacobians和linearized_residuals的值的目的、算法是什么？
		 */
		Eigen::MatrixXd A(pos, pos);
		Eigen::VectorXd b(pos);
		A.setZero();
		b.setZero();

		pthread_t tids[NUM_THREADS];
		ThreadsStruct threadsstruct[NUM_THREADS];
		int i = 0;
		for (auto it : factors)
		{
			threadsstruct[i].sub_factors.push_back(it);
			i++;
			i = i % NUM_THREADS;
		}
		/* 启动四个线程计算A和b */
		for (int i = 0; i < NUM_THREADS; i++)
		{
			threadsstruct[i].A = Eigen::MatrixXd::Zero(pos,pos);
			threadsstruct[i].b = Eigen::VectorXd::Zero(pos);
			threadsstruct[i].parameter_block_size = parameter_block_size;
			threadsstruct[i].parameter_block_idx = parameter_block_idx;
			int ret = pthread_create( &tids[i], NULL, ThreadsConstructA ,(void*)&(threadsstruct[i]));
			if (ret != 0)
			{
				std::cout<<"pthread_create error"<<std::endl;
				exit(1);
			}
		}
		for( int i = NUM_THREADS - 1; i >= 0; i--)
		{
			pthread_join( tids[i], NULL );
			A += threadsstruct[i].A;
			b += threadsstruct[i].b;
		}
		Eigen::MatrixXd Amm = 0.5 * (A.block(0, 0, m, m) + A.block(0, 0, m, m).transpose());
		/* 对矩阵Amm求解特征值和特征向量 */
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);

		Eigen::MatrixXd Amm_inv = saes.eigenvectors() * Eigen::VectorXd((saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() * saes.eigenvectors().transpose();

		Eigen::VectorXd bmm = b.segment(0, m);
		Eigen::MatrixXd Amr = A.block(0, m, m, n);
		Eigen::MatrixXd Arm = A.block(m, 0, n, m);
		Eigen::MatrixXd Arr = A.block(m, m, n, n);
		Eigen::VectorXd brr = b.segment(m, n);
		A = Arr - Arm * Amm_inv * Amr;
		b = brr - Arm * Amm_inv * bmm;

		/* 对矩阵A求解特征值和特征向量 */
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);
		Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
		Eigen::VectorXd S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));

		Eigen::VectorXd S_sqrt = S.cwiseSqrt();
		Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

		linearized_jacobians = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
		linearized_residuals = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;
	}

	/**
	 * @brief Get the Parameter Blocks object
	 *  将新增的两个参数块，即第j帧的para_PR[1]和para_VBias[1]，添加到keep_block中，然后返回该帧在下一周期的
	 *  参数块地址，即para_PR[0]和para_VBias[0]的地址。
	 * 
	 *  将本周期第j帧参数块的副本（即优化结果）保存到keep_block中，添加到边缘化中，然后参与到下一周期的优化中，
	 *  并返回本周第j帧（即下一周期第i帧）在下一周期对应的参数块即para_PR[0]和para_VBias[0]地址。
	 * 
	 *  在下一周期的优化中，将会调用MarginalizationFactor::Evaluate()方法进行优化，优化的方法是求本周期第j帧
	 *  （即下一周期第i帧）在下一周期优化后的结果与本周期优化后的结果（保存在keep_block中）之间的残差。
	 * 
	 *  本函数表面上看起来返回的是本周期第i帧即para_PR[0]和para_VBias[0]的地址，但实际上返回的是本周期第j帧
	 * （即下一周期第i帧）在下一周期对应的参数块地址。
	 * 
	 * @param addr_shift 从本周期第j帧参数块地址到该帧在下一周期即第i帧对应参数块地址之间的转换关系
	 * @return std::vector<double *> 返回本周期第i帧在下一周期对应参数块的地址
	 */
	std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift){
		std::vector<double *> keep_block_addr;
		keep_block_size.clear();
		keep_block_idx.clear();
		keep_block_data.clear();

		/* 遍历parameter_block_idx中记录的参数块索引 */
		for (const auto &it : parameter_block_idx)
		{
			/* 检查是否是新增的参数块索引 */
			if (it.second >= m)
			{
				/* 将新增参数块的size、索引、数据、地址添加到keep_block中 */
				/** FIXME: 为什么本函数返回的是para_PR[0]和para_VBias[0]的地址，而不是参数块的真实地址 */
				keep_block_size.push_back(parameter_block_size[it.first]);
				keep_block_idx.push_back(parameter_block_idx[it.first]);
				keep_block_data.push_back(parameter_block_data[it.first]);
				keep_block_addr.push_back(addr_shift[it.first]);
				/**
				 * @brief 经打印分析，添加到keep_block中的总是para_PR[1]和para_VBias[1]两个参数块，即第j帧的参数块，
				 * size分别是6和9，索引分别是0x18和0xf，addr_shift分别是para_PR[0]和para_VBias[0]的地址，即第i帧地址
				 */
			}
		}
		sum_block_size = std::accumulate(std::begin(keep_block_size), std::end(keep_block_size), 0);

		/* 返回第i帧参数块para_PR[0]和para_VBias[0]的地址 */
		return keep_block_addr;
	}

	/* 保存参与到边缘化的所有因子 */
	std::vector<ResidualBlockInfo *> factors;
	
	/* 用于标识本周期丢弃和保留参数块分界线的变量 */
	/* m之前是要丢弃的，m和n之间是要留到下一周期的参数块 */
	int m, n;
	
	/**
	 * 缓存本周期边缘化参数块的变量，parameter_block相关的三个变量全都是map类型，三个变量都以参数块的
	 * 地址做key来分别存放参数块的size、索引和数据。需要注意的是这三个变量都使用para_PR和para_VBias的
	 * 地址做key，当滑动窗口长度为2的时候，总共只有四个参数块及其地址：para_PR[0]、para_VBias[0]、
	 * para_PR[1]、para_VBias[1]。
	 */
	std::unordered_map<long, int> parameter_block_size; //key是参数块的地址，value是参数块的大小
	int sum_block_size;
	std::unordered_map<long, int> parameter_block_idx;  //key是参数块的地址，value是参数块的索引
	std::unordered_map<long, double *> parameter_block_data;

	/* keep_block保存本周期第j帧参数块优化后的结果，用于下一周期的优化 */
	std::vector<int> keep_block_size;
	std::vector<int> keep_block_idx;
	std::vector<double *> keep_block_data;

	/* 用于实现边缘化的雅可比 */
	Eigen::MatrixXd linearized_jacobians;
	Eigen::VectorXd linearized_residuals;
	const double eps = 1e-8;

};

/** \brief 定义防止同一个节点（一帧点云）的多次优化结果之间脱节的代价函数（因子）
 *  
 * 定义了一个新的代价函数，该代价函数的作用是确保下一周期的优化结果不要偏离上一周期的优化结果，
 * 不要出现过大的偏差。
 * 
 * 上一周期第j帧的优化结果被保存在marginalization_info中，然后求本周期第i帧（即上一周期第j帧）
 * 优化后的结果与上一周期优化后的结果之间的残差，确保本周优化后的结果不会过于偏离上一周期的结果。
 * 
 * 由于上一周期第j帧和下一个周期第i帧指向同一帧点云，即同一个节点，该节点至少要参与两次优化，具
 * 体的优化次数取决于滑动窗口的大小，那么该代价函数的作用就是防止同一个节点的状态在每次优化前后
 * 出现过大的偏差，防止后一次优化的结果与前一次优化的结果脱节。
 */
class MarginalizationFactor : public ceres::CostFunction
{
public:

	/**
	 * @brief 构造防止同一个节点的多次优化结果之间脱节的代价函数（因子）
	 * @param _marginalization_info 保存有上一周期优化结果的边缘化信息
	 */
	explicit MarginalizationFactor(MarginalizationInfo* _marginalization_info):marginalization_info(_marginalization_info){
		int cnt = 0;
		for (auto it : marginalization_info->keep_block_size)
		{
			mutable_parameter_block_sizes()->push_back(it);
			cnt += it;
		}
		set_num_residuals(marginalization_info->n);
	};

	/**
	 * @brief 代价函数的具体实现，定义了残差并求解
	 * 
	 * 该代价函数的作用是防止同一个节点（一帧点云）前后两次的优化结果之间脱节。残差公式非常简单，就是
	 * 求前后两次优化结果的差，然后乘以雅可比获得增量，叠加到既有的残差上，最后更新雅可比。
	 * 
	 * FIXME: 这里残差以及雅可比的更新算法还需要进一步分析。
	 * 
	 * @param parameters 下一周期（或者说本周期）第i帧的优化后结果
	 * @param residuals 残差
	 * @param jacobians 雅可比
	 * @return true 总是返回true
	 */
	bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override{
		int n = marginalization_info->n;
		int m = marginalization_info->m;

		/** 
		 * 求上一周期优化结果和本周期优化结果之差，上一周期的优化结果保存在marginalization_info->keep_block中，
		 * 本周期的优化结果通过参数parameters获得。
		 */
		Eigen::VectorXd dx(n);
		for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
		{
			int size = marginalization_info->keep_block_size[i];
			int idx = marginalization_info->keep_block_idx[i] - m;
			Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size);
			Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(marginalization_info->keep_block_data[i], size);
			/* 求参数块的残差 */
			if(size == 6){
				/* 该参数块是位置和姿态，前三个是位置，后三个是姿态，即para_PR */
				dx.segment<3>(idx + 0) = x.segment<3>(0) - x0.segment<3>(0);
				dx.segment<3>(idx + 3) = (Sophus::SO3d::exp(x.segment<3>(3)).inverse() * Sophus::SO3d::exp(x0.segment<3>(3))).log();
			}else{
				/* 该参数块是速度和偏差，即para_VBias */
				dx.segment(idx, size) = x - x0;
			}
			/**
			 * @brief 经过打印分析，这里的m总是等于0xf，marginalization_info->keep_block_idx[i]中总是只有两个索引：0xf和0x18；
			 *        idx的值总是0x0和0x9
			 */
		}
		
		/**
		 * 更新残差，注意这里除了上面计算出来的前后两周期优化结果的残差dx，还用到了linearized_residuals和linearized_jacobians，
		 * 这两个值是在marginalize()中启动四个线程并行计算出来的。
		 * FIXME: 这两个线性值是怎么计算出来的，为什么残差的计算公式是这样，还不是很明白。 
		 */
		Eigen::Map<Eigen::VectorXd>(residuals, n) = marginalization_info->linearized_residuals + marginalization_info->linearized_jacobians * dx;
		
		/* 更新雅可比 */
		if (jacobians)
		{

			for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
			{
				if (jacobians[i])
				{
					int size = marginalization_info->keep_block_size[i];
					int idx = marginalization_info->keep_block_idx[i] - m;
					Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(jacobians[i], n, size);
					jacobian.setZero();
					jacobian.leftCols(size) = marginalization_info->linearized_jacobians.middleCols(idx, size);
				}
			}
		}
		return true;
	}

	MarginalizationInfo* marginalization_info;
};

/** \brief Ceres Cost Funtion between Lidar Pose and IMU Preintegration
 *  @brief IMU预积分代价函数
 *  @param [in] measure_ 当前帧的IMUIntegrator，从中获得IMU测量值
 *  @param [in] GravityVec_ 重力加速度向量
 *  @param [in] sqrt_information_ IMU预积分测量噪声协方差矩阵的平方根
 */
struct Cost_NavState_PRV_Bias
{
	Cost_NavState_PRV_Bias(IMUIntegrator& measure_,
							Eigen::Vector3d& GravityVec_,
							Eigen::Matrix<double, 15, 15>  sqrt_information_):
					imu_measure(measure_),
					GravityVec(GravityVec_),
					sqrt_information(std::move(sqrt_information_)){}

/** @brief 雷达位姿和IMU预积分之间的代价函数
  * @param [in] pri_        第i帧点云对应的位姿
  * @param [in] velobiasi   第i帧点云对应的速度和偏差
  * @param [in] prj_        第j帧点云对应的位姿
  * @param [in] velobiasj_  第j帧点云对应的速度和偏差
  * @param [out] residual   残差
  */
	template <typename T>
	bool operator()( const T *pri_, const T *velobiasi_, const T *prj_, const T *velobiasj_, T *residual) const {

		/**
		 * @brief 第一步：从输入参数（数组形式）取得所有待优化变量：Pi,Pj,Ri,Rj,Vi,Vj,dbgi,dbai
		 */

    	/* 以矩阵的方式访问C++数组pri_，分别获得位置Pi和姿态Ri */
        Eigen::Map<const Eigen::Matrix<T, 6, 1>> PRi(pri_);
		Eigen::Matrix<T, 3, 1> Pi = PRi.template segment<3>(0);
		Sophus::SO3<T> SO3_Ri = Sophus::SO3<T>::exp(PRi.template segment<3>(3));
    	/* 获得位置Pj和姿态Rj */
		Eigen::Map<const Eigen::Matrix<T, 6, 1>> PRj(prj_);
		Eigen::Matrix<T, 3, 1> Pj = PRj.template segment<3>(0);
		Sophus::SO3<T> SO3_Rj = Sophus::SO3<T>::exp(PRj.template segment<3>(3));
    	/* 获得速度Vi和偏差增量dbgi、dbai */
		Eigen::Map<const Eigen::Matrix<T, 9, 1>> velobiasi(velobiasi_);
		Eigen::Matrix<T, 3, 1> Vi = velobiasi.template segment<3>(0);
		/* 偏差增量 = 新的偏差 - 旧的偏差 */
		/* 新的偏差是待优化的变量，旧的偏差来自上一轮优化的结果，如果这是第一次优化，则旧的偏差等于0 */
		/* 每一轮优化后，新的偏差都会被更新到lidarFrame中，成为下一轮优化中的“旧的偏差” */
		Eigen::Matrix<T, 3, 1> dbgi = velobiasi.template segment<3>(3) - imu_measure.GetBiasGyr().cast<T>();
		Eigen::Matrix<T, 3, 1> dbai = velobiasi.template segment<3>(6) - imu_measure.GetBiasAcc().cast<T>();
    	/* 获得速度Vj */
		Eigen::Map<const Eigen::Matrix<T, 9, 1>> velobiasj(velobiasj_);
		Eigen::Matrix<T, 3, 1> Vj = velobiasj.template segment<3>(0);

		Eigen::Map<Eigen::Matrix<T, 15, 1> > eResiduals(residual);
		eResiduals = Eigen::Matrix<T, 15, 1>::Zero();

    	/* 求i、j两帧之间的dt和dt^2 */
		T dTij = T(imu_measure.GetDeltaTime());
		T dT2 = dTij*dTij;
    	/* 求i、j两帧之间的位置增量dPij，速度增量dVij，姿态增量dRij */
		Eigen::Matrix<T, 3, 1> dPij = imu_measure.GetDeltaP().cast<T>();
		Eigen::Matrix<T, 3, 1> dVij = imu_measure.GetDeltaV().cast<T>();
		Sophus::SO3<T> dRij = Sophus::SO3<T>(imu_measure.GetDeltaQ().cast<T>());
		Sophus::SO3<T> RiT = SO3_Ri.inverse();

		/**
		 * @brief 第二步：计算残差 rPij，rRij，rVij
		 *   残差的计算公式与邱笑晨《IMU预积分总结与公式推导》第六章中提到的残差公式完全一致
		 *   公式的由两部分组成，第一部分是非IMU方式的估计值，先验来自IMU积分，后验来自上一周期的点云到地图的优化
		 *   第二部分是IMU预积分的测量值，包含有近似的修正值，用到了测量噪声雅可比矩阵
		 */

		/* 求残差第一项rPij：雷达位置增量和IMU预积分位置增量之差 */
		/* 第1行是非IMU方式的估计值，先验来自IMU积分，后验来自上一周期的优化 */
		/* 第2、3行是IMU预积分的测量值，包含有近似的修正值 */
		Eigen::Matrix<T, 3, 1> rPij = RiT*(Pj - Pi - Vi*dTij - 0.5*GravityVec.cast<T>()*dT2) -
						(dPij + imu_measure.GetJacobian().block<3,3>(IMUIntegrator::O_P, IMUIntegrator::O_BG).cast<T>()*dbgi +
						imu_measure.GetJacobian().block<3,3>(IMUIntegrator::O_P, IMUIntegrator::O_BA).cast<T>()*dbai);

    	/* 求残差第二项rPhiij：雷达姿态增量和IMU预积分姿态增量之差*/
    	/* (RiT * SO3_Rj)即是Rj乘以Ri的逆，得到雷达的姿态增量 */
    	/* dRij是IMU预积分姿态增量，叠加偏差dR_dbg后得到IMU预积分姿态增量 */
    	/* IMU预积分姿态增量求逆，再乘以雷达的姿态增量，得到两者之差 */
    	/* 姿态增量之差通过对数映射转成李代数，即向量的形式 */
		Sophus::SO3<T> dR_dbg = Sophus::SO3<T>::exp(
						imu_measure.GetJacobian().block<3,3>(IMUIntegrator::O_R, IMUIntegrator::O_BG).cast<T>()*dbgi);
		Sophus::SO3<T> rRij = (dRij * dR_dbg).inverse() * RiT * SO3_Rj;
		Eigen::Matrix<T, 3, 1> rPhiij = rRij.log();

    	/* 求残差第三项rVij：雷达速度增量和IMU预积分速度增量之差 */
    	/* RiT*(……)即是PoseEstimation中求第j帧lidarFrame.V的逆过程，求得雷达的速度增量 */
    	/* 下面第2、3行在IMU预积分速度增量的基础上叠加偏差，得到IMU预积分速度增量 */
		Eigen::Matrix<T, 3, 1> rVij = RiT*(Vj - Vi - GravityVec.cast<T>()*dTij) -
						(dVij + imu_measure.GetJacobian().block<3,3>(IMUIntegrator::O_V, IMUIntegrator::O_BG).cast<T>()*dbgi +
										imu_measure.GetJacobian().block<3,3>(IMUIntegrator::O_V, IMUIntegrator::O_BA).cast<T>()*dbai);

		eResiduals.template segment<3>(0) = rPij;
		eResiduals.template segment<3>(3) = rPhiij;
		eResiduals.template segment<3>(6) = rVij;

    	/* 残差第四项：第i,j两帧的偏差之差*/
    	/* segment<6>(3)表示从velobiasj的第3个元素开始，取6个，即偏差部分 */
		eResiduals.template segment<6>(9) = velobiasj.template segment<6>(3) - velobiasi.template segment<6>(3);

		/**
		 * @brief 第三步：残差左乘信息矩阵
		 *  -所谓信息矩阵是指IMU预积分测量噪声协方差矩阵的逆矩阵的平方根
		 *  -协方差矩阵的逆矩阵相当于取了方差的倒数，方差越大，权重越小，反之权重越大
		 *  -逆矩阵的平方根通过对协方差矩阵进行Cholesky分解获得，下三角矩阵L即是原矩阵的平方根
		 *  -在残差上左乘信息矩阵能够起到平衡权重的作用。优化过程中误差只是减少并不是完全消除，不能消除的误差去哪里呢？
		 *  -当然是每条边（因子图）分摊了，但是每条边都分摊一样多的误差显然是不科学的，这个时候就需要信息矩阵，它表达
		 *   了每条边要分摊的误差比例。
		 */
		eResiduals.applyOnTheLeft(sqrt_information.template cast<T>());

		return true;
	}

	static ceres::CostFunction *Create(IMUIntegrator& measure_,
										Eigen::Vector3d& GravityVec_,
										Eigen::Matrix<double, 15, 15>  sqrt_information_) {
		return (new ceres::AutoDiffCostFunction<Cost_NavState_PRV_Bias, 15, 6, 9, 6, 9>(
						new Cost_NavState_PRV_Bias(measure_,
													GravityVec_,
													std::move(sqrt_information_))));
	}

	IMUIntegrator imu_measure;
	Eigen::Vector3d GravityVec;
	Eigen::Matrix<double, 15, 15> sqrt_information;
};

/** \brief Ceres Cost Funtion between PointCloud Sharp Feature and Map Cloud
 *  @brief 角点特征点云与Map的Ceres代价函数
 *  @param _p 角点特征点p
 *  @param _vtx1 Map点a
 *  @param _vtx2 Map点b
 *  @param Tbl IMU到Lidar的外参矩阵
 *  @param sqrt_information_ 具体的含义还不清楚，数值等于1/lidar_m 
 */
struct Cost_NavState_IMU_Line
{
    Cost_NavState_IMU_Line(Eigen::Vector3d  _p, Eigen::Vector3d  _vtx1, Eigen::Vector3d  _vtx2,
                           const Eigen::Matrix4d& Tbl, Eigen::Matrix<double, 1, 1>  sqrt_information_):
            point(std::move(_p)), vtx1(std::move(_vtx1)), vtx2(std::move(_vtx2)),
            sqrt_information(std::move(sqrt_information_)){
      /* 求a、b两点的距离 */
	  l12 = std::sqrt((vtx1(0) - vtx2(0))*(vtx1(0) - vtx2(0)) + (vtx1(1) - vtx2(1))*
                                                                (vtx1(1) - vtx2(1)) + (vtx1(2) - vtx2(2))*(vtx1(2) - vtx2(2)));
      Eigen::Matrix3d m3d = Tbl.topLeftCorner(3,3);
      qbl = Eigen::Quaterniond(m3d).normalized();
      qlb = qbl.conjugate();
      Pbl = Tbl.topRightCorner(3,1);
      Plb = -(qlb * Pbl);
    }

	/**
	 * @brief 代价函数的实体
	 * @tparam T 
	 * @param PRi 待优化的位姿，初始值来自para_PR
	 * @param residual 本轮优化后获得的残差值
	 * @return true 总是返回true
	 */
    template <typename T>
    bool operator()(const T *PRi, T *residual) const {
      Eigen::Matrix<T, 3, 1> cp{T(point.x()), T(point.y()), T(point.z())};
      Eigen::Matrix<T, 3, 1> lpa{T(vtx1.x()), T(vtx1.y()), T(vtx1.z())};
      Eigen::Matrix<T, 3, 1> lpb{T(vtx2.x()), T(vtx2.y()), T(vtx2.z())};

      /* 从添加到优化problem的参数块中取得当前点云帧的位姿估计值，对p点进行位姿变换，变换到Map坐标系 */
      /* 和processPointToLine方法中将p点转到地图坐标系的区别是，这里用到了外参矩阵，精度更高 */
      Eigen::Map<const Eigen::Matrix<T, 6, 1>> pri_wb(PRi);
      Eigen::Quaternion<T> q_wb = Sophus::SO3<T>::exp(pri_wb.template segment<3>(3)).unit_quaternion();
      Eigen::Matrix<T, 3, 1> t_wb = pri_wb.template segment<3>(0);
      Eigen::Quaternion<T> q_wl = q_wb * qbl.cast<T>();
      Eigen::Matrix<T, 3, 1> t_wl = q_wb * Pbl.cast<T>() + t_wb;
      Eigen::Matrix<T, 3, 1> P_to_Map = q_wl * cp + t_wl;

      /* 求线段pa和pb叉乘的结果，即pa和pb所围成的平行四边形的面积 */
      T a012 = ceres::sqrt(
              ((P_to_Map(0) - lpa(0)) * (P_to_Map(1) - lpb(1)) - (P_to_Map(0) - lpb(0)) * (P_to_Map(1) - lpa(1)))
              * ((P_to_Map(0) - lpa(0)) * (P_to_Map(1) - lpb(1)) - (P_to_Map(0) - lpb(0)) * (P_to_Map(1) - lpa(1)))
              + ((P_to_Map(0) - lpa(0)) * (P_to_Map(2) - lpb(2)) - (P_to_Map(0) - lpb(0)) * (P_to_Map(2) - lpa(2)))
                * ((P_to_Map(0) - lpa(0)) * (P_to_Map(2) - lpb(2)) - (P_to_Map(0) - lpb(0)) * (P_to_Map(2) - lpa(2)))
              + ((P_to_Map(1) - lpa(1)) * (P_to_Map(2) - lpb(2)) - (P_to_Map(1) - lpb(1)) * (P_to_Map(2) - lpa(2)))
                * ((P_to_Map(1) - lpa(1)) * (P_to_Map(2) - lpb(2)) - (P_to_Map(1) - lpb(1)) * (P_to_Map(2) - lpa(2))));
      /* 平行四边形面积除以对角线ab的长度，就得到点p到直线ab的距离 */
      T ld2 = a012 / T(l12);
      /* 计算p点的权重: */
      /* p点的深度越大，权重越高，但是深度做了开方，进行了衰减*/
      /* p点到直线ab的距离越远，权重越小 */
      T _weight = T(1) - T(0.9) * ceres::abs(ld2) / ceres::sqrt(
              ceres::sqrt( P_to_Map(0) * P_to_Map(0) +
                           P_to_Map(1) * P_to_Map(1) +
                           P_to_Map(2) * P_to_Map(2) ));
      /* 求得残差 */
      /* FIXME:残差为什么要乘以sqrt_information=1/lidar_m？大约是666.67 */
      /* FIXME:lidar_m = 1.5e-3，是定义在IMUIntergrator.h文件中的常数，是什么含义？ */
      residual[0] = T(sqrt_information(0)) * _weight * ld2;

      return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector3d& curr_point_,
                                       const Eigen::Vector3d& last_point_a_,
                                       const Eigen::Vector3d& last_point_b_,
                                       const Eigen::Matrix4d& Tbl,
                                       Eigen::Matrix<double, 1, 1>  sqrt_information_) {
      return (new ceres::AutoDiffCostFunction<Cost_NavState_IMU_Line, 1, 6>(
              new Cost_NavState_IMU_Line(curr_point_, last_point_a_, last_point_b_, Tbl, std::move(sqrt_information_))));
    }

    Eigen::Vector3d point;
    Eigen::Vector3d vtx1;
    Eigen::Vector3d vtx2;
    double l12;
    Eigen::Quaterniond qbl, qlb;
    Eigen::Vector3d Pbl, Plb;
    Eigen::Matrix<double, 1, 1> sqrt_information;
};

/** \brief Ceres Cost Funtion between PointCloud Flat Feature and Map Cloud
 */
struct Cost_NavState_IMU_Plan
{
    Cost_NavState_IMU_Plan(Eigen::Vector3d  _p, double _pa, double _pb, double _pc, double _pd,

                           const Eigen::Matrix4d& Tbl, Eigen::Matrix<double, 1, 1>  sqrt_information_):
            point(std::move(_p)), pa(_pa), pb(_pb), pc(_pc), pd(_pd), sqrt_information(std::move(sqrt_information_)){
      Eigen::Matrix3d m3d = Tbl.topLeftCorner(3,3);
      qbl = Eigen::Quaterniond(m3d).normalized();
      qlb = qbl.conjugate();
      Pbl = Tbl.topRightCorner(3,1);
      Plb = -(qlb * Pbl);
    }

    template <typename T>
    bool operator()(const T *PRi, T *residual) const {
      Eigen::Matrix<T, 3, 1> cp{T(point.x()), T(point.y()), T(point.z())};

      Eigen::Map<const Eigen::Matrix<T, 6, 1>> pri_wb(PRi);
      Eigen::Quaternion<T> q_wb = Sophus::SO3<T>::exp(pri_wb.template segment<3>(3)).unit_quaternion();
      Eigen::Matrix<T, 3, 1> t_wb = pri_wb.template segment<3>(0);
      Eigen::Quaternion<T> q_wl = q_wb * qbl.cast<T>();
      Eigen::Matrix<T, 3, 1> t_wl = q_wb * Pbl.cast<T>() + t_wb;
      Eigen::Matrix<T, 3, 1> P_to_Map = q_wl * cp + t_wl;

      T pd2 = T(pa) * P_to_Map(0) + T(pb) * P_to_Map(1) + T(pc) * P_to_Map(2) + T(pd);
      T _weight = T(1) - T(0.9) * ceres::abs(pd2) /ceres::sqrt(
              ceres::sqrt( P_to_Map(0) * P_to_Map(0) +
                           P_to_Map(1) * P_to_Map(1) +
                           P_to_Map(2) * P_to_Map(2) ));
      residual[0] = T(sqrt_information(0)) * _weight * pd2;

      return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector3d& curr_point_,
                                       const double& pa_,
                                       const double& pb_,
                                       const double& pc_,
                                       const double& pd_,
                                       const Eigen::Matrix4d& Tbl,
                                       Eigen::Matrix<double, 1, 1>  sqrt_information_) {
      return (new ceres::AutoDiffCostFunction<Cost_NavState_IMU_Plan, 1, 6>(
              new Cost_NavState_IMU_Plan(curr_point_, pa_, pb_, pc_, pd_, Tbl, std::move(sqrt_information_))));
    }

    double pa, pb, pc, pd;
    Eigen::Vector3d point;
    Eigen::Quaterniond qbl, qlb;
    Eigen::Vector3d Pbl, Plb;
    Eigen::Matrix<double, 1, 1> sqrt_information;
};


/** \brief Ceres Cost Funtion between PointCloud Flat Feature and Map Cloud
 *  @brief 平面特征点云与Map点云之间的Ceres代价函数
 *  @param _p 平面特征点p
 *  @param _p_proj 点p在Map最近五点构成的平面上的投影点
 *  @param Tbl IMU到Lidar的外参矩阵
 *  @param _sqrt_information 具体的含义还不清楚，用到了SVD分解 
 */
struct Cost_NavState_IMU_Plan_Vec
{
    Cost_NavState_IMU_Plan_Vec(Eigen::Vector3d  _p, 
							   Eigen::Vector3d  _p_proj,
							   const Eigen::Matrix4d& Tbl,
							   Eigen::Matrix<double, 3, 3> _sqrt_information):
                               point(std::move(_p)),
							   point_proj(std::move(_p_proj)), 
							   sqrt_information(std::move(_sqrt_information)){
      Eigen::Matrix3d m3d = Tbl.topLeftCorner(3,3);
      qbl = Eigen::Quaterniond(m3d).normalized();
      qlb = qbl.conjugate();
      Pbl = Tbl.topRightCorner(3,1);
      Plb = -(qlb * Pbl);
    }

    template <typename T>
    bool operator()(const T *PRi, T *residual) const {
      Eigen::Matrix<T, 3, 1> cp{T(point.x()), T(point.y()), T(point.z())};
      Eigen::Matrix<T, 3, 1> cp_proj{T(point_proj.x()), T(point_proj.y()), T(point_proj.z())};

      /* 从添加到优化problem的参数块中取得当前点云帧的位姿估计值，对p点进行位姿变换，变换到Map坐标系 */
      /* 和processPointToLine方法中将p点转到地图坐标系的区别是，这里用到了外参矩阵，精度更高 */
      Eigen::Map<const Eigen::Matrix<T, 6, 1>> pri_wb(PRi);
      Eigen::Quaternion<T> q_wb = Sophus::SO3<T>::exp(pri_wb.template segment<3>(3)).unit_quaternion();
      Eigen::Matrix<T, 3, 1> t_wb = pri_wb.template segment<3>(0);
      Eigen::Quaternion<T> q_wl = q_wb * qbl.cast<T>();
      Eigen::Matrix<T, 3, 1> t_wl = q_wb * Pbl.cast<T>() + t_wb;
      Eigen::Matrix<T, 3, 1> P_to_Map = q_wl * cp + t_wl;

      /* 残差就是点p与其在Map平面投影点的差，实际上是个向量 */
      Eigen::Map<Eigen::Matrix<T, 3, 1> > eResiduals(residual);
      eResiduals = P_to_Map - cp_proj;
      /* 计算残差的权重 */
      /* p点的深度越大，权重越高，但是深度做了开方，进行了衰减*/
      /* p点到其在Map平面投影点的差越大，权重越小 */
      T _weight = T(1) - T(0.9) * (P_to_Map - cp_proj).norm() /ceres::sqrt(
              ceres::sqrt( P_to_Map(0) * P_to_Map(0) +
                           P_to_Map(1) * P_to_Map(1) +
                           P_to_Map(2) * P_to_Map(2) ));
      eResiduals *= _weight;
      /* FIXME:残差为什么要左乘信息矩阵不是很明白 */
      eResiduals.applyOnTheLeft(sqrt_information.template cast<T>());

      return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector3d& curr_point_,
                                       const Eigen::Vector3d&  p_proj_,
                                       const Eigen::Matrix4d& Tbl,
									   const Eigen::Matrix<double, 3, 3> sqrt_information_) {
      return (new ceres::AutoDiffCostFunction<Cost_NavState_IMU_Plan_Vec, 3, 6>(
              new Cost_NavState_IMU_Plan_Vec(curr_point_, p_proj_, Tbl, sqrt_information_)));
    }

    Eigen::Vector3d point;
    Eigen::Vector3d point_proj;
    Eigen::Quaterniond qbl, qlb;
    Eigen::Vector3d Pbl, Plb;
    Eigen::Matrix<double, 3, 3> sqrt_information;
};

/** @brief 不规则特征点云与Map的Ceres代价函数
 *  @param _p 角点特征点p
 *  @param _pa Map平面方程参数A
 *  @param _pb Map平面方程参数B
 *  @param _pb Map平面方程参数C
 *  @param _pb Map平面方程参数D
 *  @param Tbl IMU到Lidar的外参矩阵
 *  @param sqrt_information_ 具体的含义还不清楚，数值等于1/lidar_m 
 */
struct Cost_NonFeature_ICP
{
    Cost_NonFeature_ICP(Eigen::Vector3d  _p, double _pa, double _pb, double _pc, double _pd,
                        const Eigen::Matrix4d& Tbl, Eigen::Matrix<double, 1, 1>  sqrt_information_):
            			point(std::move(_p)), pa(_pa), pb(_pb), pc(_pc), pd(_pd), sqrt_information(std::move(sqrt_information_)){
      Eigen::Matrix3d m3d = Tbl.topLeftCorner(3,3);
      qbl = Eigen::Quaterniond(m3d).normalized();
      qlb = qbl.conjugate();
      Pbl = Tbl.topRightCorner(3,1);
      Plb = -(qlb * Pbl);
    }

    template <typename T>
    bool operator()(const T *PRi, T *residual) const {
      Eigen::Matrix<T, 3, 1> cp{T(point.x()), T(point.y()), T(point.z())};

      /* 从添加到优化problem的参数块中取得当前点云帧的位姿估计值，对p点进行位姿变换，变换到Map坐标系 */
      /* 和processPointToLine方法中将p点转到地图坐标系的区别是，这里用到了外参矩阵，精度更高 */
      Eigen::Map<const Eigen::Matrix<T, 6, 1>> pri_wb(PRi);
      Eigen::Quaternion<T> q_wb = Sophus::SO3<T>::exp(pri_wb.template segment<3>(3)).unit_quaternion();
      Eigen::Matrix<T, 3, 1> t_wb = pri_wb.template segment<3>(0);
      Eigen::Quaternion<T> q_wl = q_wb * qbl.cast<T>();
      Eigen::Matrix<T, 3, 1> t_wl = q_wb * Pbl.cast<T>() + t_wb;
      Eigen::Matrix<T, 3, 1> P_to_Map = q_wl * cp + t_wl;

      /* 点p到平面的距离即是残差 */
      T pd2 = T(pa) * P_to_Map(0) + T(pb) * P_to_Map(1) + T(pc) * P_to_Map(2) + T(pd);
      /* 计算残差的权重 */
      /* p点的深度越大，权重越高，但是深度做了开方，进行了衰减*/
      /* p点到平面的距离越大，权重越小 */
      T _weight = T(1) - T(0.9) * ceres::abs(pd2) /ceres::sqrt(
              ceres::sqrt( P_to_Map(0) * P_to_Map(0) +
                           P_to_Map(1) * P_to_Map(1) +
                           P_to_Map(2) * P_to_Map(2) ));
      residual[0] = T(sqrt_information(0)) * _weight * pd2;

      return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector3d& curr_point_,
                                       const double& pa_,
                                       const double& pb_,
                                       const double& pc_,
                                       const double& pd_,
                                       const Eigen::Matrix4d& Tbl,
                                       Eigen::Matrix<double, 1, 1>  sqrt_information_) {
      return (new ceres::AutoDiffCostFunction<Cost_NonFeature_ICP, 1, 6>(
              new Cost_NonFeature_ICP(curr_point_, pa_, pb_, pc_, pd_, Tbl, std::move(sqrt_information_))));
    }

    double pa, pb, pc, pd;
    Eigen::Vector3d point;
    Eigen::Quaterniond qbl, qlb;
    Eigen::Vector3d Pbl, Plb;
    Eigen::Matrix<double, 1, 1> sqrt_information;
};

/** \brief Ceres Cost Funtion for Initial Gravity Direction
 */
struct Cost_Initial_G
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	Cost_Initial_G(Eigen::Vector3d acc_): acc(acc_){}

	template <typename T>
	bool operator()( const T *q, T *residual) const {
		Eigen::Matrix<T, 3, 1> acc_T = acc.cast<T>();
		Eigen::Quaternion<T> q_wg{q[0], q[1], q[2], q[3]};
		Eigen::Matrix<T, 3, 1> g_I{T(0), T(0), T(-9.805)};
		Eigen::Matrix<T, 3, 1> resi = q_wg * g_I - acc_T;
		residual[0] = resi[0];
		residual[1] = resi[1];
		residual[2] = resi[2];

		return true;
	}

	static ceres::CostFunction *Create(Eigen::Vector3d acc_) {
		return (new ceres::AutoDiffCostFunction<Cost_Initial_G, 3, 4>(
						new Cost_Initial_G(acc_)));
	}

	Eigen::Vector3d acc;
};

/** \brief Ceres Cost Funtion of IMU Factor in LIO Initialization
 *  @brief IMU预积分代价函数
 *   主要目的是获得精确的重力加速度方向，此外获得每帧的精确速度和整体偏差
 *  @param measure_ 当前帧的IMUIntegrator，从中获得IMU测量值
 *  @param ri_ 第i帧的姿态
 *  @param rj_ 第j帧的姿态
 *  @param dp_ 从i到j的位置增量
 *  @param sqrt_information_ 具体的含义还不清楚，来自IMU测量噪声协方差矩阵的分解
 */struct Cost_Initialization_IMU
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	Cost_Initialization_IMU(IMUIntegrator& measure_,
									Eigen::Vector3d ri_,
									Eigen::Vector3d rj_,
									Eigen::Vector3d dp_,
									Eigen::Matrix<double, 9, 9>  sqrt_information_):
									imu_measure(measure_),
									ri(ri_),
									rj(rj_),
									dp(dp_),
									sqrt_information(std::move(sqrt_information_)){}

/**
 * @brief IMU预积分初始化代价函数，主要目的是获得精确的重力加速度方向，此外获得每帧的精确速度和整体偏差
 * @tparam T 
 * @param rwg_ 重力加速度方向
 * @param vi_  第i帧的初始速度
 * @param vj_  第j帧的初始速度
 * @param ba_  待优化的加速度计偏差
 * @param bg_  待优化的角速度计偏差
 * @param residual 残差
 */
	template <typename T>
	bool operator()(const T *rwg_, const T *vi_, const T *vj_, const T *ba_, const T *bg_, T *residual) const {
		Eigen::Matrix<T, 3, 1> G_I{T(0), T(0), T(-9.805)};
		
    /* 获得偏差增量dbgi、dbai */
		Eigen::Map<const Eigen::Matrix<T, 3, 1>> ba(ba_);
		Eigen::Map<const Eigen::Matrix<T, 3, 1>> bg(bg_);
		/* 偏差增量=新的偏差-旧的偏差 */
		/* 新的偏差是待优化的变量，而旧的偏差来自点云到地图的匹配，如果这是第一次优化，则旧的偏差等于0 */
		/* 每一轮优化后，新的偏差都会被更新到lidarFrame中，成为下一轮优化中的“旧的偏差” */
		Eigen::Matrix<T, 3, 1> dbg = bg - imu_measure.GetBiasGyr().cast<T>();
		Eigen::Matrix<T, 3, 1> dba = ba - imu_measure.GetBiasAcc().cast<T>();
		
		Sophus::SO3<T> SO3_Ri = Sophus::SO3<T>::exp(ri.cast<T>());
		Sophus::SO3<T> SO3_Rj = Sophus::SO3<T>::exp(rj.cast<T>());

		Eigen::Matrix<T, 3, 1> dP = dp.cast<T>();

    /* 获得初始姿态，即初始的重力加速度方向 */
		Eigen::Map<const Eigen::Matrix<T, 3, 1>> rwg(rwg_);
		Sophus::SO3<T> SO3_Rwg = Sophus::SO3<T>::exp(rwg);

    /* 获得Vi和Vj */
		Eigen::Map<const Eigen::Matrix<T, 3, 1>> vi(vi_);
		Eigen::Matrix<T, 3, 1> Vi = vi;
		Eigen::Map<const Eigen::Matrix<T, 3, 1>> vj(vj_);
		Eigen::Matrix<T, 3, 1> Vj = vj;

		Eigen::Map<Eigen::Matrix<T, 9, 1> > eResiduals(residual);
		eResiduals = Eigen::Matrix<T, 9, 1>::Zero();

    /* 求i、j两帧之间的dt和dt^2 */
		T dTij = T(imu_measure.GetDeltaTime());
		T dT2 = dTij*dTij;
    /* 求i、j两帧之间的位置增量dPij，速度增量dVij，姿态增量dRij */
		Eigen::Matrix<T, 3, 1> dPij = imu_measure.GetDeltaP().cast<T>();
		Eigen::Matrix<T, 3, 1> dVij = imu_measure.GetDeltaV().cast<T>();
		Sophus::SO3<T> dRij = Sophus::SO3<T>(imu_measure.GetDeltaQ().cast<T>());
		Sophus::SO3<T> RiT = SO3_Ri.inverse();

    /* 求残差第一项rPij：雷达位置增量和IMU预积分位置增量之差 */
    /* 下面第1行求雷达位置增量 */
    /* 下面第2、3行在IMU预积分位置增量dPij的基础上叠加角速度和加速度偏差 */
		Eigen::Matrix<T, 3, 1> rPij = RiT*(dP - Vi*dTij - SO3_Rwg*G_I*dT2*T(0.5)) -
						(dPij + imu_measure.GetJacobian().block<3,3>(IMUIntegrator::O_P, IMUIntegrator::O_BG).cast<T>()*dbg +
						imu_measure.GetJacobian().block<3,3>(IMUIntegrator::O_P, IMUIntegrator::O_BA).cast<T>()*dba);

    /* 求残差第二项rPhiij：雷达姿态增量和IMU预积分姿态增量之差*/
    /* (RiT * SO3_Rj)即是Rj乘以Ri的逆，得到雷达的姿态增量 */
    /* dRij是IMU预积分姿态增量，叠加偏差dR_dbg后得到IMU预积分姿态增量 */
    /* IMU预积分姿态增量求逆，再乘以雷达的姿态增量，得到两者之差 */
    /* 姿态增量之差通过对数映射转成李代数，即向量的形式 */
		Sophus::SO3<T> dR_dbg = Sophus::SO3<T>::exp(
						imu_measure.GetJacobian().block<3,3>(IMUIntegrator::O_R, IMUIntegrator::O_BG).cast<T>()*dbg);
		Sophus::SO3<T> rRij = (dRij * dR_dbg).inverse() * RiT * SO3_Rj;
		Eigen::Matrix<T, 3, 1> rPhiij = rRij.log();

    /* 求残差第三项rVij：雷达速度增量和IMU预积分速度增量之差 */
    /* RiT*(……)即是PoseEstimation中求第j帧lidarFrame.V的逆过程，求得雷达的速度增量 */
    /* 下面第2、3行在IMU预积分速度增量的基础上叠加偏差，得到IMU预积分速度增量 */
		Eigen::Matrix<T, 3, 1> rVij = RiT*(Vj - Vi - SO3_Rwg*G_I*dTij) -
						(dVij + imu_measure.GetJacobian().block<3,3>(IMUIntegrator::O_V, IMUIntegrator::O_BG).cast<T>()*dbg +
										imu_measure.GetJacobian().block<3,3>(IMUIntegrator::O_V, IMUIntegrator::O_BA).cast<T>()*dba);

		eResiduals.template segment<3>(0) = rPij;
		eResiduals.template segment<3>(3) = rPhiij;
		eResiduals.template segment<3>(6) = rVij;

		/**
		 * @brief 第三步：残差左乘信息矩阵
		 *  -所谓信息矩阵是指IMU预积分测量噪声协方差矩阵的逆矩阵的平方根
		 *  -协方差矩阵的逆矩阵相当于取了方差的倒数，方差越大，权重越小，反之权重越大
		 *  -逆矩阵的平方根通过对协方差矩阵进行Cholesky分解获得，下三角矩阵L即是原矩阵的平方根
		 *  -在残差上左乘信息矩阵能够起到平衡权重的作用。优化过程中误差只是减少并不是完全消除，不能消除的误差去哪里呢？
		 *  -当然是每条边（因子图）分摊了，但是每条边都分摊一样多的误差显然是不科学的，这个时候就需要信息矩阵，它表达
		 *   了每条边要分摊的误差比例。
		 */
		eResiduals.applyOnTheLeft(sqrt_information.template cast<T>());

		return true;
	}

	static ceres::CostFunction *Create(IMUIntegrator& measure_,
										Eigen::Vector3d ri_,
										Eigen::Vector3d rj_,
										Eigen::Vector3d dp_,
										Eigen::Matrix<double, 9, 9>  sqrt_information_) {
		return (new ceres::AutoDiffCostFunction<Cost_Initialization_IMU, 9, 3, 3, 3, 3, 3>(
						new Cost_Initialization_IMU(measure_,
															ri_,
															rj_,
															dp_,
															std::move(sqrt_information_))));
	}

	IMUIntegrator imu_measure;
	Eigen::Vector3d ri;
	Eigen::Vector3d rj;
	Eigen::Vector3d dp;
	Eigen::Matrix<double, 9, 9> sqrt_information;
};

/** \brief Ceres Cost Funtion of IMU Biases and Velocity Prior Factor in LIO Initialization
 *  @brief 定义速度、偏差状态变量的代价函数
 *    残差就是bv_与prior_之差
 */
struct Cost_Initialization_Prior_bv
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	Cost_Initialization_Prior_bv(Eigen::Vector3d prior_, 
									Eigen::Matrix3d sqrt_information_):
									prior(prior_),
									sqrt_information(std::move(sqrt_information_)){}

	template <typename T>
	bool operator()(const T *bv_, T *residual) const {
		Eigen::Map<const Eigen::Matrix<T, 3, 1>> bv(bv_);
		Eigen::Matrix<T, 3, 1> Bv = bv;

		Eigen::Matrix<T, 3, 1> prior_T(prior.cast<T>());
		Eigen::Matrix<T, 3, 1> prior_Bv = prior_T;

		Eigen::Map<Eigen::Matrix<T, 3, 1> > eResiduals(residual);
		eResiduals = Eigen::Matrix<T, 3, 1>::Zero();

		eResiduals = Bv - prior_Bv;

		eResiduals.applyOnTheLeft(sqrt_information.template cast<T>());

		return true;
	}

	static ceres::CostFunction *Create(Eigen::Vector3d prior_, Eigen::Matrix3d sqrt_information_) {
		return (new ceres::AutoDiffCostFunction<Cost_Initialization_Prior_bv, 3, 3>(
						new Cost_Initialization_Prior_bv(prior_, std::move(sqrt_information_))));
	}

	Eigen::Vector3d prior;
	Eigen::Matrix3d sqrt_information;
};

/** \brief Ceres Cost Funtion of Rwg Prior Factor in LIO Initialization
 *  @brief 定义姿态状态变量的代价函数
 *    残差就是待优化姿态r_wg_和初始姿态prior_之间的姿态增量
 */
struct Cost_Initialization_Prior_R
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	Cost_Initialization_Prior_R(Eigen::Vector3d prior_, 
								Eigen::Matrix3d sqrt_information_):
								prior(prior_),
								sqrt_information(std::move(sqrt_information_)){}

	template <typename T>
	bool operator()( const T *r_wg_, T *residual) const {
		Eigen::Map<const Eigen::Matrix<T, 3, 1>> r_wg(r_wg_);
		Eigen::Matrix<T, 3, 1> R_wg = r_wg;
		Sophus::SO3<T> SO3_R_wg = Sophus::SO3<T>::exp(R_wg);

		Eigen::Matrix<T, 3, 1> prior_T(prior.cast<T>());
		Sophus::SO3<T> prior_R_wg = Sophus::SO3<T>::exp(prior_T);

		Sophus::SO3<T> d_R = SO3_R_wg.inverse() * prior_R_wg;
		Eigen::Matrix<T, 3, 1> d_Phi = d_R.log();

		Eigen::Map<Eigen::Matrix<T, 3, 1> > eResiduals(residual);
		eResiduals = Eigen::Matrix<T, 3, 1>::Zero();

		eResiduals = d_Phi;

		eResiduals.applyOnTheLeft(sqrt_information.template cast<T>());

		return true;
	}

	static ceres::CostFunction *Create(Eigen::Vector3d prior_, Eigen::Matrix3d sqrt_information_) {
		return (new ceres::AutoDiffCostFunction<Cost_Initialization_Prior_R, 3, 3>(
						new Cost_Initialization_Prior_R(prior_, std::move(sqrt_information_))));
	}

	Eigen::Vector3d prior;
	Eigen::Matrix3d sqrt_information;
};

#endif //LIO_LIVOX_CERESFUNC_H
