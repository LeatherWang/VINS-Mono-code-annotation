#include "estimator.h"

Estimator::Estimator(): f_manager{Rs} //将Rs的指针传了过去
{
    ROS_INFO("init begins");
    clearState();
    failure_occur = 0;
}

void Estimator::setParameter()
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
    }
    f_manager.setRic(ric);
    ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity(); //!@done 设置优化时候构造重投影误差时，测量的不确定性，即构成协方差
}

void Estimator::clearState()
{
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
        {
            delete pre_integrations[i];
        }

        pre_integrations[i] = nullptr;
    }

    // 把外参也给复位了
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    solver_flag = INITIAL;
    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();
    relocalize = false;
    retrive_data_vector.clear();
    relocalize_t = Eigen::Vector3d(0, 0, 0);
    relocalize_r = Eigen::Matrix3d::Identity();

    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();
}

void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    // 当滑窗不满的时候，把当前测量值加入到滑窗指定位置，所以在这个阶段做预计分的时候相当于是在和自己做预积分
    if (!pre_integrations[frame_count])
    {
        // 构造对象
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }

    if (frame_count != 0)
    {
/*【步骤1】 前后两帧之间的IMU预积分,使用中值法*/
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity); //预积分

        //if(solver_flag != NON_LINEAR)
        tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity); //! @attention 仅仅是两个相邻的图像帧之间预积分，在新进来一幅图像后复位

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        int j = frame_count;

/*【步骤2】 使用IMU测量值估计当前帧的位姿，注意这不是预积分，仅仅借用了IMU的数据估计了当前帧的位姿，注意与步骤1的区别*/
        // IMU积分出来的第 j 时刻的物理量可以作为第 j 帧图像的初始值
        // noise是zero mean Gassu，在这里忽略了
        // 下面都采用的是中值积分的传播方式，noise被忽略了
        // 参考崔的推到:公式(2)(3)
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        // 角度
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        // 位移
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        // 速度
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

void Estimator::processImage(const map<int, vector<pair<int, Vector3d>>> &image, const std_msgs::Header &header)
{
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());

    // 通过`检测两帧之间的视差`和`跟踪成功的数量`决定是否作为关键帧
    // 同时添加之前检测到的特征点到feature容器中，计算每一个点跟踪的次数，以及它的视差
    if (f_manager.addFeatureCheckParallax(frame_count, image))
        marginalization_flag = MARGIN_OLD; //0，是关键帧
    else
        marginalization_flag = MARGIN_SECOND_NEW; //1，不是关键帧

    ROS_DEBUG("this frame is--------------------%s", marginalization_flag ? "reject" : "accept");
    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header;

    ImageFrame imageframe(image, header.stamp.toSec());
    imageframe.pre_integration = tmp_pre_integration; //!从上一帧开始到这一帧到来期间，所有IMU的预积分结果@attention 都是属于当前帧的
    all_image_frame.insert(make_pair(header.stamp.toSec(), imageframe));

    /* 为后面来的IMU数据进行预积分做初始化准备
     * 1.上一帧IMU数据加速度原始值和陀螺仪原始值都给了acc_0和gyr_0
     * 2.接下来IMU使用的bias都取自最新的bias
     * 3.delta_p、delta_v、delta_q都被复位了
     */
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    //! @attention 外参标定
    if(ESTIMATE_EXTRINSIC == 2)
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            // 粗略标定下，后面继续优化
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric)) //! @attention 只标定旋转矩阵
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1; //! @attention 还需要优化，ESTIMATE_EXTRINSIC为0时，固定，不优化
            }
        }
    }

    if (solver_flag == INITIAL)
    {
        // 只有当等于WINDOW_SIZE时，才进行初始化
        if (frame_count == WINDOW_SIZE)
        {
            bool result = false;
            if( ESTIMATE_EXTRINSIC != 2 && (header.stamp.toSec() - initial_timestamp) > 0.1)
            {
               //! @attention 视觉的结构初始化
               result = initialStructure();
               initial_timestamp = header.stamp.toSec();
            }

            if(result)
            {
                solver_flag = NON_LINEAR;
                solveOdometry();
                slideWindow();
                f_manager.removeFailures();
                ROS_INFO("Initialization finish!");
                last_R = Rs[WINDOW_SIZE];
                last_P = Ps[WINDOW_SIZE];
                last_R0 = Rs[0];
                last_P0 = Ps[0];
                
            }
            else
                slideWindow(); //没有初始化成功，就要进行滑窗
        }
        else
            frame_count++; //增加窗口内帧的数量
    }
    else
    {
        TicToc t_solve;
        solveOdometry();
        ROS_DEBUG("solver costs: %fms", t_solve.toc());

        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            return;
        }

        TicToc t_margin;
        slideWindow();
        f_manager.removeFailures();
        ROS_DEBUG("marginalization costs: %fms", t_margin.toc());
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
    }
}


//视觉的结构初始化
//1.确保IMU有足够的excitation
//2.选择跟最后一帧中有足够多特征点和视差的某一帧，利用五点法恢复相对旋转和平移量 恢复某两帧
//3.sfm.construct 全局SFM 恢复滑动窗口的帧的位姿
//4.利用pnp恢复其他帧
//5.开始初始化，视觉IMU数据对齐
//6.给滑动窗口中要优化的变量一个合理的初始值以便进行非线性优化
bool Estimator::initialStructure()
{
    TicToc t_sfm;
    //check imu observibility
/*【步骤1】 通过重力variance确保IMU有足够的excitation*/
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt; //delta_v是仅仅使用加速度测量并没有减去重力g，不过并不影响估计激励
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1); //计算一个均值
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g); //这是个标量，计算三个分量的方差的和，不是协方差
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        //ROS_WARN("IMU variation %f!", var);
        if(var < 0.25)
        {
            ROS_INFO("IMU excitation not enouth!");
            //return false;
        }
    }
    // global sfm
    Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
    
    //特征点对应的3D feature
    vector<SFMFeature> sfm_f; //每一个元素为一个feature，包括其id、观测到它的所有frame编号、在该帧上的投影坐标
    for (auto &it_per_id : f_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    }
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
/*【步骤2】选择跟最后一帧中有足够多特征点和视差的某一帧，利用五点法恢复相对旋转和平移量 */
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }
/*【步骤3】全局SFM初始化全部初始帧中的相机位置和特征点空间3D位置 */
    GlobalSFM sfm;
    if(!sfm.construct(frame_count + 1, Q, T, l,
              relative_R, relative_T,
              sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    //solve pnp for all frame
/*【步骤4】对于非滑动窗口的所有帧，提供一个初始的R,T，然后solve pnp求解pose*/
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin();
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if((frame_it->first) == Headers[i].stamp.toSec()) //在滑窗内的，已经计算过位姿了
        {
            frame_it->second.is_key_frame = true; //转换成关键帧
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose(); //! @attention 转换成IMU坐标系在参考坐标系的坐标: Rwi
            frame_it->second.T = T[i];
            i++;
            continue;
        }

        if((frame_it->first) > Headers[i].stamp.toSec()) //!@todo (貌似是中间的普通帧??，对！就是中间的普通帧，参考slideWindow()函数)
        {
            i++;
        }
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix(); //Rwc->Rcw
        Vector3d P_inital = - R_inital * T[i]; //twc->tcw
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            //cout << "feature id " << feature_id;
            for (auto &i_p : id_pts.second)
            {
                //cout << " pts image_frame " << (i_p.second.head<2>() * 460 ).transpose() << endl;
                it = sfm_tracked_points.find(feature_id); //步骤3最后更新sfm_tracked_points，表示那些被三角化成功的feature
                if(it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);     
        if(pts_3_vector.size() < 6)
        {
            //cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose(); //Rcw->Rwc
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp); //tcw->twc
        frame_it->second.R = R_pnp * RIC[0].transpose(); //Rwc->Rwi
        frame_it->second.T = T_pnp; //twc
    }

/*【步骤5】camera与IMU对齐*/
    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }
}

//Ps:世界坐标下的平移(IMU坐标系在世界坐标系)
//Rs:世界坐标下的旋转
//Vs:世界坐标下的速度

//vision IMU数据对齐
//这里涉及的量有： 陀螺仪的Bias(加速度Bias这里没有处理) 速度V[0:n] 重力g 尺度s
//更新了Bgs后，IMU测量量需要repropagate(重新传递，公式12)
bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    //solve scale
/*【步骤1】求解陀螺仪的Bias(加速度Bias这里没有处理) 速度V[0:n] 重力g 尺度s*/
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if(!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    // change state
/*【步骤2】使用视觉得到位姿变换直接替换IMU的位姿??*/
    // all_image_frame里面会包含一些当MARGIN_SECOND_NEW时添加的普通帧，而Headers是滑窗的大小，所以下面的操作只是取关键帧的相机pose
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i].stamp.toSec()].R; //Rwi，对没错，见330行，w是指上面initialStructure() sfm得到的cx，不是c0
        Vector3d Pi = all_image_frame[Headers[i].stamp.toSec()].T; //pwc，对没错，这里之所以没有转换成pwi，是为了后面步骤3降低运算量
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i].stamp.toSec()].is_key_frame = true;
    }

    VectorXd dep = f_manager.getDepthVector(); //! @attention 哪里来的初值，initialStructure里面会得到一个up to scale的深度
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1;
    f_manager.clearDepth(dep); //将深度全部置成-1

    //triangulat on cam pose , no tic
    Vector3d TIC_TMP[NUM_OF_CAM];
    for(int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero();
    ric[0] = RIC[0];
    f_manager.setRic(ric);
/*【步骤3】三角化得到的是feature在其start_frame上的3D坐标，使用了所有能观测到feature的帧，构造超定矩阵，使用SVD求解*/
    //! @done 为什么令外参的位移为0矩阵TIC_TMP[0]，让IMU坐标系原点与camera重合??，错! 机智啊!
    f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0])); //! @done 因为这个Ps就是twc啊，所以使用的外参的位移为0矩阵TIC_TMP[0]啊，这样就不用来回转换了

    // 重新预积分
    double s = (x.tail<1>())(0);
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]); //更新下滑动窗口的预积分
    }

/*【步骤4】恢复位移的尺度，同时以滑窗第一帧为参考坐标系(准确的说还不是，这里只是位移差，并没有转换到第一帧的坐标系，见步骤7)*/
    // Pc0cy=Pcxcy-Pcxc0
    // cx是与最新的一帧完成初始化成功的那一帧，滑窗中间任一帧都有可能
    // cy是下面的i，滑窗内任一帧
    // c0是滑窗中的第一帧
    /**
      * 这里的步骤将Ps从原来的Ps_cx_ci，先转换到Ps_cx_iy，再都减去Ps_cx_i0，得到`Ps_cx_iy-Ps_cx_i0`，相当于坐标系的原点平移(旋转不变)
      */
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]); //! @attention 转换之前Ps[i]还是pwc，转换之后变成pwi，这种转换可以使用向量头尾相连进行理解

/*【步骤5】恢复关键帧的速度，即，从imu坐标系下转换w(即，cx)坐标系下*/
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3); //Rwi，这里w=cx
        }
    }

/*【步骤6】恢复mapPoint的深度*/
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth *= s;
    }

    /**
     * cx是与最新的一帧初始化成功时使用的那一帧，经过上面的转换后，这里的Ps和Rs的坐标定义如下(i是指IMU):
     * Ps: Ps_cx_i，错，其实已经变成了`Ps_cx_iy-Ps_cx_i0`，见上面【步骤4】
     * Rs: Rs_cx_i
     * 上面使用cx作为w，下面的【步骤7】开始进行转换，以使用惯性坐标系作为w
     * 惯性坐标系的重力向量是[0,0,1]，而该重力向量在cx是另外一个向量，然后就能计算出两个向量对应的两个坐标系之间的旋转矩阵，即下面的`R0=Rwcx`
     */

/*【步骤7】将各个时刻的*IMU位姿*从以cx坐标系(不是c0)为参考转换到以世界坐标系为参考*/
    // 没有必要先从cx转换到c0坐标下，再转换到世界坐标系
    Matrix3d R0 = Utility::g2R(g); //c0指的是滑窗第一帧，cx表示与最新的一帧初始化成功时使用的那一帧，所以`R0=Rwcx`，g也是cx坐标系
    double yaw = Utility::R2ypr(R0 * Rs[0]).x(); //R0 * Rs[0] : Rwi0 = Rwcx*Rcxi0
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0; //为了与第一个IMU坐标系(i0)的yaw方向对齐
    g = R0 * g;
    //Matrix3d rot_diff = R0 * Rs[0].transpose(); //注意，省略的这句话，并没有将所有帧转换到c0坐标系下面，在从Rwc0进行转换到w坐标系下，而是下面这种方法
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i]; //PwExp(-yaw) * Rwcx * (Ps_cx_iy-Ps_cx_i0)，这里的`Ps_cx_iy-Ps_cx_i0`，相当于坐标系的原点平移(旋转不变)
        Rs[i] = rot_diff * Rs[i]; //Exp(-yaw) * Rwcx * Rcxiy
        Vs[i] = rot_diff * Vs[i]; //Exp(-yaw) * Rwcx * viy
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose()); 

    return true;
}

//判断两帧有足够视差
bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE); //取帧i(编号为i)和帧WINDOW_SIZE(编号为WINDOW_SIZE)对应的feature
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;

            // 先算一下视差
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T)) //调用的opencv的库函数，求解E矩阵，并分解
            {
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

void Estimator::solveOdometry()
{
    if (frame_count < WINDOW_SIZE)
        return;
    if (solver_flag == NON_LINEAR)
    {
        TicToc t_tri;
        //Ps是唯一，tic和ric是相机到IMU的外参
        f_manager.triangulate(Ps, tic, ric); //! @todo 这里的还是正确的，visualInitialAlign()里面那个貌似不对吧
        ROS_DEBUG("triangulation costs %f", t_tri.toc());
        optimization();
    }
}

//数据转换，因为ceres使用数值数组
void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = Vs[i].x(); //将速度和bias放在一个vector中，即para_SpeedBias
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);
}

//数据转换，因为ceres使用数值数组，vector2double的相反过程
void Estimator::double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                      para_Pose[0][3],
                                                      para_Pose[0][4],
                                                      para_Pose[0][5]).toRotationMatrix());
    double y_diff = origin_R0.x() - origin_R00.x(); //优化前后计算一下yaw diff
    //TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {

        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
        Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                para_Pose[i][1] - para_Pose[0][1],
                                para_Pose[i][2] - para_Pose[0][2]) + origin_P0;
        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                    para_SpeedBias[i][1],
                                    para_SpeedBias[i][2]);

        Bas[i] = Vector3d(para_SpeedBias[i][3],
                          para_SpeedBias[i][4],
                          para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                          para_SpeedBias[i][7],
                          para_SpeedBias[i][8]);
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d(para_Ex_Pose[i][0],
                          para_Ex_Pose[i][1],
                          para_Ex_Pose[i][2]);
        ric[i] = Quaterniond(para_Ex_Pose[i][6],
                             para_Ex_Pose[i][3],
                             para_Ex_Pose[i][4],
                             para_Ex_Pose[i][5]).toRotationMatrix();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);

    if (LOOP_CLOSURE && relocalize && retrive_data_vector[0].relative_pose && !retrive_data_vector[0].relocalized)
    {
        for (int i = 0; i < (int)retrive_data_vector.size();i++)
            retrive_data_vector[i].relocalized = true;
        Matrix3d vio_loop_r;
        Vector3d vio_loop_t;
        vio_loop_r = rot_diff * Quaterniond(retrive_data_vector[0].loop_pose[6], retrive_data_vector[0].loop_pose[3], retrive_data_vector[0].loop_pose[4], retrive_data_vector[0].loop_pose[5]).normalized().toRotationMatrix();
        vio_loop_t = rot_diff * Vector3d(retrive_data_vector[0].loop_pose[0] - para_Pose[0][0],
                                retrive_data_vector[0].loop_pose[1] - para_Pose[0][1],
                                retrive_data_vector[0].loop_pose[2] - para_Pose[0][2]) + origin_P0;
        Quaterniond vio_loop_q(vio_loop_r);
        double relocalize_yaw;
        relocalize_yaw = Utility::R2ypr(retrive_data_vector[0].R_old).x() - Utility::R2ypr(vio_loop_r).x();
        relocalize_r = Utility::ypr2R(Vector3d(relocalize_yaw, 0, 0));
        relocalize_t = retrive_data_vector[0].P_old- relocalize_r * vio_loop_t;
    }
}

//检测系统是否失败
bool Estimator::failureDetection()
{
    if (f_manager.last_track_num < 2)
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        ROS_INFO(" big translation");
        return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        ROS_INFO(" big z translation");
        return true; 
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        return true;
    }
    return false;
}

/** 基于滑动窗口的紧耦合的非线性优化
 * 1.添加要优化的变量 也就是滑动窗口的位置`para_Pose[0:n]`，速度和Bias`para_SpeedBias[0:n]`，一共15个自由度，IMU的外参也可以加进来估计
 * 2.添加残差，残差项分为4块，先验残差+IMU残差+视觉残差+闭环检测的残差
 * 3.根据倒数第二帧是不是关键帧确定marginization的结果，下面有详细解释
*/
void Estimator::optimization()
{
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    //loss_function = new ceres::HuberLoss(1.0);
    loss_function = new ceres::CauchyLoss(1.0);

/*【步骤1:】添加优化节点*/
    //! @attention 注意这里并没有mapPoint的3D位置，因为VINS里面不存在这个mapPoint，只存在在第一个观测到该feature的帧中的深度
    //! 为什么不在problem中加入这个para_Feature呢，不优化??，如果不优化直接设置成SetParameterBlockConstant就可以了吗，那就是还要优化??
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    //camera IMU的外参也添加到估计
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if (!ESTIMATE_EXTRINSIC)
        {
            ROS_DEBUG("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]); //将外参设置为常量，不优化
        }
        else
            ROS_DEBUG("estimate extinsic param");
    }

    TicToc t_whole, t_prepare;

/*【步骤2:】转换数据，给节点指向的内存赋予初值*/
    // 添加sliding window frame的state，(pose,v,q,ba,bg),因为ceres用的是double数组，所以在下面用vector2double做类型装换，
    // 把原来的Ps、Vs、Bgs、Bas转到para_Pose、para_SpeedBias下
    vector2double();

/*【步骤3:】添加marginalization的residual*/
    //这个marginalization的结构是始终存在的，随着下面marginazation的结构更新，last_marginalization_parameter_blocks对应的是还在sliding window的变量
    //关于这一部分的理解，请看heyijia大神的BLOG:http://blog.csdn.net/heyijia0327/article/details/53707261,可能是说的最清楚的一个了
    // 这里可以这样理解，下面会添加对IMU和视觉的残差，但是，这些对应的变量实际上跟之前被margin掉的变量是有约束的，
    // 这里的`last_marginalization_parameter_blocks`就是保存的这些变量，也就是heyijia博客中对应的`Xb`变量，
    // `last_marginalization_info`中对应的是Xb对应的测量Zb的，这里用先验来表示这个约束，整个margin部分实际上就是在维护这个结构：
    if (last_marginalization_info)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks); //heyijia博客中对应的`Xb`变量，即保留部分(去掉marg部分)
    }
    
/*【步骤4:】添加IMU的residual*/
    // 这里IMU项和camera项之间是有一个系数，这个系数就是他们各自的协方差矩阵：IMU的协方差是预积分的协方差(IMUFactor::Evaluate，中添加IMU协方差，求解jacobian矩阵)，
    // 而camera的测量残差则是一个固定的系数（f/1.5）
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;
        IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
    }
    int f_m_cnt = 0;
    int feature_index = -1;
/*【步骤5:】添加vision的residual*/
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
 
        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i == imu_j)
            {
                continue;
            }
            Vector3d pts_j = it_per_frame.point;
            ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
            problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
            f_m_cnt++;
        }
    }


    relocalize = false;
    //loop close factor
    //添加闭环的参数和residual
    if(LOOP_CLOSURE)
    {
        int loop_constraint_num = 0;
        for (int k = 0; k < (int)retrive_data_vector.size(); k++)
        {    
            for(int i = 0; i < WINDOW_SIZE; i++)
            {
                if(retrive_data_vector[k].header == Headers[i].stamp.toSec())
                {
                    relocalize = true;
                    ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
                    problem.AddParameterBlock(retrive_data_vector[k].loop_pose, SIZE_POSE, local_parameterization);
                    loop_window_index = i;
                    loop_constraint_num++;
                    int retrive_feature_index = 0;
                    int feature_index = -1;
                    for (auto &it_per_id : f_manager.feature)
                    {
                        it_per_id.used_num = it_per_id.feature_per_frame.size();
                        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                            continue;

                        ++feature_index;
                        int start = it_per_id.start_frame;
                        if(start <= i)
                        {   
                            while(retrive_data_vector[k].features_ids[retrive_feature_index] < it_per_id.feature_id)
                            {
                                retrive_feature_index++;
                            }

                            if(retrive_data_vector[k].features_ids[retrive_feature_index] == it_per_id.feature_id)
                            {
                                Vector3d pts_j = Vector3d(retrive_data_vector[k].measurements[retrive_feature_index].x, retrive_data_vector[k].measurements[retrive_feature_index].y, 1.0);
                                Vector3d pts_i = it_per_id.feature_per_frame[0].point;
                                
                                ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                                problem.AddResidualBlock(f, loss_function, para_Pose[start], retrive_data_vector[k].loop_pose, para_Ex_Pose[0], para_Feature[feature_index]);
                            
                                retrive_feature_index++;
                            }     
                        }
                    }
                            
                }
            }
        }
        ROS_DEBUG("loop constraint num: %d", loop_constraint_num);
    }
    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    ROS_DEBUG("prepare for ceres: %f", t_prepare.toc());

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG; //信任区域
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0; //关键帧优化时间更短???
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    ROS_DEBUG("solver costs: %f", t_solver.toc());

    // relative info between two loop frame
    if(LOOP_CLOSURE && relocalize)
    { 
        for (int k = 0; k < (int)retrive_data_vector.size(); k++)
        {
            for(int i = 0; i< WINDOW_SIZE; i++)
            {
                if(retrive_data_vector[k].header == Headers[i].stamp.toSec())
                {
                    retrive_data_vector[k].relative_pose = true;
                    Matrix3d Rs_i = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
                    Vector3d Ps_i = Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
                    Quaterniond Qs_loop;
                    Qs_loop = Quaterniond(retrive_data_vector[k].loop_pose[6],  retrive_data_vector[k].loop_pose[3],  retrive_data_vector[k].loop_pose[4],  retrive_data_vector[k].loop_pose[5]).normalized().toRotationMatrix();
                    Matrix3d Rs_loop = Qs_loop.toRotationMatrix();
                    Vector3d Ps_loop = Vector3d( retrive_data_vector[k].loop_pose[0],  retrive_data_vector[k].loop_pose[1],  retrive_data_vector[k].loop_pose[2]);

                    retrive_data_vector[k].relative_t = Rs_loop.transpose() * (Ps_i - Ps_loop);
                    retrive_data_vector[k].relative_q = Rs_loop.transpose() * Rs_i;
                    retrive_data_vector[k].relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs_i).x() - Utility::R2ypr(Rs_loop).x());
                    if (abs(retrive_data_vector[k].relative_yaw) > 30.0 || retrive_data_vector[k].relative_t.norm() > 20.0)
                        retrive_data_vector[k].relative_pose = false;
                        
                }
            } 
        } 
    }

    double2vector();

    TicToc t_whole_marginalization;
    //margin部分，如果倒数第二帧是关键帧：
    //1.把之前的存的残差部分加进来
    //2.把与当前要margin掉帧所有相关的残差项都加进来，IMU,vision
    //3.preMarginalize-> 调用Evaluate计算所有ResidualBlock的残差，
    //  parameter_block_data parameter_block_idx parameter_block_size是marinazation中存参数块的容器(unordered_map),key都是addr,
    //  分别对应这些参数的data，在稀疏矩阵A中的index(要被margin掉的参数会被移到前面)，A中的大小
    //4.Marginalize->多线程构造Hx=b的结构，H是边缘化后的结果，First Esitimate Jacobian,在X0处线性化
    //5.margin结束，调整参数块在下一次window中对应的位置（往前移一格），注意这里是指针，后面slideWindow中会赋新值，
    //  这里只是提前占座（知乎上有人问：https://www.zhihu.com/question/63754583/answer/259699612）
    if (marginalization_flag == MARGIN_OLD)
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();

        if (last_marginalization_info)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                //将最老的一帧对应的优化变量放入待marg的vector中
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set); //para_Pose[0]和para_SpeedBias[0]

            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        // 对最老一帧的marg，帧0和帧1
        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info =
                        new ResidualBlockInfo(imu_factor, NULL,
                                              vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                              vector<int>{0, 1}); //marg掉0、1，对，这里0、1指得是para_Pose[0], para_SpeedBias[0]
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        // 对feature的marg选择，第一个观测到该feature的帧和该feature
        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if (imu_i == imu_j)
                        continue;

                    Vector3d pts_j = it_per_frame.point;
                    ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                    ResidualBlockInfo *residual_block_info =
                            new ResidualBlockInfo(f, loss_function,
                                                  vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                                                  vector<int>{0, 3}); //0、3指得是para_Pose[imu_i]、para_Feature[feature_index]
                                                  //! @todo 为什么要将para_Pose[imu_i]、para_Feature[feature_index]也marg掉，不是只marg掉最老的一帧吗
                    marginalization_info->addResidualBlockInfo(residual_block_info);
                }
            }
        }

        TicToc t_pre_margin;
        //计算雅可比、拷贝参数块，为下面的marginalize()函数做准备
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());
        
        TicToc t_margin;
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        //这里不是赋值哈，形象的来说就是给下一次占个座，往前移一格
        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info; //析构上次的
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
        
    }
    else
    {
    //如果倒数第二帧不是关键帧
    // 1.保留该帧的IMU测量，margin该帧的visual
    // 2.premargin
    // 3.marginalize
    // 4.滑动窗口移动（去掉倒数第二个）
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks),
                       std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    //这是干毛呢?`para_SpeedBias[WINDOW_SIZE - 1]`不会出现??
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);

                    //! 果不其然，是在寻找次新的帧，@todo 为什么没有`para_SpeedBias[WINDOW_SIZE - 1]`
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());
            
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
        }
    }
    ROS_DEBUG("whole marginalization costs: %f", t_whole_marginalization.toc());
    
    ROS_DEBUG("whole time for ceres: %f", t_whole.toc());
}

// 实际滑动窗口的地方，如果第二最新帧是关键帧的话，那么这个关键帧就会留在滑动窗口中，时间最长的一帧和其测量值就会被边缘化掉，
// 如果第二最新帧不是关键帧的话，则把这帧的视觉测量舍弃掉而保留IMU测量值在滑动窗口中这样的策略会保证系统的稀疏性
void Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD) //需要把当前帧增加为关键帧
    {
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            //【1】
            //! @todo 数据进行滑动，为什么要使用交换的方式呢，这种效率不高吧??
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Rs[i].swap(Rs[i + 1]);

                std::swap(pre_integrations[i], pre_integrations[i + 1]);

                dt_buf[i].swap(dt_buf[i + 1]);
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                Headers[i] = Headers[i + 1];
                Ps[i].swap(Ps[i + 1]);
                Vs[i].swap(Vs[i + 1]);
                Bas[i].swap(Bas[i + 1]);
                Bgs[i].swap(Bgs[i + 1]);
            }
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1]; //! @attention 最后一个用前一个代替，这样是为了给IMU预积分时提供初值
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

            //【2】
            //! @attention 将编号为WINDOW_SIZE的帧位置空出来，后面新来的帧都是用这个编号
            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]}; //!@attention 下次将使用最新的bias

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            //【3】all_image_frame相关
            if (true || solver_flag == INITIAL)
            {
                double t_0 = Headers[0].stamp.toSec();
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration; //! @todo 预积分为什么不要了??
                all_image_frame.erase(all_image_frame.begin(), it_0); //[first,last)，擦除all_image_frame中在it_0之前的所有帧
            }

            //【4】feature相关
            slideWindowOld(); //! @attention @todo 对看到该feature所有帧的编号进行滑动
        }
    }
    else //不是关键帧
    {
        if (frame_count == WINDOW_SIZE)
        {
            //! @attention 保留IMU的数据
            // 因为次新的帧被marg掉了，所以将frame_count的数据放到frame_count - 1的位置
            for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
            {
                double tmp_dt = dt_buf[frame_count][i];
                Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                // 使用的是push_back，所以就是向pre_integrations[frame_count - 1]增加IMU数据
                pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                dt_buf[frame_count - 1].push_back(tmp_dt);
                linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
            }

            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Vs[frame_count - 1] = Vs[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];
            Bas[frame_count - 1] = Bas[frame_count];
            Bgs[frame_count - 1] = Bgs[frame_count];

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            slideWindowNew();
        }
    }
}

//根据margin滑动窗口结果处理特征点出现的帧号
//! @attention real marginalization is removed in solve_ceres()
void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}
//! @attention real marginalization is removed in solve_ceres()
void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth) //VIO阶段
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0]; //back_P0、back_R0是刚才marg掉的那一帧的位姿
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1); ///MARGIN_OLD时，调用，如果初始化成功或者VIO阶段调用，与下面的函数的区别在于对深度是否处理
    }
    else //MARGIN_OLD时，调用，初始化阶段(没有初始化成功呢，如果初始化成功就和VIO计算一样的处理了)调用，VIO阶段调用上面的函数，区别在于对深度是否处理
        f_manager.removeBack();

}

