#include <rclcpp/rclcpp.hpp>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <memory>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include "onboard_detector/dynamicDetector.h"


namespace onboardDetector {


DynamicDetector::DynamicDetector() : Node("dynamic_detector"), ns_("onboard_detector"), hint_("[onboardDetector]")
{
}

DynamicDetector::DynamicDetector(const rclcpp::NodeOptions & options) : Node("dynamic_detector", options), ns_("onboard_detector"), hint_("[onboardDetector]")
{
}

void DynamicDetector::initDetector(){
    RCLCPP_INFO(this->get_logger(), "%s: Init of dynamic_detector", hint_.c_str());
    this->initParam();
    this->registerPub();
    this->registerCallback();
}



void DynamicDetector::initParam()
{
    

    this->declare_parameter<int>(ns_ + ".localization_mode", 0);
    if (!this->get_parameter(ns_ + ".localization_mode", localizationMode_)) {
        localizationMode_ = 0;
        RCLCPP_INFO(this->get_logger(), "%s:localization_mode not provided, using default: 0 (pose)", hint_.c_str());
    } else {
        RCLCPP_INFO(this->get_logger(), "%s: Localization mode: %d", hint_.c_str(), localizationMode_);
    }

    this->declare_parameter<std::string>(ns_ + ".depth_image_topic", "/camera/depth/image_raw");
    this->get_parameter(ns_ + ".depth_image_topic", depthTopicName_);
    RCLCPP_INFO(this->get_logger(), "%s: Depth topic: %s", hint_.c_str(), depthTopicName_.c_str());

    this->declare_parameter<std::string>(ns_ + ".aligned_depth_image_topic", "/camera/aligned_depth_to_color/image_raw");
    this->get_parameter(ns_ + ".aligned_depth_image_topic", alignedDepthTopicName_);
    RCLCPP_INFO(this->get_logger(), "%s: Aligned depth topic: %s", hint_.c_str(), alignedDepthTopicName_.c_str());

    if (localizationMode_ == 0) {
        this->declare_parameter<std::string>(ns_ + ".pose_topic", "/CERLAB/quadcopter/pose");
        this->get_parameter(ns_ + ".pose_topic", poseTopicName_);
        RCLCPP_INFO(this->get_logger(), "%s: Pose topic: %s", hint_.c_str(), poseTopicName_.c_str());
    } else if (localizationMode_ == 1) {
        this->declare_parameter<std::string>(ns_ + ".odom_topic", "/CERLAB/quadcopter/odom");
        this->get_parameter(ns_ + ".odom_topic", odomTopicName_);
        RCLCPP_INFO(this->get_logger(), "%s: Odom topic: %s", hint_.c_str(), odomTopicName_.c_str());
    }

    this->declare_parameter<std::vector<double>>(ns_ + ".depth_intrinsics", std::vector<double>{0.0, 0.0, 0.0, 0.0});
    std::vector<double> depthIntrinsics;
    this->get_parameter(ns_ + ".depth_intrinsics", depthIntrinsics);
    if (depthIntrinsics.size() != 4) {
        RCLCPP_ERROR(this->get_logger(), "%s: Please verify intrinsic parameters of depth sensor", hint_.c_str());
        rclcpp::shutdown();
    } else {
        fx_ = depthIntrinsics[0];
        fy_ = depthIntrinsics[1];
        cx_ = depthIntrinsics[2];
        cy_ = depthIntrinsics[3];
        RCLCPP_INFO(this->get_logger(), "%s: fx, fy, cx, cy: [%f, %f, %f, %f]", hint_.c_str(), fx_, fy_, cx_, cy_);
    }

    this->declare_parameter<double>(ns_ + ".depth_scale_factor", 1000.0);
    this->get_parameter(ns_ + ".depth_scale_factor", depthScale_);
    RCLCPP_INFO(this->get_logger(), "%s: Depth scale factor: %f", hint_.c_str(), depthScale_);

    this->declare_parameter<double>(ns_ + ".depth_min_value", 0.2);
    this->get_parameter(ns_ + ".depth_min_value", depthMinValue_);
    RCLCPP_INFO(this->get_logger(), "%s: Depth min value: %f", hint_.c_str(), depthMinValue_);

    this->declare_parameter<double>(ns_ + ".depth_max_value", 5.0);
    this->get_parameter(ns_ + ".depth_max_value", depthMaxValue_);
    RCLCPP_INFO(this->get_logger(), "%s: Depth max value: %f", hint_.c_str(), depthMaxValue_);

    this->declare_parameter<int>(ns_ + ".depth_filter_margin", 0);
    this->get_parameter(ns_ + ".depth_filter_margin", depthFilterMargin_);
    RCLCPP_INFO(this->get_logger(), "%s: Depth filter margin: %f", hint_.c_str(), depthFilterMargin_);

    this->declare_parameter<int>(ns_ + ".depth_skip_pixel", 1);
    this->get_parameter(ns_ + ".depth_skip_pixel", skipPixel_);
    RCLCPP_INFO(this->get_logger(), "%s: Depth skip pixel: %d", hint_.c_str(), skipPixel_);

    // --- Rozmiary obrazu ---
    this->declare_parameter<int>(ns_ + ".image_cols", 640);
    this->get_parameter(ns_ + ".image_cols", imgCols_);
    RCLCPP_INFO(this->get_logger(), "%s: Depth image columns: %d", hint_.c_str(), imgCols_);

    this->declare_parameter<int>(ns_ + ".image_rows", 480);
    this->get_parameter(ns_ + ".image_rows", imgRows_);
    RCLCPP_INFO(this->get_logger(), "%s: Depth image rows: %d", hint_.c_str(), imgRows_);

    projPoints_.resize(imgCols_ * imgRows_ / (skipPixel_ * skipPixel_));
    pointsDepth_.resize(imgCols_ * imgRows_ / (skipPixel_ * skipPixel_));

    this->declare_parameter<std::vector<double>>(ns_ + ".body_to_camera", std::vector<double>(16, 0.0));
    std::vector<double> body2CamVec;
    this->get_parameter(ns_ + ".body_to_camera", body2CamVec);
    if (body2CamVec.size() != 16) {
        RCLCPP_ERROR(this->get_logger(), "Please verify body_to_camera transform");
    } else {
            body2Cam_.resize(4, 4);
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    body2Cam_(i, j) = body2CamVec[i * 4 + j];
            }
        }
        RCLCPP_INFO(this->get_logger(), "%s: Transform body_to_camera has been set", hint_.c_str());
    }
    this->declare_parameter<std::vector<double>>(ns_ + ".color_intrinsics", std::vector<double>(4, 0.0));
    std::vector<double> colorIntrinsics;
    this->get_parameter(ns_ + ".color_intrinsics", colorIntrinsics);
    if (colorIntrinsics.size() != 4) {
      RCLCPP_ERROR(this->get_logger(), "%s: Please verify intrinsic parameters of rgb camera", hint_.c_str());
      rclcpp::shutdown();
    } else {
      fxC_ = colorIntrinsics[0];
      fyC_ = colorIntrinsics[1];
      cxC_ = colorIntrinsics[2];
      cyC_ = colorIntrinsics[3];
      RCLCPP_INFO(this->get_logger(), "%s: fxC, fyC, cxC, cyC: [%f, %f, %f, %f]", hint_.c_str(), fxC_, fyC_, cxC_, cyC_);
    }

    this->declare_parameter<std::vector<double>>(ns_ + ".body_to_camera_color", std::vector<double>(16, 0.0));
    std::vector<double> body2CamColorVec;
    this->get_parameter(ns_ + ".body_to_camera_color", body2CamColorVec);
    if (body2CamColorVec.size() != 16) {
      RCLCPP_ERROR(this->get_logger(), "Please verify body_to_camera_color transform");
    } else {
      body2CamColor_.resize(4, 4);
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          body2CamColor_(i, j) = body2CamColorVec[i * 4 + j];
        }
      }
    }

    this->declare_parameter<double>(ns_ + ".raycast_max_length", 5.0);
    this->get_parameter(ns_ + ".raycast_max_length", raycastMaxLength_);
    RCLCPP_INFO(this->get_logger(), "%s: Raycast max length: %f", hint_.c_str(), raycastMaxLength_);

    this->declare_parameter<int>(ns_ + ".voxel_occupied_thresh", 10);
    this->get_parameter(ns_ + ".voxel_occupied_thresh", voxelOccThresh_);
    RCLCPP_INFO(this->get_logger(), "%s: Voxel occupied threshold: %d", hint_.c_str(), voxelOccThresh_);

    this->declare_parameter<double>(ns_ + ".ground_height", 0.1);
    this->get_parameter(ns_ + ".ground_height", groundHeight_);
    RCLCPP_INFO(this->get_logger(), "%s: Ground height: %f", hint_.c_str(), groundHeight_);

    this->declare_parameter<int>(ns_ + ".dbscan_min_points_cluster", 18);
    this->get_parameter(ns_ + ".dbscan_min_points_cluster", dbMinPointsCluster_);
    RCLCPP_INFO(this->get_logger(), "%s: DBSCAN min points per cluster: %d", hint_.c_str(), dbMinPointsCluster_);

    this->declare_parameter<double>(ns_ + ".dbscan_search_range_epsilon", 0.3);
    this->get_parameter(ns_ + ".dbscan_search_range_epsilon", dbEpsilon_);
    RCLCPP_INFO(this->get_logger(), "%s: DBSCAN epsilon: %f", hint_.c_str(), dbEpsilon_);

    this->declare_parameter<double>(ns_ + ".filtering_BBox_IOU_threshold", 0.5);
    this->get_parameter(ns_ + ".filtering_BBox_IOU_threshold", boxIOUThresh_);
    RCLCPP_INFO(this->get_logger(), "%s: BBox IOU threshold: %f", hint_.c_str(), boxIOUThresh_);

    this->declare_parameter<double>(ns_ + ".yolo_overwrite_distance", 3.5);
    this->get_parameter(ns_ + ".yolo_overwrite_distance", yoloOverwriteDistance_);
    RCLCPP_INFO(this->get_logger(), "%s: YOLO overwrite distance: %f", hint_.c_str(), yoloOverwriteDistance_);

    this->declare_parameter<int>(ns_ + ".history_size", 5);
    this->get_parameter(ns_ + ".history_size", histSize_);
    RCLCPP_INFO(this->get_logger(), "%s: History size: %d", hint_.c_str(), histSize_);

    this->declare_parameter<double>(ns_ + ".time_difference", 0.033);
    this->get_parameter(ns_ + ".time_difference", dt_);
    RCLCPP_INFO(this->get_logger(), "%s: Time difference: %f", hint_.c_str(), dt_);

    this->declare_parameter<double>(ns_ + ".similarity_threshold", 0.9);
    this->get_parameter(ns_ + ".similarity_threshold", simThresh_);
    RCLCPP_INFO(this->get_logger(), "%s: Similarity threshold: %f", hint_.c_str(), simThresh_);

    this->declare_parameter<int>(ns_ + ".frame_skip", 5);
    this->get_parameter(ns_ + ".frame_skip", skipFrame_);
    RCLCPP_INFO(this->get_logger(), "%s: Frame skip: %d", hint_.c_str(), skipFrame_);

    this->declare_parameter<double>(ns_ + ".dynamic_velocity_threshold", 0.35);
    this->get_parameter(ns_ + ".dynamic_velocity_threshold", dynaVelThresh_);
    RCLCPP_INFO(this->get_logger(), "%s: Dynamic velocity threshold: %f", hint_.c_str(), dynaVelThresh_);

    this->declare_parameter<double>(ns_ + ".dynamic_voting_threshold", 0.8);
    this->get_parameter(ns_ + ".dynamic_voting_threshold", dynaVoteThresh_);
    RCLCPP_INFO(this->get_logger(), "%s: Dynamic voting threshold: %f", hint_.c_str(), dynaVoteThresh_);

    this->declare_parameter<double>(ns_ + ".maximum_skip_ratio", 0.5);
    this->get_parameter(ns_ + ".maximum_skip_ratio", maxSkipRatio_);
    RCLCPP_INFO(this->get_logger(), "%s: Maximum skip ratio: %f", hint_.c_str(), maxSkipRatio_);

    this->declare_parameter<int>(ns_ + ".fix_size_history_threshold", 10);
    this->get_parameter(ns_ + ".fix_size_history_threshold", fixSizeHistThresh_);
    RCLCPP_INFO(this->get_logger(), "%s: Fix size history threshold: %d", hint_.c_str(), fixSizeHistThresh_);

    this->declare_parameter<double>(ns_ + ".fix_size_dimension_threshold", 0.4);
    this->get_parameter(ns_ + ".fix_size_dimension_threshold", fixSizeDimThresh_);
    RCLCPP_INFO(this->get_logger(), "%s: Fix size dimension threshold: %f", hint_.c_str(), fixSizeDimThresh_);

    this->declare_parameter<double>(ns_ + ".e_p", 0.5);
    this->get_parameter(ns_ + ".e_p", eP_);
    RCLCPP_INFO(this->get_logger(), "%s: Covariance for Kalman Filter (e_p): %f", hint_.c_str(), eP_);

    this->declare_parameter<double>(ns_ + ".e_q_pos", 0.5);
    this->get_parameter(ns_ + ".e_q_pos", eQPos_);
    RCLCPP_INFO(this->get_logger(), "%s: Noise for prediction for position (e_q_pos): %f", hint_.c_str(), eQPos_);

    this->declare_parameter<double>(ns_ + ".e_q_vel", 0.5);
    this->get_parameter(ns_ + ".e_q_vel", eQVel_);
    RCLCPP_INFO(this->get_logger(), "%s: Noise for prediction for velocity (e_q_vel): %f", hint_.c_str(), eQVel_);

    this->declare_parameter<double>(ns_ + ".e_q_acc", 0.5);
    this->get_parameter(ns_ + ".e_q_acc", eQAcc_);
    RCLCPP_INFO(this->get_logger(), "%s: Noise for prediction for acceleration (e_q_acc): %f", hint_.c_str(), eQAcc_);

    this->declare_parameter<double>(ns_ + ".e_r_pos", 0.5);
    this->get_parameter(ns_ + ".e_r_pos", eRPos_);
    RCLCPP_INFO(this->get_logger(), "%s: Noise for measurement for position (e_r_pos): %f", hint_.c_str(), eRPos_);

}

void DynamicDetector::registerPub(){

    // Create image transport
    std::shared_ptr<image_transport::ImageTransport> it =
    std::make_shared<image_transport::ImageTransport>(this->shared_from_this());
    
    // Image publishers
    this->uvDepthMapPub_ = it->advertise(this->ns_ + "/detected_depth_map", 1);
    this->uDepthMapPub_ = it->advertise(this->ns_ + "/detected_u_depth_map", 1);
    this->uvBirdViewPub_ = it->advertise(this->ns_ + "/bird_view", 1);
    this->detectedAlignedDepthImgPub_ = it->advertise(this->ns_ + "/detected_aligned_depth_map_yolo", 1);

    // MarkerArray publishers
    this->uvBBoxesPub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        this->ns_ + "/uv_bboxes", 10);
    this->dbBBoxesPub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        this->ns_ + "/dbscan_bboxes", 10);
    this->yoloBBoxesPub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        this->ns_ + "/yolo_3d_bboxes", 10);
    this->filteredBBoxesPub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        this->ns_ + "/filtered_bboxes", 10);
    this->trackedBBoxesPub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        this->ns_ + "/tracked_bboxes", 10);
    this->dynamicBBoxesPub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        this->ns_ + "/dynamic_bboxes", 10);
    this->historyTrajPub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        this->ns_ + "/history_trajectories", 10);
    this->velVisPub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        this->ns_ + "/velocity_visualization", 10);

    // PointCloud2 publishers
    this->dynamicPointsPub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        this->ns_ + "/dynamic_point_cloud", 10);
    this->filteredPointsPub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        this->ns_ + "/filtered_depth_cloud", 10);
}

void DynamicDetector::registerCallback() {
    // depth subscriber (message_filters)
    this->depthSub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
        shared_from_this(), this->depthTopicName_, rmw_qos_profile_sensor_data);

    if (this->localizationMode_ == 0) {
        // pose subscriber
        this->poseSub_ = std::make_shared<message_filters::Subscriber<geometry_msgs::msg::PoseStamped>>(
            shared_from_this(), this->poseTopicName_, rmw_qos_profile_sensor_data);

        // synchronizer
        this->depthPoseSync_ = std::make_shared<message_filters::Synchronizer<depthPoseSync>>(
            depthPoseSync(100), *this->depthSub_, *this->poseSub_);

        this->depthPoseSync_->registerCallback(
            std::bind(&DynamicDetector::depthPoseCB, this, std::placeholders::_1, std::placeholders::_2));
    }
    else if (this->localizationMode_ == 1) {
        // odometry subscriber
        this->odomSub_ = std::make_shared<message_filters::Subscriber<nav_msgs::msg::Odometry>>(
            shared_from_this(), this->odomTopicName_, rmw_qos_profile_sensor_data);

        // synchronizer
        this->depthOdomSync_ = std::make_shared<message_filters::Synchronizer<depthOdomSync>>(
            depthOdomSync(100), *this->depthSub_, *this->odomSub_);

        this->depthOdomSync_->registerCallback(
            std::bind(&DynamicDetector::depthOdomCB, this, std::placeholders::_1, std::placeholders::_2));
    }
    else {
        RCLCPP_ERROR(this->get_logger(), "[DynamicDetector]: Invalid localization mode!");
        rclcpp::shutdown();
        return;
    }

    // aligned depth subscription
    this->alignedDepthSub_ = this->create_subscription<sensor_msgs::msg::Image>(
        this->alignedDepthTopicName_, 10,
        std::bind(&DynamicDetector::alignedDepthCB, this, std::placeholders::_1));

    // YOLO detection results subscription
    this->yoloDetectionSub_ = this->create_subscription<vision_msgs::msg::Detection2DArray>(
        "yolo_detector/detected_bounding_boxes",
        rclcpp::QoS(10),
        std::bind(&DynamicDetector::yoloDetectionCB, this, std::placeholders::_1)
    );

    // detection timer
    this->detectionTimer_ = this->create_wall_timer(
        std::chrono::duration<double>(this->dt_),
        std::bind(&DynamicDetector::detectionCB, this)
    );
    
    // tracking timer
    this->trackingTimer_ = this->create_wall_timer(
        std::chrono::duration<double>(this->dt_),
        std::bind(&DynamicDetector::trackingCB, this)
    );

    // classification timer
    this->classificationTimer_ = this->create_wall_timer(
        std::chrono::duration<double>(this->dt_),
        std::bind(&DynamicDetector::classificationCB, this));

    // visualization timer
    this->visTimer_ = this->create_wall_timer(
        std::chrono::duration<double>(this->dt_),
        std::bind(&DynamicDetector::visCB, this));
}

void DynamicDetector::depthPoseCB(
    const sensor_msgs::msg::Image::ConstSharedPtr img,
    const geometry_msgs::msg::PoseStamped::ConstSharedPtr pose)
{
    // store current depth image
    cv_bridge::CvImagePtr imgPtr = cv_bridge::toCvCopy(img, img->encoding);
    if (img->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
        imgPtr->image.convertTo(imgPtr->image, CV_16UC1, this->depthScale_);
    }
    imgPtr->image.copyTo(this->depthImage_);

    // store current position and orientation (camera)
    Eigen::Matrix4d camPoseMatrix, camPoseColorMatrix;
    this->getCameraPose(pose, camPoseMatrix, camPoseColorMatrix);

    this->position_(0) = camPoseMatrix(0, 3);
    this->position_(1) = camPoseMatrix(1, 3);
    this->position_(2) = camPoseMatrix(2, 3);
    this->orientation_ = camPoseMatrix.block<3, 3>(0, 0);

    this->positionColor_(0) = camPoseColorMatrix(0, 3);
    this->positionColor_(1) = camPoseColorMatrix(1, 3);
    this->positionColor_(2) = camPoseColorMatrix(2, 3);
    this->orientationColor_ = camPoseColorMatrix.block<3, 3>(0, 0);
}

void DynamicDetector::depthOdomCB(
    const sensor_msgs::msg::Image::ConstSharedPtr img,
    const nav_msgs::msg::Odometry::ConstSharedPtr odom)
{
    // store current depth image
    cv_bridge::CvImagePtr imgPtr = cv_bridge::toCvCopy(img, img->encoding);
    if (img->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
        imgPtr->image.convertTo(imgPtr->image, CV_16UC1, this->depthScale_);
    }
    imgPtr->image.copyTo(this->depthImage_);

    // store current position and orientation (camera)
    Eigen::Matrix4d camPoseMatrix, camPoseColorMatrix;
    this->getCameraPose(odom, camPoseMatrix, camPoseColorMatrix);

    this->position_(0) = camPoseMatrix(0, 3);
    this->position_(1) = camPoseMatrix(1, 3);
    this->position_(2) = camPoseMatrix(2, 3);
    this->orientation_ = camPoseMatrix.block<3, 3>(0, 0);

    this->positionColor_(0) = camPoseColorMatrix(0, 3);
    this->positionColor_(1) = camPoseColorMatrix(1, 3);
    this->positionColor_(2) = camPoseColorMatrix(2, 3);
    this->orientationColor_ = camPoseColorMatrix.block<3, 3>(0, 0);
}

void DynamicDetector::alignedDepthCB(const sensor_msgs::msg::Image::ConstSharedPtr img)
{
    cv_bridge::CvImagePtr imgPtr = cv_bridge::toCvCopy(img, img->encoding);
    if (img->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
        imgPtr->image.convertTo(imgPtr->image, CV_16UC1, this->depthScale_);
    }
    imgPtr->image.copyTo(this->alignedDepthImage_);

    cv::Mat depthNormalized;
    imgPtr->image.copyTo(depthNormalized);
    double min, max;
    cv::minMaxIdx(depthNormalized, &min, &max);
    cv::convertScaleAbs(depthNormalized, depthNormalized, 255.0 / max);
    depthNormalized.convertTo(depthNormalized, CV_8UC1);
    cv::applyColorMap(depthNormalized, depthNormalized, cv::COLORMAP_BONE);
    this->detectedAlignedDepthImg_ = depthNormalized;
}

void DynamicDetector::yoloDetectionCB(const vision_msgs::msg::Detection2DArray::SharedPtr detections) {
    this->yoloDetectionResults_ = *detections;
}


void DynamicDetector::detectionCB()
{
    this->dbscanDetect();
    this->uvDetect();
    this->yoloDetectionTo3D();
    this->filterBBoxes();
    this->newDetectFlag_ = true;
}

void DynamicDetector::trackingCB()
{
    std::vector<int> bestMatch;
    this->boxAssociation(bestMatch);

    if (!bestMatch.empty()) {
        this->kalmanFilterAndUpdateHist(bestMatch);
    } else {
        this->boxHist_.clear();
        this->pcHist_.clear();
    }
}

void DynamicDetector::classificationCB()
{
    // Identification thread
    std::vector<onboardDetector::box3D> dynamicBBoxesTemp;

    // Iterate through all pointcloud/bounding boxes history (note that yolo's pointclouds are dummy pointcloud (empty))
    // NOTE: There are 3 cases which we don't need to perform dynamic obstacle identification.
    for (size_t i = 0; i < this->pcHist_.size(); ++i)
    {
        // ===================================================================================
        // CASE I: yolo recognized as dynamic dynamic obstacle
        if (this->boxHist_[i][0].is_human)
        {
            dynamicBBoxesTemp.push_back(this->boxHist_[i][0]);
            continue;
        }
        // ===================================================================================


        // ===================================================================================
        // CASE II: history length is not enough to run classification
        int curFrameGap;
        if (int(this->pcHist_[i].size()) < this->skipFrame_ + 1)
        {
            curFrameGap = this->pcHist_[i].size() - 1;
        }
        else
        {
            curFrameGap = this->skipFrame_;
        }
        // ===================================================================================


        // ==================================================================================
        // CASE III: Force Dynamic (if the obstacle is classifed as dynamic for several time steps)
        int dynaFrames = 0;
        if (int(this->boxHist_[i].size()) > this->forceDynaCheckRange_)
        {
            for (int j = 1; j < this->forceDynaCheckRange_ + 1; ++j)
            {
                if (this->boxHist_[i][j].is_dynamic)
                {
                    ++dynaFrames;
                }
            }
        }

        if (dynaFrames >= this->forceDynaFrames_)
        {
            this->boxHist_[i][0].is_dynamic = true;
            dynamicBBoxesTemp.push_back(this->boxHist_[i][0]);
            continue;
        }
        // ===================================================================================

        std::vector<Eigen::Vector3d> currPc = this->pcHist_[i][0];
        std::vector<Eigen::Vector3d> prevPc = this->pcHist_[i][curFrameGap];
        Eigen::Vector3d Vcur(0., 0., 0.); // single point velocity
        Eigen::Vector3d Vbox(0., 0., 0.); // bounding box velocity
        Eigen::Vector3d Vkf(0., 0., 0.);  // velocity estimated from kalman filter
        int numPoints = currPc.size(); // it changes within loop
        int votes = 0;

        Vbox(0) = (this->boxHist_[i][0].x - this->boxHist_[i][curFrameGap].x) / (this->dt_ * curFrameGap);
        Vbox(1) = (this->boxHist_[i][0].y - this->boxHist_[i][curFrameGap].y) / (this->dt_ * curFrameGap);
        Vbox(2) = (this->boxHist_[i][0].z - this->boxHist_[i][curFrameGap].z) / (this->dt_ * curFrameGap);
        Vkf(0) = this->boxHist_[i][0].Vx;
        Vkf(1) = this->boxHist_[i][0].Vy;

        // find nearest neighbor
        int numSkip = 0;
        for (size_t j = 0; j < currPc.size(); ++j)
        {
            // don't perform classification for points unseen in previous frame
            if (!this->isInFov(this->positionHist_[curFrameGap], this->orientationHist_[curFrameGap], currPc[j]))
            {
                ++numSkip;
                --numPoints;
                continue;
            }

            double minDist = 2;
            Eigen::Vector3d nearestVect;
            for (size_t k = 0; k < prevPc.size(); k++) // find the nearest point in the previous pointcloud
            {
                double dist = (currPc[j] - prevPc[k]).norm();
                if (abs(dist) < minDist)
                {
                    minDist = dist;
                    nearestVect = currPc[j] - prevPc[k];
                }
            }
            Vcur = nearestVect / (this->dt_ * curFrameGap);
            Vcur(2) = 0;
            double velSim = Vcur.dot(Vbox) / (Vcur.norm() * Vbox.norm());

            if (velSim < 0)
            {
                ++numSkip;
                --numPoints;
            }
            else
            {
                if (Vcur.norm() > this->dynaVelThresh_)
                {
                    ++votes;
                }
            }
        }


        // update dynamic boxes
        double voteRatio = (numPoints > 0) ? double(votes) / double(numPoints) : 0;
        double velNorm = Vkf.norm();

        // voting and velocity threshold
        // 1. point cloud voting ratio.
        // 2. velocity (from kalman filter)
        // 3. enough valid point correspondence
        if (voteRatio >= this->dynaVoteThresh_ && velNorm >= this->dynaVelThresh_ && double(numSkip) / double(numPoints) < this->maxSkipRatio_)
        {
            this->boxHist_[i][0].is_dynamic_candidate = true;
            // dynamic-consistency check
            int dynaConsistCount = 0;
            if (int(this->boxHist_[i].size()) >= this->dynamicConsistThresh_)
            {
                for (int j = 0; j < this->dynamicConsistThresh_; ++j)
                {
                    if (this->boxHist_[i][j].is_dynamic_candidate)
                    {
                        ++dynaConsistCount;
                    }
                }
            }
            if (dynaConsistCount == this->dynamicConsistThresh_)
            {
                // set as dynamic and push into history
                this->boxHist_[i][0].is_dynamic = true;
                dynamicBBoxesTemp.push_back(this->boxHist_[i][0]);
            }
        }
    }

    // filter the dynamic obstacles based on the target sizes
    if (this->constrainSize_)
    {
        std::vector<onboardDetector::box3D  > dynamicBBoxesBeforeConstrain = dynamicBBoxesTemp;
        dynamicBBoxesTemp.clear();

        for (onboardDetector::box3D   ob : dynamicBBoxesBeforeConstrain)
        {
            bool findMatch = false;
            for (Eigen::Vector3d targetSize : this->targetObjectSize_)
            {
                double xdiff = std::abs(ob.x_width - targetSize(0));
                double ydiff = std::abs(ob.y_width - targetSize(1));
                double zdiff = std::abs(ob.z_width - targetSize(2));
                if (xdiff < 0.5 and ydiff < 0.5 and zdiff < 0.5)
                {
                    findMatch = true;
                }
            }

            if (findMatch)
            {
                dynamicBBoxesTemp.push_back(ob);
            }
        }
    }

    this->dynamicBBoxes_ = dynamicBBoxesTemp;
}


void DynamicDetector::visCB(){
    this->publishUVImages();
    this->publish3dBox(this->uvBBoxes_, this->uvBBoxesPub_, 0, 1, 0);
    std::vector<Eigen::Vector3d> dynamicPoints;
    this->getDynamicPc(dynamicPoints);
    this->publishPoints(dynamicPoints, this->dynamicPointsPub_);
    this->publishPoints(this->filteredPoints_, this->filteredPointsPub_);
    this->publish3dBox(this->dbBBoxes_, this->dbBBoxesPub_, 1, 0, 0);
    this->publishYoloImages();
    this->publish3dBox(this->yoloBBoxes_, this->yoloBBoxesPub_, 1, 0, 1);
    this->publish3dBox(this->filteredBBoxes_, this->filteredBBoxesPub_, 0, 1, 1);
    this->publish3dBox(this->trackedBBoxes_, this->trackedBBoxesPub_, 1, 1, 0);
    this->publish3dBox(this->dynamicBBoxes_, this->dynamicBBoxesPub_, 0, 0, 1);
    this->publishHistoryTraj();
    this->publishVelVis();
}

inline void DynamicDetector::getCameraPose(const geometry_msgs::msg::PoseStamped::ConstSharedPtr& pose, Eigen::Matrix4d& camPoseMatrix, Eigen::Matrix4d& camPoseColorMatrix){
    Eigen::Quaterniond quat;
    quat = Eigen::Quaterniond(pose->pose.orientation.w, pose->pose.orientation.x, pose->pose.orientation.y, pose->pose.orientation.z);
    Eigen::Matrix3d rot = quat.toRotationMatrix();

    // convert body pose to camera pose
    Eigen::Matrix4d map2body; map2body.setZero();
    map2body.block<3, 3>(0, 0) = rot;
    map2body(0, 3) = pose->pose.position.x; 
    map2body(1, 3) = pose->pose.position.y;
    map2body(2, 3) = pose->pose.position.z;
    map2body(3, 3) = 1.0;

    camPoseMatrix = map2body * this->body2Cam_;
    camPoseColorMatrix = map2body * this->body2CamColor_;
}

inline void DynamicDetector::getCameraPose(const nav_msgs::msg::Odometry::ConstSharedPtr& odom, Eigen::Matrix4d& camPoseMatrix, Eigen::Matrix4d& camPoseColorMatrix){
    Eigen::Quaterniond quat;
    quat = Eigen::Quaterniond(odom->pose.pose.orientation.w, odom->pose.pose.orientation.x, odom->pose.pose.orientation.y, odom->pose.pose.orientation.z);
    Eigen::Matrix3d rot = quat.toRotationMatrix();

    // convert body pose to camera pose
    Eigen::Matrix4d map2body; map2body.setZero();
    map2body.block<3, 3>(0, 0) = rot;
    map2body(0, 3) = odom->pose.pose.position.x; 
    map2body(1, 3) = odom->pose.pose.position.y;
    map2body(2, 3) = odom->pose.pose.position.z;
    map2body(3, 3) = 1.0;

    camPoseMatrix = map2body * this->body2Cam_;
    camPoseColorMatrix = map2body * this->body2CamColor_;
}


inline bool DynamicDetector::isInFilterRange(const Eigen::Vector3d& pos){
    if ((pos(0) >= this->position_(0) - this->localSensorRange_(0)) and (pos(0) <= this->position_(0) + this->localSensorRange_(0)) and 
        (pos(1) >= this->position_(1) - this->localSensorRange_(1)) and (pos(1) <= this->position_(1) + this->localSensorRange_(1)) and 
        (pos(2) >= this->position_(2) - this->localSensorRange_(2)) and (pos(2) <= this->position_(2) + this->localSensorRange_(2))){
        return true;
    }
    else{
        return false;
    }        
}

inline void DynamicDetector::posToIndex(const Eigen::Vector3d& pos, Eigen::Vector3i& idx, double res){
    idx(0) = floor( (pos(0) - this->position_(0) + localSensorRange_(0)) / res);
    idx(1) = floor( (pos(1) - this->position_(1) + localSensorRange_(1)) / res);
    idx(2) = floor( (pos(2) - this->position_(2) + localSensorRange_(2)) / res);
}

inline int DynamicDetector::indexToAddress(const Eigen::Vector3i& idx, double res){
    return idx(0) * ceil(2*this->localSensorRange_(1)/res) * ceil(2*this->localSensorRange_(2)/res) + idx(1) * ceil(2*this->localSensorRange_(2)/res) + idx(2);
    // return idx(0) * ceil(this->localSensorRange_(0)/res) + idx(1) * ceil(this->localSensorRange_(1)/res) + idx(2);
}

inline int DynamicDetector::posToAddress(const Eigen::Vector3d& pos, double res){
    Eigen::Vector3i idx;
    this->posToIndex(pos, idx, res);
    // ROS_INFO("passed posToIndex, idx: %i", idx);
    // cout << "PASSED posToIndex, idx: " << idx <<endl;
    return this->indexToAddress(idx, res);
}

inline void DynamicDetector::indexToPos(const Eigen::Vector3i& idx, Eigen::Vector3d& pos, double res){
    pos(0) = (idx(0) + 0.5) * res - localSensorRange_(0) + this->position_(0);
    pos(1) = (idx(1) + 0.5) * res - localSensorRange_(1) + this->position_(1);
    pos(2) = (idx(2) + 0.5) * res - localSensorRange_(2) + this->position_(2);
}

inline onboardDetector::Point DynamicDetector::eigenToDBPoint(const Eigen::Vector3d& p){
    onboardDetector::Point pDB;
    pDB.x = p(0);
    pDB.y = p(1);
    pDB.z = p(2);
    pDB.clusterID = -1;
    return pDB;
}

inline Eigen::Vector3d DynamicDetector::dbPointToEigen(const onboardDetector::Point& pDB){
    Eigen::Vector3d p;
    p(0) = pDB.x;
    p(1) = pDB.y;
    p(2) = pDB.z;
    return p;
}

inline void DynamicDetector::eigenToDBPointVec(const std::vector<Eigen::Vector3d>& points, std::vector<onboardDetector::Point>& pointsDB, int size){
    for (int i=0; i<size; ++i){
        Eigen::Vector3d p = points[i];
        onboardDetector::Point pDB = this->eigenToDBPoint(p);
        pointsDB.push_back(pDB);
    }
}


void DynamicDetector::uvDetect(){
    // initialization
    if (this->uvDetector_ == nullptr){
        this->uvDetector_ = std::make_unique<UVdetector>();
        this->uvDetector_->fx = this->fx_;
        this->uvDetector_->fy = this->fy_;
        this->uvDetector_->px = this->cx_;
        this->uvDetector_->py = this->cy_;
        this->uvDetector_->depthScale_ = this->depthScale_; 
        this->uvDetector_->max_dist = this->raycastMaxLength_ * 1000;
    }

    // detect from depth map
    if (!this->depthImage_.empty()){
        this->uvDetector_->depth = this->depthImage_;
        this->uvDetector_->detect();
        this->uvDetector_->extract_3Dbox();

        this->uvDetector_->display_U_map();
        this->uvDetector_->display_bird_view();
        this->uvDetector_->display_depth();

        // transform to the world frame (recalculate the bounding boxes)
        std::vector<onboardDetector::box3D> uvBBoxes;
        this->transformUVBBoxes(uvBBoxes);
        this->uvBBoxes_ = uvBBoxes;
    }
}

void DynamicDetector::dbscanDetect(){
    // 1. get pointcloud
    this->projectDepthImage();

    // 2. update pose history
    this->updatePoseHist();

    // 3. filter points
    this->filterPoints(this->projPoints_, this->filteredPoints_);

    // 4. cluster points and get bounding boxes
    this->clusterPointsAndBBoxes(this->filteredPoints_, this->dbBBoxes_, this->pcClusters_, this->pcClusterCenters_, this->pcClusterStds_);
}

void DynamicDetector::yoloDetectionTo3D(){
    std::vector<onboardDetector::box3D> yoloBBoxesTemp;
    for (size_t i = 0; i < this->yoloDetectionResults_.detections.size(); ++i){
        onboardDetector::box3D bbox3D;
        cv::Rect bboxVis;
        this->getYolo3DBBox(this->yoloDetectionResults_.detections[i], bbox3D, bboxVis);
        cv::rectangle(this->detectedAlignedDepthImg_, bboxVis, cv::Scalar(0, 255, 0), 5, 8, 0);
        yoloBBoxesTemp.push_back(bbox3D);
    }
    this->yoloBBoxes_ = yoloBBoxesTemp;    
}

void DynamicDetector::boxAssociation(std::vector<int>& bestMatch){
    int numObjs = this->filteredBBoxes_.size();
    
    if (this->boxHist_.size() == 0){ // initialize new bounding box history if no history exists
        this->boxHist_.resize(numObjs);
        this->pcHist_.resize(numObjs);
        bestMatch.resize(this->filteredBBoxes_.size(), -1); // first detection no match
        for (int i=0 ; i<numObjs ; ++i){
            // initialize history for bbox, pc and KF
            this->boxHist_[i].push_back(this->filteredBBoxes_[i]);
            this->pcHist_[i].push_back(this->filteredPcClusters_[i]);
            MatrixXd states, A, B, H, P, Q, R;       
            this->kalmanFilterMatrixAcc(this->filteredBBoxes_[i], states, A, B, H, P, Q, R);
            onboardDetector::kalman_filter newFilter;
            newFilter.setup(states, A, B, H, P, Q, R);
            this->filters_.push_back(newFilter);
        }
    }
    else{
        // start association only if a new detection is available
        if (this->newDetectFlag_){
            this->boxAssociationHelper(bestMatch);
        }
    }

    this->newDetectFlag_ = false; // the most recent detection has been associated
}

void DynamicDetector::boxAssociationHelper(std::vector<int>& bestMatch){
    int numObjs = this->filteredBBoxes_.size();
    std::vector<onboardDetector::box3D> propedBoxes;
    std::vector<Eigen::VectorXd> propedBoxesFeat;
    std::vector<Eigen::VectorXd> currBoxesFeat;
    bestMatch.resize(numObjs);
    std::deque<std::deque<onboardDetector::box3D>> boxHistTemp; 

    // linear propagation: prediction of previous box in current frame
    this->linearProp(propedBoxes);

    // generate feature
    this->genFeat(propedBoxes, numObjs, propedBoxesFeat, currBoxesFeat);

    // calculate association: find best match
    this->findBestMatch(propedBoxesFeat, currBoxesFeat, propedBoxes, bestMatch);

}

void DynamicDetector::genFeat(const std::vector<onboardDetector::box3D>& propedBoxes, int numObjs, std::vector<Eigen::VectorXd>& propedBoxesFeat, std::vector<Eigen::VectorXd>& currBoxesFeat){
    propedBoxesFeat.resize(propedBoxes.size());
    currBoxesFeat.resize(numObjs);
    this->genFeatHelper(propedBoxesFeat, propedBoxes);
    this->genFeatHelper(currBoxesFeat, this->filteredBBoxes_);
}

void DynamicDetector::genFeatHelper(std::vector<Eigen::VectorXd>& features, const std::vector<onboardDetector::box3D>& boxes){ 
    Eigen::VectorXd featureWeights(10); // 3pos + 3size + 1 pc length + 3 pc std
    featureWeights << 2, 2, 2, 1, 1, 1, 0.5, 0.5, 0.5, 0.5;
    for (size_t i=0 ; i<boxes.size() ; i++){
        Eigen::VectorXd feature(10);
        features[i] = feature;
        features[i](0) = (boxes[i].x - this->position_(0)) * featureWeights(0) ;
        features[i](1) = (boxes[i].y - this->position_(1)) * featureWeights(1);
        features[i](2) = (boxes[i].z - this->position_(2)) * featureWeights(2);
        features[i](3) = boxes[i].x_width * featureWeights(3);
        features[i](4) = boxes[i].y_width * featureWeights(4);
        features[i](5) = boxes[i].z_width * featureWeights(5);
        features[i](6) = this->filteredPcClusters_[i].size() * featureWeights(6);
        features[i](7) = this->filteredPcClusterStds_[i](0) * featureWeights(7);
        features[i](8) = this->filteredPcClusterStds_[i](1) * featureWeights(8);
        features[i](9) = this->filteredPcClusterStds_[i](2) * featureWeights(9);
    }
}

void DynamicDetector::linearProp(std::vector<onboardDetector::box3D>& propedBoxes){
    onboardDetector::box3D propedBox;
    for (size_t i=0 ; i<this->boxHist_.size() ; i++){
        propedBox = this->boxHist_[i][0];
        propedBox.x += propedBox.Vx*this->dt_;
        propedBox.y += propedBox.Vy*this->dt_;
        propedBoxes.push_back(propedBox);
    }
}

void DynamicDetector::findBestMatch(const std::vector<Eigen::VectorXd>& propedBoxesFeat, const std::vector<Eigen::VectorXd>& currBoxesFeat, const std::vector<onboardDetector::box3D>& propedBoxes, std::vector<int>& bestMatch){
    
    int numObjs = this->filteredBBoxes_.size();
    std::vector<double> bestSims; // best similarity
    bestSims.resize(numObjs);

    for (int i=0 ; i<numObjs ; i++){
        double bestSim = -1.;
        int bestMatchInd = -1;
        for (size_t j=0 ; j<propedBoxes.size() ; j++){
            double sim = propedBoxesFeat[j].dot(currBoxesFeat[i])/(propedBoxesFeat[j].norm()*currBoxesFeat[i].norm());
            if (sim >= bestSim){
                bestSim = sim;
                bestSims[i] = sim;
                bestMatchInd = j;
            }
        }

        double iou = this->calBoxIOU(this->filteredBBoxes_[i], propedBoxes[bestMatchInd]);
        if(!(bestSims[i]>this->simThresh_ && iou)){
            bestSims[i] = 0;
            bestMatch[i] = -1;
        }
        else {
            bestMatch[i] = bestMatchInd;
        }
    }
}


void DynamicDetector::kalmanFilterAndUpdateHist(const std::vector<int>& bestMatch){
    std::vector<std::deque<onboardDetector::box3D>> boxHistTemp; 
    std::vector<std::deque<std::vector<Eigen::Vector3d>>> pcHistTemp;
    std::vector<onboardDetector::kalman_filter> filtersTemp;
    std::deque<onboardDetector::box3D> newSingleBoxHist;
    std::deque<std::vector<Eigen::Vector3d>> newSinglePcHist; 
    onboardDetector::kalman_filter newFilter;
    std::vector<onboardDetector::box3D> trackedBBoxesTemp;

    newSingleBoxHist.resize(0);
    newSinglePcHist.resize(0);
    int numObjs = this->filteredBBoxes_.size();

    for (int i=0 ; i<numObjs ; i++){
        onboardDetector::box3D newEstimatedBBox; // from kalman filter

        // inheret history. push history one by one
        if (bestMatch[i]>=0){
            boxHistTemp.push_back(this->boxHist_[bestMatch[i]]);
            pcHistTemp.push_back(this->pcHist_[bestMatch[i]]);
            filtersTemp.push_back(this->filters_[bestMatch[i]]);

            // kalman filter to get new state estimation
            onboardDetector::box3D currDetectedBBox = this->filteredBBoxes_[i];

            Eigen::MatrixXd Z;
            this->getKalmanObservationAcc(currDetectedBBox, bestMatch[i], Z);
            filtersTemp.back().estimate(Z, MatrixXd::Zero(6,1));
            
            
            newEstimatedBBox.x = filtersTemp.back().output(0);
            newEstimatedBBox.y = filtersTemp.back().output(1);
            newEstimatedBBox.z = currDetectedBBox.z;
            newEstimatedBBox.Vx = filtersTemp.back().output(2);
            newEstimatedBBox.Vy = filtersTemp.back().output(3);
            newEstimatedBBox.Ax = filtersTemp.back().output(4);
            newEstimatedBBox.Ay = filtersTemp.back().output(5);   
                      

            newEstimatedBBox.x_width = currDetectedBBox.x_width;
            newEstimatedBBox.y_width = currDetectedBBox.y_width;
            newEstimatedBBox.z_width = currDetectedBBox.z_width;
            newEstimatedBBox.is_dynamic = currDetectedBBox.is_dynamic;
            newEstimatedBBox.is_human = currDetectedBBox.is_human;
        }
        else{
            boxHistTemp.push_back(newSingleBoxHist);
            pcHistTemp.push_back(newSinglePcHist);

            // create new kalman filter for this object
            onboardDetector::box3D currDetectedBBox = this->filteredBBoxes_[i];
            MatrixXd states, A, B, H, P, Q, R;    
            this->kalmanFilterMatrixAcc(currDetectedBBox, states, A, B, H, P, Q, R);
            
            newFilter.setup(states, A, B, H, P, Q, R);
            filtersTemp.push_back(newFilter);
            newEstimatedBBox = currDetectedBBox;
            
        }

        // pop old data if len of hist > size limit
        if (int(boxHistTemp[i].size()) == this->histSize_){
            boxHistTemp[i].pop_back();
            pcHistTemp[i].pop_back();
        }

        // push new data into history
        boxHistTemp[i].push_front(newEstimatedBBox); 
        pcHistTemp[i].push_front(this->filteredPcClusters_[i]);

        // update new tracked bounding boxes
        trackedBBoxesTemp.push_back(newEstimatedBBox);
    }

    if (boxHistTemp.size()){
        for (size_t i=0; i<trackedBBoxesTemp.size(); ++i){ 
            if (int(boxHistTemp[i].size()) >= this->fixSizeHistThresh_){
                if ((abs(trackedBBoxesTemp[i].x_width-boxHistTemp[i][1].x_width)/boxHistTemp[i][1].x_width) <= this->fixSizeDimThresh_ &&
                    (abs(trackedBBoxesTemp[i].y_width-boxHistTemp[i][1].y_width)/boxHistTemp[i][1].y_width) <= this->fixSizeDimThresh_&&
                    (abs(trackedBBoxesTemp[i].z_width-boxHistTemp[i][1].z_width)/boxHistTemp[i][1].z_width) <= this->fixSizeDimThresh_){
                    trackedBBoxesTemp[i].x_width = boxHistTemp[i][1].x_width;
                    trackedBBoxesTemp[i].y_width = boxHistTemp[i][1].y_width;
                    trackedBBoxesTemp[i].z_width = boxHistTemp[i][1].z_width;
                    boxHistTemp[i][0].x_width = trackedBBoxesTemp[i].x_width;
                    boxHistTemp[i][0].y_width = trackedBBoxesTemp[i].y_width;
                    boxHistTemp[i][0].z_width = trackedBBoxesTemp[i].z_width;
                }

            }
        }
    }
    
    // update history member variable
    this->boxHist_ = boxHistTemp;
    this->pcHist_ = pcHistTemp;
    this->filters_ = filtersTemp;

    // update tracked bounding boxes
    this->trackedBBoxes_=  trackedBBoxesTemp;

}

void DynamicDetector::kalmanFilterMatrixVel(const onboardDetector::box3D& currDetectedBBox, MatrixXd& states, MatrixXd& A, MatrixXd& B, MatrixXd& H, MatrixXd& P, MatrixXd& Q, MatrixXd& R){
    states.resize(4,1);
    states(0) = currDetectedBBox.x;
    states(1) = currDetectedBBox.y;
    // init vel and acc to zeros
    states(2) = 0.;
    states(3) = 0.;

    MatrixXd ATemp;
    ATemp.resize(4, 4);
    ATemp <<  0, 0, 1, 0,
              0, 0, 0, 1,
              0, 0, 0, 0,
              0 ,0, 0, 0;
    A = MatrixXd::Identity(4,4) + this->dt_*ATemp;
    B = MatrixXd::Zero(4, 4);
    H = MatrixXd::Identity(4, 4);
    P = MatrixXd::Identity(4, 4) * this->eP_;
    Q = MatrixXd::Identity(4, 4);
    Q(0,0) *= this->eQPos_; Q(1,1) *= this->eQPos_; Q(2,2) *= this->eQVel_; Q(3,3) *= this->eQVel_; 
    R = MatrixXd::Identity(4, 4);
    R(0,0) *= this->eRPos_; R(1,1) *= this->eRPos_; R(2,2) *= this->eRVel_; R(3,3) *= this->eRVel_;

}

void DynamicDetector::kalmanFilterMatrixAcc(const onboardDetector::box3D& currDetectedBBox, MatrixXd& states, MatrixXd& A, MatrixXd& B, MatrixXd& H, MatrixXd& P, MatrixXd& Q, MatrixXd& R){
    states.resize(6,1);
    states(0) = currDetectedBBox.x;
    states(1) = currDetectedBBox.y;
    // init vel and acc to zeros
    states(2) = 0.;
    states(3) = 0.;
    states(4) = 0.;
    states(5) = 0.;

    MatrixXd ATemp;
    ATemp.resize(6, 6);

    ATemp <<  1, 0, this->dt_, 0, 0.5*pow(this->dt_, 2), 0,
              0, 1, 0, this->dt_, 0, 0.5*pow(this->dt_, 2),
              0, 0, 1, 0, this->dt_, 0,
              0 ,0, 0, 1, 0, this->dt_,
              0, 0, 0, 0, 1, 0,
              0, 0, 0, 0, 0, 1;
    A = ATemp;
    B = MatrixXd::Zero(6, 6);
    H = MatrixXd::Identity(6, 6);
    P = MatrixXd::Identity(6, 6) * this->eP_;
    Q = MatrixXd::Identity(6, 6);
    Q(0,0) *= this->eQPos_; Q(1,1) *= this->eQPos_; Q(2,2) *= this->eQVel_; Q(3,3) *= this->eQVel_; Q(4,4) *= this->eQAcc_; Q(5,5) *= this->eQAcc_;
    R = MatrixXd::Identity(6, 6);
    R(0,0) *= this->eRPos_; R(1,1) *= this->eRPos_; R(2,2) *= this->eRVel_; R(3,3) *= this->eRVel_; R(4,4) *= this->eRAcc_; R(5,5) *= this->eRAcc_;
}

void DynamicDetector::getKalmanObservationVel(const onboardDetector::box3D& currDetectedBBox, int bestMatchIdx, MatrixXd& Z){
    Z.resize(4,1);
    Z(0) = currDetectedBBox.x; 
    Z(1) = currDetectedBBox.y;

    // use previous k frame for velocity estimation
    int k = this->kfAvgFrames_;
    int historySize = this->boxHist_[bestMatchIdx].size();
    if (historySize < k){
        k = historySize;
    }
    onboardDetector::box3D prevMatchBBox = this->boxHist_[bestMatchIdx][k-1];

    Z(2) = (currDetectedBBox.x-prevMatchBBox.x)/(this->dt_*k);
    Z(3) = (currDetectedBBox.y-prevMatchBBox.y)/(this->dt_*k);
}

void DynamicDetector::getKalmanObservationAcc(const onboardDetector::box3D& currDetectedBBox, int bestMatchIdx, MatrixXd& Z){
    Z.resize(6, 1);
    Z(0) = currDetectedBBox.x;
    Z(1) = currDetectedBBox.y;

    // use previous k frame for velocity estimation
    int k = this->kfAvgFrames_;
    int historySize = this->boxHist_[bestMatchIdx].size();
    if (historySize < k){
        k = historySize;
    }
    onboardDetector::box3D prevMatchBBox = this->boxHist_[bestMatchIdx][k-1];

    Z(2) = (currDetectedBBox.x - prevMatchBBox.x)/(this->dt_*k);
    Z(3) = (currDetectedBBox.y - prevMatchBBox.y)/(this->dt_*k);
    Z(4) = (Z(2) - prevMatchBBox.Vx)/(this->dt_*k);
    Z(5) = (Z(3) - prevMatchBBox.Vy)/(this->dt_*k);
}

void DynamicDetector::transformUVBBoxes(std::vector<onboardDetector::box3D>& bboxes){
    bboxes.clear();
    for(size_t i = 0; i < this->uvDetector_->box3Ds.size(); ++i){
        onboardDetector::box3D bbox;
        double x = this->uvDetector_->box3Ds[i].x; 
        double y = this->uvDetector_->box3Ds[i].y;
        double z = this->uvDetector_->box3Ds[i].z;
        double xWidth = this->uvDetector_->box3Ds[i].x_width;
        double yWidth = this->uvDetector_->box3Ds[i].y_width;
        double zWidth = this->uvDetector_->box3Ds[i].z_width;

        Eigen::Vector3d center (x, y, z);
        Eigen::Vector3d size (xWidth, yWidth, zWidth);
        Eigen::Vector3d newCenter, newSize;

        this->transformBBox(center, size, this->position_, this->orientation_, newCenter, newSize);

        // assign values to bounding boxes in the map frame
        bbox.x = newCenter(0);
        bbox.y = newCenter(1);
        bbox.z = newCenter(2);
        bbox.x_width = newSize(0);
        bbox.y_width = newSize(1);
        bbox.z_width = newSize(2);
        bboxes.push_back(bbox);            
    }        
}

void DynamicDetector::filterBBoxes(){
    std::vector<onboardDetector::box3D> filteredBBoxesTemp;
    std::vector<std::vector<Eigen::Vector3d>> filteredPcClustersTemp;
    std::vector<Eigen::Vector3d> filteredPcClusterCentersTemp;
    std::vector<Eigen::Vector3d> filteredPcClusterStdsTemp; 
    // find best IOU match for both uv and dbscan. If they are best for each other, then add to filtered bbox and fuse.
    for (size_t i=0 ; i<this->uvBBoxes_.size(); ++i){
        onboardDetector::box3D uvBBox = this->uvBBoxes_[i];
        double bestIOUForUVBBox, bestIOUForDBBBox;
        int bestMatchForUVBBox = this->getBestOverlapBBox(uvBBox, this->dbBBoxes_, bestIOUForUVBBox);
        if (bestMatchForUVBBox == -1) continue; // no match at all
        onboardDetector::box3D matchedDBBBox = this->dbBBoxes_[bestMatchForUVBBox]; 
        std::vector<Eigen::Vector3d> matchedPcCluster = this->pcClusters_[bestMatchForUVBBox];
        Eigen::Vector3d matchedPcClusterCenter = this->pcClusterCenters_[bestMatchForUVBBox];
        Eigen::Vector3d matchedPcClusterStd = this->pcClusterStds_[bestMatchForUVBBox];
        int bestMatchForDBBBox = this->getBestOverlapBBox(matchedDBBBox, this->uvBBoxes_, bestIOUForDBBBox);

        // if best match is each other and both the IOU is greater than the threshold
        if (bestMatchForDBBBox == int(i) and bestIOUForUVBBox > this->boxIOUThresh_ and bestIOUForDBBBox > this->boxIOUThresh_){
            onboardDetector::box3D bbox;
            
            // take concervative strategy
            double xmax = std::max(uvBBox.x+uvBBox.x_width/2, matchedDBBBox.x+matchedDBBBox.x_width/2);
            double xmin = std::min(uvBBox.x-uvBBox.x_width/2, matchedDBBBox.x-matchedDBBBox.x_width/2);
            double ymax = std::max(uvBBox.y+uvBBox.y_width/2, matchedDBBBox.y+matchedDBBBox.y_width/2);
            double ymin = std::min(uvBBox.y-uvBBox.y_width/2, matchedDBBBox.y-matchedDBBBox.y_width/2);
            double zmax = std::max(uvBBox.z+uvBBox.z_width/2, matchedDBBBox.z+matchedDBBBox.z_width/2);
            double zmin = std::min(uvBBox.z-uvBBox.z_width/2, matchedDBBBox.z-matchedDBBBox.z_width/2);
            bbox.x = (xmin+xmax)/2;
            bbox.y = (ymin+ymax)/2;
            bbox.z = (zmin+zmax)/2;
            bbox.x_width = xmax-xmin;
            bbox.y_width = ymax-ymin;
            bbox.z_width = zmax-zmin;
            bbox.Vx = 0;
            bbox.Vy = 0;

            filteredBBoxesTemp.push_back(bbox);
            filteredPcClustersTemp.push_back(matchedPcCluster);      
            filteredPcClusterCentersTemp.push_back(matchedPcClusterCenter);
            filteredPcClusterStdsTemp.push_back(matchedPcClusterStd);
        }
    }

    // yolo bounding box filter
    if (this->yoloBBoxes_.size() != 0){ // if no detected or not using yolo, this will not triggered
        std::vector<onboardDetector::box3D> filteredBBoxesTempCopy = filteredBBoxesTemp;
        std::vector<std::vector<Eigen::Vector3d>> filteredPcClustersTempCopy = filteredPcClustersTemp;
        std::vector<Eigen::Vector3d> filteredPcClusterCentersTempCopy = filteredPcClusterCentersTemp;
        std::vector<Eigen::Vector3d> filteredPcClusterStdsTempCopy = filteredPcClusterStdsTemp;
        std::vector<Eigen::Vector3d> emptyPoints {};
        Eigen::Vector3d emptyPcFeat {0,0,0};
        for (size_t i=0; i<this->yoloBBoxes_.size(); ++i){
            onboardDetector::box3D yoloBBox = this->yoloBBoxes_[i]; yoloBBox.is_dynamic = true; yoloBBox.is_human = true; // dynamic obstacle detected by yolo
            Eigen::Vector3d bboxPos (this->yoloBBoxes_[i].x, this->yoloBBoxes_[i].y, this->yoloBBoxes_[i].z);
            double distanceToCamera = (bboxPos - this->position_).norm();
            if (distanceToCamera >= this->raycastMaxLength_){
                continue; // do not use unreliable YOLO resutls which are distance too far from camera
            }
            double bestIOUForYoloBBox, bestIOUForFilteredBBox;
            int bestMatchForYoloBBox = this->getBestOverlapBBox(yoloBBox, filteredBBoxesTemp, bestIOUForYoloBBox);
            if (bestMatchForYoloBBox == -1){ // no match for yolo bounding boxes with any filtered bbox. 2 reasons: a) distance too far, filtered boxes no detection, b) distance not far but cannot match. Probably Yolo error
                if (distanceToCamera >= this->yoloOverwriteDistance_){ // a) distance too far, filtered boxes no detection. directly add results
                    filteredBBoxesTempCopy.push_back(yoloBBox); // add yolo bbox because filtered bbox is not able to get detection results at far distance
                    filteredPcClustersTempCopy.push_back(emptyPoints); // no pc need for yolo 
                    filteredPcClusterCentersTempCopy.push_back(emptyPcFeat);
                    filteredPcClusterStdsTempCopy.push_back(emptyPcFeat);
                }
                else{ // b) distance not far but cannot match. Probably Yolo error, ignore results
                    continue;
                }
            }
            else{ // find best match for yolo bbox
                onboardDetector::box3D matchedFilteredBBox = filteredBBoxesTemp[bestMatchForYoloBBox];
                int bestMatchForFilteredBBox = this->getBestOverlapBBox(matchedFilteredBBox, this->yoloBBoxes_, bestIOUForFilteredBBox);
                // if best match is each other and both the IOU is greater than the threshold
                if (bestMatchForFilteredBBox == int(i) and bestIOUForYoloBBox > this->boxIOUThresh_ and bestIOUForFilteredBBox > this->boxIOUThresh_){
                    onboardDetector::box3D bbox; bbox.is_dynamic = true; bbox.is_human = true;
                    
                    // take concervative strategy
                    double xmax = std::max(yoloBBox.x+yoloBBox.x_width/2, matchedFilteredBBox.x+matchedFilteredBBox.x_width/2);
                    double xmin = std::min(yoloBBox.x-yoloBBox.x_width/2, matchedFilteredBBox.x-matchedFilteredBBox.x_width/2);
                    double ymax = std::max(yoloBBox.y+yoloBBox.y_width/2, matchedFilteredBBox.y+matchedFilteredBBox.y_width/2);
                    double ymin = std::min(yoloBBox.y-yoloBBox.y_width/2, matchedFilteredBBox.y-matchedFilteredBBox.y_width/2);
                    double zmax = std::max(yoloBBox.z+yoloBBox.z_width/2, matchedFilteredBBox.z+matchedFilteredBBox.z_width/2);
                    double zmin = std::min(yoloBBox.z-yoloBBox.z_width/2, matchedFilteredBBox.z-matchedFilteredBBox.z_width/2);
                    bbox.x = (xmin+xmax)/2;
                    bbox.y = (ymin+ymax)/2;
                    bbox.z = (zmin+zmax)/2;
                    bbox.x_width = xmax-xmin;
                    bbox.y_width = ymax-ymin;
                    bbox.z_width = zmax-zmin;
                    bbox.Vx = 0;
                    bbox.Vy = 0;
                    
                    filteredBBoxesTempCopy[bestMatchForYoloBBox] = bbox; // replace the filtered bbox with the new fused bounding box
                    filteredPcClustersTempCopy[bestMatchForYoloBBox] = emptyPoints;      // since it is yolo based, we dont need pointcloud for classification                     
                    filteredPcClusterCentersTempCopy[bestMatchForYoloBBox] = emptyPcFeat;
                    filteredPcClusterStdsTempCopy[bestMatchForYoloBBox] = emptyPcFeat;
                }
            }
        }
        filteredBBoxesTemp = filteredBBoxesTempCopy;
        filteredPcClustersTemp = filteredPcClustersTempCopy;
        filteredPcClusterCentersTemp = filteredPcClusterCentersTempCopy;
        filteredPcClusterStdsTemp = filteredPcClusterStdsTempCopy;
    }

    this->filteredBBoxes_ = filteredBBoxesTemp;
    this->filteredPcClusters_ = filteredPcClustersTemp;
    this->filteredPcClusterCenters_ = filteredPcClusterCentersTemp;
    this->filteredPcClusterStds_ = filteredPcClusterStdsTemp;
}

void DynamicDetector::projectDepthImage(){
    this->projPointsNum_ = 0;

    int cols = this->depthImage_.cols;
    int rows = this->depthImage_.rows;
    uint16_t* rowPtr;

    Eigen::Vector3d currPointCam, currPointMap;
    double depth;
    const double inv_factor = 1.0 / this->depthScale_;
    const double inv_fx = 1.0 / this->fx_;
    const double inv_fy = 1.0 / this->fy_;

    // iterate through each pixel in the depth image
    for (int v=this->depthFilterMargin_; v<rows-this->depthFilterMargin_; v=v+this->skipPixel_){ // row
        rowPtr = this->depthImage_.ptr<uint16_t>(v) + this->depthFilterMargin_;
        for (int u=this->depthFilterMargin_; u<cols-this->depthFilterMargin_; u=u+this->skipPixel_){ // column
            depth = (*rowPtr) * inv_factor;
            
            if (*rowPtr == 0) {
                depth = this->raycastMaxLength_ + 0.1;
            } else if (depth < this->depthMinValue_) {
                continue;
            } else if (depth > this->depthMaxValue_) {
                depth = this->raycastMaxLength_ + 0.1;
            }
            rowPtr =  rowPtr + this->skipPixel_;

            // get 3D point in camera frame
            currPointCam(0) = (u - this->cx_) * depth * inv_fx;
            currPointCam(1) = (v - this->cy_) * depth * inv_fy;
            currPointCam(2) = depth;
            currPointMap = this->orientation_ * currPointCam + this->position_; // transform to map coordinate

            this->projPoints_[this->projPointsNum_] = currPointMap;
            this->pointsDepth_[this->projPointsNum_] = depth;
            this->projPointsNum_ = this->projPointsNum_ + 1;
        }
    } 
}


void DynamicDetector::filterPoints(const std::vector<Eigen::Vector3d>& points, std::vector<Eigen::Vector3d>& filteredPoints){
    // currently there is only one filtered (might include more in the future)
    std::vector<Eigen::Vector3d> voxelFilteredPoints;
    this->voxelFilter(points, voxelFilteredPoints);
    filteredPoints = voxelFilteredPoints;
}


void DynamicDetector::clusterPointsAndBBoxes(const std::vector<Eigen::Vector3d>& points, std::vector<onboardDetector::box3D>& bboxes, std::vector<std::vector<Eigen::Vector3d>>& pcClusters, std::vector<Eigen::Vector3d>& pcClusterCenters, std::vector<Eigen::Vector3d>& pcClusterStds){
    std::vector<onboardDetector::Point> pointsDB;
    this->eigenToDBPointVec(points, pointsDB, points.size());

    this->dbCluster_.reset(new DBSCAN (this->dbMinPointsCluster_, this->dbEpsilon_, pointsDB));

    // DBSCAN clustering
    this->dbCluster_->run();

    // get the cluster data with bounding boxes
    // iterate through all the clustered points and find number of clusters
    int clusterNum = 0;
    for (size_t i=0; i<this->dbCluster_->m_points.size(); ++i){
        onboardDetector::Point pDB = this->dbCluster_->m_points[i];
        if (pDB.clusterID > clusterNum){
            clusterNum = pDB.clusterID;
        }
    }

    pcClusters.clear();
    pcClusters.resize(clusterNum);
    for (size_t i=0; i<this->dbCluster_->m_points.size(); ++i){
        onboardDetector::Point pDB = this->dbCluster_->m_points[i];
        if (pDB.clusterID > 0){
            Eigen::Vector3d p = this->dbPointToEigen(pDB);
            pcClusters[pDB.clusterID-1].push_back(p);
        }            
    }

    for (size_t i=0 ; i<pcClusters.size() ; ++i){
        Eigen::Vector3d pcClusterCenter(0.,0.,0.);
        Eigen::Vector3d pcClusterStd(0.,0.,0.);
        this->calcPcFeat(pcClusters[i], pcClusterCenter, pcClusterStd);
        pcClusterCenters.push_back(pcClusterCenter);
        pcClusterStds.push_back(pcClusterStd);
    }

    // calculate the bounding boxes based on the clusters
    bboxes.clear();
    // bboxes.resize(clusterNum);
    for (size_t i=0; i<pcClusters.size(); ++i){
        onboardDetector::box3D box;

        double xmin = pcClusters[i][0](0);
        double ymin = pcClusters[i][0](1);
        double zmin = pcClusters[i][0](2);
        double xmax = pcClusters[i][0](0);
        double ymax = pcClusters[i][0](1);
        double zmax = pcClusters[i][0](2);
        for (size_t j=0; j<pcClusters[i].size(); ++j){
            xmin = (pcClusters[i][j](0)<xmin)?pcClusters[i][j](0):xmin;
            ymin = (pcClusters[i][j](1)<ymin)?pcClusters[i][j](1):ymin;
            zmin = (pcClusters[i][j](2)<zmin)?pcClusters[i][j](2):zmin;
            xmax = (pcClusters[i][j](0)>xmax)?pcClusters[i][j](0):xmax;
            ymax = (pcClusters[i][j](1)>ymax)?pcClusters[i][j](1):ymax;
            zmax = (pcClusters[i][j](2)>zmax)?pcClusters[i][j](2):zmax;
        }
        box.id = i;

        box.x = (xmax + xmin)/2.0;
        box.y = (ymax + ymin)/2.0;
        box.z = (zmax + zmin)/2.0;
        box.x_width = (xmax - xmin)>0.1?(xmax-xmin):0.1;
        box.y_width = (ymax - ymin)>0.1?(ymax-ymin):0.1;
        box.z_width = (zmax - zmin);
        bboxes.push_back(box);
    }
}

void DynamicDetector::voxelFilter(const std::vector<Eigen::Vector3d>& points, std::vector<Eigen::Vector3d>& filteredPoints){
    const double res = 0.1; // resolution of voxel
    int xVoxels = ceil(2*this->localSensorRange_(0)/res); int yVoxels = ceil(2*this->localSensorRange_(1)/res); int zVoxels = ceil(2*this->localSensorRange_(2)/res);
    int totalVoxels = xVoxels * yVoxels * zVoxels;
    // std::vector<bool> voxelOccupancyVec (totalVoxels, false);
    std::vector<int> voxelOccupancyVec (totalVoxels, 0);

    // Iterate through each points in the cloud
    filteredPoints.clear();
    
    for (int i=0; i<this->projPointsNum_; ++i){
        Eigen::Vector3d p = points[i];

        if (this->isInFilterRange(p) and p(2) >= this->groundHeight_ and this->pointsDepth_[i] <= this->raycastMaxLength_){
            // find the corresponding voxel id in the vector and check whether it is occupied
            int pID = this->posToAddress(p, res);

            // add one point
            voxelOccupancyVec[pID] +=1;

            // add only if thresh points are found
            if (voxelOccupancyVec[pID] == this->voxelOccThresh_){
                filteredPoints.push_back(p);
            }
        }
    }  
}

void DynamicDetector::calcPcFeat(const std::vector<Eigen::Vector3d>& pcCluster, Eigen::Vector3d& pcClusterCenter, Eigen::Vector3d& pcClusterStd){
    int numPoints = pcCluster.size();
    
    // center
    for (int i=0 ; i<numPoints ; i++){
        pcClusterCenter(0) += pcCluster[i](0)/numPoints;
        pcClusterCenter(1) += pcCluster[i](1)/numPoints;
        pcClusterCenter(2) += pcCluster[i](2)/numPoints;
    }

    // std
    for (int i=0 ; i<numPoints ; i++){
        pcClusterStd(0) += std::pow(pcCluster[i](0) - pcClusterCenter(0),2);
        pcClusterStd(1) += std::pow(pcCluster[i](1) - pcClusterCenter(1),2);
        pcClusterStd(2) += std::pow(pcCluster[i](2) - pcClusterCenter(2),2);
    }        

    // take square root
    pcClusterStd(0) = std::sqrt(pcClusterStd(0)/numPoints);
    pcClusterStd(1) = std::sqrt(pcClusterStd(1)/numPoints);
    pcClusterStd(2) = std::sqrt(pcClusterStd(2)/numPoints);
}


double DynamicDetector::calBoxIOU(const onboardDetector::box3D& box1, const onboardDetector::box3D& box2){
    double box1Volume = box1.x_width * box1.y_width * box1.z_width;
    double box2Volume = box2.x_width * box2.y_width * box2.z_width;

    double l1Y = box1.y+box1.y_width/2-(box2.y-box2.y_width/2);
    double l2Y = box2.y+box2.y_width/2-(box1.y-box1.y_width/2);
    double l1X = box1.x+box1.x_width/2-(box2.x-box2.x_width/2);
    double l2X = box2.x+box2.x_width/2-(box1.x-box1.x_width/2);
    double l1Z = box1.z+box1.z_width/2-(box2.z-box2.z_width/2);
    double l2Z = box2.z+box2.z_width/2-(box1.z-box1.z_width/2);
    double overlapX = std::min( l1X , l2X );
    double overlapY = std::min( l1Y , l2Y );
    double overlapZ = std::min( l1Z , l2Z );
   
    if (std::max(l1X, l2X)<=std::max(box1.x_width,box2.x_width)){ 
        overlapX = std::min(box1.x_width, box2.x_width);
    }
    if (std::max(l1Y, l2Y)<=std::max(box1.y_width,box2.y_width)){ 
        overlapY = std::min(box1.y_width, box2.y_width);
    }
    if (std::max(l1Z, l2Z)<=std::max(box1.z_width,box2.z_width)){ 
        overlapZ = std::min(box1.z_width, box2.z_width);
    }


    double overlapVolume = overlapX * overlapY *  overlapZ;
    double IOU = overlapVolume / (box1Volume+box2Volume-overlapVolume);
    
    // D-IOU
    if (overlapX<=0 || overlapY<=0 ||overlapZ<=0){
        IOU = 0;
    }
    return IOU;
}

void DynamicDetector::publishUVImages(){
    sensor_msgs::msg::Image::ConstSharedPtr depthBoxMsg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", this->uvDetector_->depth_show).toImageMsg();
    sensor_msgs::msg::Image::ConstSharedPtr UmapBoxMsg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", this->uvDetector_->U_map_show).toImageMsg();
    sensor_msgs::msg::Image::ConstSharedPtr birdBoxMsg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", this->uvDetector_->bird_view).toImageMsg();  
    this->uvDepthMapPub_.publish(depthBoxMsg);
    this->uDepthMapPub_.publish(UmapBoxMsg); 
    this->uvBirdViewPub_.publish(birdBoxMsg);     
}

void DynamicDetector::publish3dBox(const std::vector<box3D>& boxes, const rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr publisher, double r, double g, double b) {
    // visualization using bounding boxes 
    visualization_msgs::msg::Marker line;
    visualization_msgs::msg::MarkerArray lines;
    line.header.frame_id = "map";
    line.type = visualization_msgs::msg::Marker::LINE_LIST;
    line.action = visualization_msgs::msg::Marker::ADD;
    line.ns = "box3D";  
    line.scale.x = 0.06;
    line.color.r = r;
    line.color.g = g;
    line.color.b = b;
    line.color.a = 1.0;
    line.lifetime = rclcpp::Duration::from_seconds(0.1);
    
    for(size_t i = 0; i < boxes.size(); i++){
        // visualization msgs
        line.text = " Vx " + std::to_string(boxes[i].Vx) + " Vy " + std::to_string(boxes[i].Vy);
        double x = boxes[i].x; 
        double y = boxes[i].y; 
        double z = (boxes[i].z+boxes[i].z_width/2)/2; 

        // double x_width = std::max(boxes[i].x_width,boxes[i].y_width);
        // double y_width = std::max(boxes[i].x_width,boxes[i].y_width);
        double x_width = boxes[i].x_width;
        double y_width = boxes[i].y_width;
        double z_width = 2*z;

        // double z = 
        
        vector<geometry_msgs::msg::Point> verts;
        geometry_msgs::msg::Point p;
        // vertice 0
        p.x = x-x_width / 2.; p.y = y-y_width / 2.; p.z = z-z_width / 2.;
        verts.push_back(p);

        // vertice 1
        p.x = x-x_width / 2.; p.y = y+y_width / 2.; p.z = z-z_width / 2.;
        verts.push_back(p);

        // vertice 2
        p.x = x+x_width / 2.; p.y = y+y_width / 2.; p.z = z-z_width / 2.;
        verts.push_back(p);

        // vertice 3
        p.x = x+x_width / 2.; p.y = y-y_width / 2.; p.z = z-z_width / 2.;
        verts.push_back(p);

        // vertice 4
        p.x = x-x_width / 2.; p.y = y-y_width / 2.; p.z = z+z_width / 2.;
        verts.push_back(p);

        // vertice 5
        p.x = x-x_width / 2.; p.y = y+y_width / 2.; p.z = z+z_width / 2.;
        verts.push_back(p);

        // vertice 6
        p.x = x+x_width / 2.; p.y = y+y_width / 2.; p.z = z+z_width / 2.;
        verts.push_back(p);

        // vertice 7
        p.x = x+x_width / 2.; p.y = y-y_width / 2.; p.z = z+z_width / 2.;
        verts.push_back(p);
        
        int vert_idx[12][2] = {
            {0,1},
            {1,2},
            {2,3},
            {0,3},
            {0,4},
            {1,5},
            {3,7},
            {2,6},
            {4,5},
            {5,6},
            {4,7},
            {6,7}
        };
        
        for (size_t i=0;i<12;i++){
            line.points.push_back(verts[vert_idx[i][0]]);
            line.points.push_back(verts[vert_idx[i][1]]);
        }
        
        lines.markers.push_back(line);
        
        line.id++;
    }
    // publish
    publisher->publish(lines);
}

void DynamicDetector::transformBBox(const Eigen::Vector3d& center, const Eigen::Vector3d& size, const Eigen::Vector3d& position, const Eigen::Matrix3d& orientation,
    Eigen::Vector3d& newCenter, Eigen::Vector3d& newSize){
double x = center(0); 
double y = center(1);
double z = center(2);
double xWidth = size(0);
double yWidth = size(1);
double zWidth = size(2);

// get 8 bouding boxes coordinates in the camera frame
Eigen::Vector3d p1 (x+xWidth/2.0, y+yWidth/2.0, z+zWidth/2.0);
Eigen::Vector3d p2 (x+xWidth/2.0, y+yWidth/2.0, z-zWidth/2.0);
Eigen::Vector3d p3 (x+xWidth/2.0, y-yWidth/2.0, z+zWidth/2.0);
Eigen::Vector3d p4 (x+xWidth/2.0, y-yWidth/2.0, z-zWidth/2.0);
Eigen::Vector3d p5 (x-xWidth/2.0, y+yWidth/2.0, z+zWidth/2.0);
Eigen::Vector3d p6 (x-xWidth/2.0, y+yWidth/2.0, z-zWidth/2.0);
Eigen::Vector3d p7 (x-xWidth/2.0, y-yWidth/2.0, z+zWidth/2.0);
Eigen::Vector3d p8 (x-xWidth/2.0, y-yWidth/2.0, z-zWidth/2.0);

// transform 8 points to the map coordinate frame
Eigen::Vector3d p1m = orientation * p1 + position;
Eigen::Vector3d p2m = orientation * p2 + position;
Eigen::Vector3d p3m = orientation * p3 + position;
Eigen::Vector3d p4m = orientation * p4 + position;
Eigen::Vector3d p5m = orientation * p5 + position;
Eigen::Vector3d p6m = orientation * p6 + position;
Eigen::Vector3d p7m = orientation * p7 + position;
Eigen::Vector3d p8m = orientation * p8 + position;
std::vector<Eigen::Vector3d> pointsMap {p1m, p2m, p3m, p4m, p5m, p6m, p7m, p8m};

// find max min in x, y, z directions
double xmin=p1m(0); double xmax=p1m(0); 
double ymin=p1m(1); double ymax=p1m(1);
double zmin=p1m(2); double zmax=p1m(2);
for (Eigen::Vector3d pm : pointsMap){
if (pm(0) < xmin){xmin = pm(0);}
if (pm(0) > xmax){xmax = pm(0);}
if (pm(1) < ymin){ymin = pm(1);}
if (pm(1) > ymax){ymax = pm(1);}
if (pm(2) < zmin){zmin = pm(2);}
if (pm(2) > zmax){zmax = pm(2);}
}
newCenter(0) = (xmin + xmax)/2.0;
newCenter(1) = (ymin + ymax)/2.0;
newCenter(2) = (zmin + zmax)/2.0;
newSize(0) = xmax - xmin;
newSize(1) = ymax - ymin;
newSize(2) = zmax - zmin;
}

void DynamicDetector::getYolo3DBBox(const vision_msgs::msg::Detection2D& detection, onboardDetector::box3D& bbox3D, cv::Rect& bboxVis){
    if (this->alignedDepthImage_.empty()){
        return;
    }

    const Eigen::Vector3d humanSize (0.5, 0.5, 1.8); // FIX THAT

    // 1. retrive 2D detection result
    int topX = int(detection.bbox.center.position.x); 
    int topY = int(detection.bbox.center.position.y); 
    int xWidth = int(detection.bbox.size_x); 
    int yWidth = int(detection.bbox.size_y); 
    bboxVis.x = topX;
    bboxVis.y = topY;
    bboxVis.height = yWidth;
    bboxVis.width = xWidth;

    // 2. get thickness estimation (double MAD: double Median Absolute Deviation)
    uint16_t* rowPtr;
    double depth;
    const double inv_factor = 1.0 / this->depthScale_;
    int vMin = std::max(topY, this->depthFilterMargin_);
    int uMin = std::max(topX, this->depthFilterMargin_);
    int vMax = std::min(topY+yWidth, this->imgRows_-this->depthFilterMargin_);
    int uMax = std::min(topX+xWidth, this->imgCols_-this->depthFilterMargin_);
    std::vector<double> depthValues;


    // record the depth values in the potential regions
    for (int v=vMin; v<vMax; ++v){ // row
        rowPtr = this->alignedDepthImage_.ptr<uint16_t>(v);
        for (int u=uMin; u<uMax; ++u){ // column
            depth = (*rowPtr) * inv_factor;
            if (depth >= this->depthMinValue_ and depth <= this->depthMaxValue_){
                depthValues.push_back(depth);
            }
            ++rowPtr;
        }
    }
    if (depthValues.size() == 0){ // in case of out of range
        return;
    }

    // double MAD calculation
    double depthMedian, MAD;
    this->calculateMAD(depthValues, depthMedian, MAD);
    // cout << "MAD: " << MAD << endl;

    double depthMin = 10.0; double depthMax = -10.0;
    // find min max depth value
    for (int v=vMin; v<vMax; ++v){ // row
        rowPtr = this->alignedDepthImage_.ptr<uint16_t>(v);
        for (int u=uMin; u<uMax; ++u){ // column
            depth = (*rowPtr) * inv_factor;
            if (depth >= this->depthMinValue_ and depth <= this->depthMaxValue_){
                if ((depth < depthMin) and (depth >= depthMedian - 1.5 * MAD)){
                    depthMin = depth;
                }

                if ((depth > depthMax) and (depth <= depthMedian + 1.5 * MAD)){
                    depthMax = depth;
                }
            }
            ++rowPtr;
        }
    }
    
    if (depthMin == 10.0 or depthMax == -10.0){ // in case the depth value is not available
        return;
    }

    // 3. project points into 3D in the camera frame
    Eigen::Vector3d pUL, pBR, center;
    pUL(0) = (topX - this->cxC_) * depthMedian / this->fxC_;
    pUL(1) = (topY - this->cyC_) * depthMedian / this->fyC_;
    pUL(2) = depthMedian;

    pBR(0) = (topX + xWidth - this->cxC_) * depthMedian / this->fxC_;
    pBR(1) = (topY + yWidth- this->cyC_) * depthMedian / this->fyC_;
    pBR(2) = depthMedian;

    center(0) = (pUL(0) + pBR(0))/2.0;
    center(1) = (pUL(1) + pBR(1))/2.0;
    center(2) = depthMedian;

    double xWidth3D = std::abs(pBR(0) - pUL(0));
    double yWidth3D = std::abs(pBR(1) - pUL(1));
    double zWidth3D = depthMax - depthMin; 
    if ((zWidth3D/humanSize(2)>=2.0) or (zWidth3D/humanSize(2) <= 0.5)){ // error is too large, then use the predefined size
        zWidth3D = humanSize(2);
    }       
    Eigen::Vector3d size (xWidth3D, yWidth3D, zWidth3D);

    // 4. transform 3D points into world frame
    Eigen::Vector3d newCenter, newSize;
    this->transformBBox(center, size, this->positionColor_, this->orientationColor_, newCenter, newSize);
    bbox3D.x = newCenter(0);
    bbox3D.y = newCenter(1);
    bbox3D.z = newCenter(2);

    bbox3D.x_width = newSize(0);
    bbox3D.y_width = newSize(1);
    bbox3D.z_width = newSize(2);

    // 5. check the bounding box size. If the bounding box size is too different from the predefined size, overwrite the size
    if ((bbox3D.x_width/humanSize(0)>=2.0) or (bbox3D.x_width/humanSize(0)<=0.5)){
        bbox3D.x_width = humanSize(0);
    }

    if ((bbox3D.y_width/humanSize(1)>=2.0) or (bbox3D.y_width/humanSize(1)<=0.5)){
        bbox3D.y_width = humanSize(1);
    }

    if ((bbox3D.z_width/humanSize(2)>=2.0) or (bbox3D.z_width/humanSize(2)<=0.5)){
        bbox3D.z = humanSize(2)/2.;
        bbox3D.z_width = humanSize(2);
    }
}

bool DynamicDetector::isInFov(const Eigen::Vector3d& position, const Eigen::Matrix3d& orientation, Eigen::Vector3d& point){
    Eigen::Vector3d worldRay = point - position;
    Eigen::Vector3d camUnitX(1,0,0);
    Eigen::Vector3d camUnitY(0,1,0);
    Eigen::Vector3d camUnitZ(0,0,1);
    Eigen::Vector3d camRay;
    Eigen::Vector3d displacement; 

    // z is in depth direction in camera coord
    camRay = orientation.inverse()*worldRay;
    double camRayX = abs(camRay.dot(camUnitX));
    double camRayY = abs(camRay.dot(camUnitY));
    double camRayZ = abs(camRay.dot(camUnitZ));

    double htan = camRayX/camRayZ;
    double vtan = camRayY/camRayZ;
    
    double pi = 3.1415926;
    return htan<tan(42*pi/180) && vtan<tan(28*pi/180) && camRayZ<this->depthMaxValue_; // FIX THAT
}

void DynamicDetector::updatePoseHist(){
    if (int(this->positionHist_.size()) == this->skipFrame_){
        this->positionHist_.pop_back();
    }
    else{
        this->positionHist_.push_front(this->position_);
    }
    if (int(this->orientationHist_.size()) == this->skipFrame_){
        this->orientationHist_.pop_back();
    }
    else{
        this->orientationHist_.push_front(this->orientation_);
    }
}


void DynamicDetector::calculateMAD(std::vector<double>& depthValues, double& depthMedian, double& MAD){
    std::sort(depthValues.begin(), depthValues.end());
    int medianIdx = int(depthValues.size()/2);
    depthMedian = depthValues[medianIdx]; // median of all data

    std::vector<double> deviations;
    for (size_t i=0; i<depthValues.size(); ++i){
        deviations.push_back(std::abs(depthValues[i] - depthMedian));
    }
    std::sort(deviations.begin(), deviations.end());
    MAD = deviations[int(deviations.size()/2)];
}

int DynamicDetector::getBestOverlapBBox(const onboardDetector::box3D& currBBox, const std::vector<onboardDetector::box3D>& targetBBoxes, double& bestIOU){
    bestIOU = 0.0;
    int bestIOUIdx = -1; // no match
    for (size_t i=0; i<targetBBoxes.size(); ++i){
        onboardDetector::box3D targetBBox = targetBBoxes[i];
        double IOU = this->calBoxIOU(currBBox, targetBBox);
        if (IOU > bestIOU){
            bestIOU = IOU;
            bestIOUIdx = i;
        }
    }
    return bestIOUIdx;
}

void DynamicDetector::getDynamicPc(std::vector<Eigen::Vector3d>& dynamicPc){
    Eigen::Vector3d curPoint;
    for (size_t i=0 ; i<this->filteredPoints_.size() ; ++i){
        curPoint = this->filteredPoints_[i];
        for (size_t j=0; j<this->dynamicBBoxes_.size() ; ++j){
            if (abs(curPoint(0)-this->dynamicBBoxes_[j].x)<=this->dynamicBBoxes_[j].x_width/2 and 
                abs(curPoint(1)-this->dynamicBBoxes_[j].y)<=this->dynamicBBoxes_[j].y_width/2 and 
                abs(curPoint(2)-this->dynamicBBoxes_[j].z)<=this->dynamicBBoxes_[j].z_width/2) {
                    dynamicPc.push_back(curPoint);
                    break;
                }
        }
    }
}

void DynamicDetector::publishPoints(const std::vector<Eigen::Vector3d>& points, const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher){
    pcl::PointXYZ pt;
    pcl::PointCloud<pcl::PointXYZ> cloud;        
    for (size_t i=0; i<points.size(); ++i){
        pt.x = points[i](0);
        pt.y = points[i](1);
        pt.z = points[i](2);
        cloud.push_back(pt);
    }    
    cloud.width = cloud.points.size();
    cloud.height = 1; // FIX THAT
    cloud.is_dense = true;
    cloud.header.frame_id = "map"; // FIX THAT

    sensor_msgs::msg::PointCloud2 cloudMsg;
    pcl::toROSMsg(cloud, cloudMsg);
    publisher->publish(cloudMsg);
}

void DynamicDetector::publishYoloImages(){
    sensor_msgs::msg::Image::ConstSharedPtr detectedAlignedImgMsg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", this->detectedAlignedDepthImg_).toImageMsg();
    this->detectedAlignedDepthImgPub_.publish(detectedAlignedImgMsg);
}

void DynamicDetector::publishHistoryTraj(){
    visualization_msgs::msg::MarkerArray trajMsg;
    int countMarker = 0;
    for (size_t i=0; i<this->boxHist_.size(); ++i){
        visualization_msgs::msg::Marker traj;
        traj.header.frame_id = "map"; // FIX THAT
        traj.header.stamp = this->now();
        traj.ns = "dynamic_detector";
        traj.id = countMarker;
        traj.type = visualization_msgs::msg::Marker::LINE_LIST;
        traj.scale.x = 0.03;
        traj.scale.y = 0.03;
        traj.scale.z = 0.03;
        traj.color.a = 1.0; // Don't forget to set the alpha!
        traj.color.r = 0.0;
        traj.color.g = 1.0;
        traj.color.b = 0.0;
        for (size_t j=0; j<this->boxHist_[i].size()-1; ++j){
            geometry_msgs::msg::Point p1, p2;
            onboardDetector::box3D box1 = this->boxHist_[i][j];
            onboardDetector::box3D box2 = this->boxHist_[i][j+1];
            p1.x = box1.x; p1.y = box1.y; p1.z = box1.z;
            p2.x = box2.x; p2.y = box2.y; p2.z = box2.z;
            traj.points.push_back(p1);
            traj.points.push_back(p2);
        }

        ++countMarker;
        trajMsg.markers.push_back(traj);
    }
    this->historyTrajPub_->publish(trajMsg);
}

void DynamicDetector::publishVelVis(){ // publish velocities for all tracked objects
    visualization_msgs::msg::MarkerArray velVisMsg;
    int countMarker = 0;
    for (size_t i=0; i<this->trackedBBoxes_.size(); ++i){
        visualization_msgs::msg::Marker velMarker;
        velMarker.header.frame_id = "map"; // FIX THAT
        velMarker.header.stamp = this->now();
        velMarker.ns = "dynamic_detector";
        velMarker.id =  countMarker;
        velMarker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        velMarker.pose.position.x = this->trackedBBoxes_[i].x;
        velMarker.pose.position.y = this->trackedBBoxes_[i].y;
        velMarker.pose.position.z = this->trackedBBoxes_[i].z + this->trackedBBoxes_[i].z_width/2. + 0.3;
        velMarker.scale.x = 0.15;
        velMarker.scale.y = 0.15;
        velMarker.scale.z = 0.15;
        velMarker.color.a = 1.0;
        velMarker.color.r = 1.0;
        velMarker.color.g = 0.0;
        velMarker.color.b = 0.0;
        velMarker.lifetime = rclcpp::Duration::from_seconds(0.1);
        double vx = this->trackedBBoxes_[i].Vx;
        double vy = this->trackedBBoxes_[i].Vy;
        double vNorm = sqrt(vx*vx+vy*vy);
        std::string velText = "Vx=" + std::to_string(vx) + ", Vy=" + std::to_string(vy) + ", |V|=" + std::to_string(vNorm);
        velMarker.text = velText;
        velVisMsg.markers.push_back(velMarker);
        ++countMarker;
    }
    this->velVisPub_->publish(velVisMsg);
}

}
