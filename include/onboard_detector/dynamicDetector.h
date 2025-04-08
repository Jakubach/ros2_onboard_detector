#ifndef DYNAMIC_DETECTOR_HPP_
#define DYNAMIC_DETECTOR_HPP_


#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include <image_transport/image_transport.hpp>
#include <image_transport/publisher.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <memory>
#include <string>

#include <cv_bridge/cv_bridge.h>
#include "onboard_detector/utils.h"
#include "onboard_detector/kalmanFilter.h"
#include "onboard_detector/uvDetector.h"
#include "onboard_detector/dbscan.h"

namespace onboardDetector {

class DynamicDetector : public rclcpp::Node
{
public:
  DynamicDetector();
  DynamicDetector(const rclcpp::NodeOptions & options);

  void initDetector();

private:
  void initParam();
  void registerCallback();
  void registerPub();

  std::string ns_;
  std::string hint_;

  std::string depthTopicName_;
  std::string alignedDepthTopicName_;
  std::string poseTopicName_;
  std::string odomTopicName_;

  int localizationMode_;



  double raycastMaxLength_;
  int voxelOccThresh_;
  double groundHeight_;

  int dbMinPointsCluster_;
  double dbEpsilon_;

  double boxIOUThresh_;
  double yoloOverwriteDistance_;

  int histSize_;
  double dt_;
  double simThresh_;
  int skipFrame_;
  double dynaVelThresh_;
  double dynaVoteThresh_;
  double maxSkipRatio_;
  int fixSizeHistThresh_;
  double fixSizeDimThresh_;

  double eP_; // kalman filter initial uncertainty matrix
  double eQPos_; // motion model uncertainty matrix for position
  double eQVel_; // motion model uncertainty matrix for velocity
  double eQAcc_; // motion model uncertainty matrix for acceleration
  double eRPos_; // observation uncertainty matrix for position
  double eRVel_; // observation uncertainty matrix for velocity
  double eRAcc_; // observation uncertainty matrix for acceleration

  
  image_transport::Publisher uvDepthMapPub_;
  image_transport::Publisher uDepthMapPub_;
  image_transport::Publisher uvBirdViewPub_;
  image_transport::Publisher detectedAlignedDepthImgPub_;

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr uvBBoxesPub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr dynamicPointsPub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filteredPointsPub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr dbBBoxesPub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr yoloBBoxesPub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr filteredBBoxesPub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr trackedBBoxesPub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr dynamicBBoxesPub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr historyTrajPub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr velVisPub_;

  // === Subscribers (standard) ===
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr alignedDepthSub_;
  rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr yoloDetectionSub_;

  // === Timers === 
  rclcpp::TimerBase::SharedPtr detectionTimer_;
  rclcpp::TimerBase::SharedPtr trackingTimer_;
  rclcpp::TimerBase::SharedPtr classificationTimer_;
  rclcpp::TimerBase::SharedPtr visTimer_;

  // === Message filters ===
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> depthSub_;
  std::shared_ptr<message_filters::Subscriber<geometry_msgs::msg::PoseStamped>> poseSub_;
  std::shared_ptr<message_filters::Subscriber<nav_msgs::msg::Odometry>> odomSub_;

  using depthPoseSync = message_filters::sync_policies::ApproximateTime<
  sensor_msgs::msg::Image, geometry_msgs::msg::PoseStamped>;
  std::shared_ptr<message_filters::Synchronizer<depthPoseSync>> depthPoseSync_;

  using depthOdomSync = message_filters::sync_policies::ApproximateTime<
  sensor_msgs::msg::Image, nav_msgs::msg::Odometry>;
  std::shared_ptr<message_filters::Synchronizer<depthOdomSync>> depthOdomSync_;

  cv::Mat depthImage_;
  cv::Mat alignedDepthImage_;
  Eigen::Vector3d position_; // depth camera position
  Eigen::Matrix3d orientation_; // depth camera orientation
  Eigen::Vector3d positionColor_; // color camera position
  Eigen::Matrix3d orientationColor_; // color camera orientation
  Eigen::Vector3d localSensorRange_ {5.0, 5.0, 5.0};

  cv::Mat detectedAlignedDepthImg_;
  std::vector<onboardDetector::box3D> yoloBBoxes_; // yolo detected bounding boxes
  vision_msgs::msg::Detection2DArray yoloDetectionResults_; // yolo detected 2D results

  bool newDetectFlag_;
  std::vector<std::deque<onboardDetector::box3D>> boxHist_; // data association result: history of filtered bounding boxes for each box in current frame
  std::vector<std::deque<std::vector<Eigen::Vector3d>>> pcHist_; // data association result: history of filtered pc clusteres for each pc cluster in current frame
  std::deque<Eigen::Vector3d> positionHist_; // current position
  std::deque<Eigen::Matrix3d> orientationHist_; // current orientation
  std::vector<onboardDetector::kalman_filter> filters_; // kalman filter for each objects


  int forceDynaFrames_;
  int forceDynaCheckRange_;
  int dynamicConsistThresh_;
  int kfAvgFrames_;
  bool constrainSize_;
  std::vector<Eigen::Vector3d> targetObjectSize_; 





  void depthOdomCB(
    const sensor_msgs::msg::Image::ConstSharedPtr img,
    const nav_msgs::msg::Odometry::ConstSharedPtr odom);

  void depthPoseCB(
    const sensor_msgs::msg::Image::ConstSharedPtr img,
    const geometry_msgs::msg::PoseStamped::ConstSharedPtr pose);

  void trackingCB();
  void detectionCB();
  void yoloDetectionCB(const vision_msgs::msg::Detection2DArray::SharedPtr detections);
  void alignedDepthCB(const sensor_msgs::msg::Image::ConstSharedPtr img);
  void classificationCB();
  void visCB();


  // inline helper functions
  bool isInFilterRange(const Eigen::Vector3d& pos);
  void posToIndex(const Eigen::Vector3d& pos, Eigen::Vector3i& idx, double res);
  int indexToAddress(const Eigen::Vector3i& idx, double res);
  int posToAddress(const Eigen::Vector3d& pos, double res);
  void indexToPos(const Eigen::Vector3i& idx, Eigen::Vector3d& pos, double res);
  void getCameraPose(const geometry_msgs::msg::PoseStamped::ConstSharedPtr& pose, Eigen::Matrix4d& camPoseMatrix, Eigen::Matrix4d& camPoseColorMatrix);
  void getCameraPose(const nav_msgs::msg::Odometry::ConstSharedPtr& odom, Eigen::Matrix4d& camPoseMatrix, Eigen::Matrix4d& camPoseColorMatrix);
  onboardDetector::Point eigenToDBPoint(const Eigen::Vector3d& p);
  Eigen::Vector3d dbPointToEigen(const onboardDetector::Point& pDB);
  void eigenToDBPointVec(const std::vector<Eigen::Vector3d>& points, std::vector<onboardDetector::Point>& pointsDB, int size);


  void getDynamicObstacles(std::vector<onboardDetector::box3D>& incomeDynamicBBoxes, const Eigen::Vector3d &robotSize = Eigen::Vector3d(0.0,0.0,0.0));

 
  void uvDetect();
  void dbscanDetect();
  void yoloDetectionTo3D();
  void filterBBoxes();

  // Data association and tracking functions
  void boxAssociation(std::vector<int>& bestMatch);
  void boxAssociationHelper(std::vector<int>& bestMatch);
  void genFeat(const std::vector<onboardDetector::box3D>& propedBoxes, int numObjs, std::vector<Eigen::VectorXd>& propedBoxesFeat, std::vector<Eigen::VectorXd>& currBoxesFeat);
  void genFeatHelper(std::vector<Eigen::VectorXd>& feature, const std::vector<onboardDetector::box3D>& boxes);
  void linearProp(std::vector<onboardDetector::box3D>& propedBoxes);
  void findBestMatch(const std::vector<Eigen::VectorXd>& propedBoxesFeat, const std::vector<Eigen::VectorXd>& currBoxesFeat, const std::vector<onboardDetector::box3D>& propedBoxes, std::vector<int>& bestMatch);
  void kalmanFilterAndUpdateHist(const std::vector<int>& bestMatch);
  void kalmanFilterMatrixVel(const onboardDetector::box3D& currDetectedBBox, MatrixXd& states, MatrixXd& A, MatrixXd& B, MatrixXd& H, MatrixXd& P, MatrixXd& Q, MatrixXd& R);
  void kalmanFilterMatrixAcc(const onboardDetector::box3D& currDetectedBBox, MatrixXd& states, MatrixXd& A, MatrixXd& B, MatrixXd& H, MatrixXd& P, MatrixXd& Q, MatrixXd& R);
  void getKalmanObservationVel(const onboardDetector::box3D& currDetectedBBox, int bestMatchIdx, MatrixXd& Z);
  void getKalmanObservationAcc(const onboardDetector::box3D& currDetectedBBox, int bestMatchIdx, MatrixXd& Z);


  // helper function
  void transformBBox(const Eigen::Vector3d& center, const Eigen::Vector3d& size, const Eigen::Vector3d& position, const Eigen::Matrix3d& orientation,
  Eigen::Vector3d& newCenter, Eigen::Vector3d& newSize);
  bool isInFov(const Eigen::Vector3d& position, const Eigen::Matrix3d& orientation, Eigen::Vector3d& point);
  int getBestOverlapBBox(const onboardDetector::box3D& currBBox, const std::vector<onboardDetector::box3D>& targetBBoxes, double& bestIOU);
  void updatePoseHist();


  // DETECTOR DATA
  std::vector<onboardDetector::box3D> uvBBoxes_; // uv detector bounding boxes
  int projPointsNum_ = 0;
  std::vector<Eigen::Vector3d> projPoints_; // projected points from depth image
  std::vector<double> pointsDepth_;
  std::vector<Eigen::Vector3d> filteredPoints_; // filtered point cloud data
  std::vector<onboardDetector::box3D> dbBBoxes_; // DBSCAN bounding boxes  
  std::vector<std::vector<Eigen::Vector3d>> pcClusters_; // pointcloud clusters
  std::vector<Eigen::Vector3d> pcClusterCenters_; // pointcloud cluster centers
  std::vector<Eigen::Vector3d> pcClusterStds_; // pointcloud cluster standard deviation in each axis
  std::vector<onboardDetector::box3D> filteredBBoxes_; // filtered bboxes
  std::vector<std::vector<Eigen::Vector3d>> filteredPcClusters_; // pointcloud clusters after filtering by UV and DBSCAN fusion
  std::vector<Eigen::Vector3d> filteredPcClusterCenters_; // filtered pointcloud cluster centers
  std::vector<Eigen::Vector3d> filteredPcClusterStds_; // filtered pointcloud cluster standard deviation in each axis
  std::vector<onboardDetector::box3D> trackedBBoxes_; // bboxes tracked from kalman filtering
  std::vector<onboardDetector::box3D> dynamicBBoxes_; // boxes classified as dynamic
  // std::vector<int> recentDynaFrames_; // recent number of frames being detected as dynamic for each obstacle

  // DETECTOR
  std::shared_ptr<onboardDetector::UVdetector> uvDetector_;
  std::shared_ptr<onboardDetector::DBSCAN> dbCluster_;

  // CAMERA
  double fx_, fy_, cx_, cy_; // depth camera intrinsics
  double depthScale_; // value / depthScale
  double depthMinValue_, depthMaxValue_;
  int depthFilterMargin_, skipPixel_; // depth filter margin
  int imgCols_, imgRows_;
  Eigen::Matrix4d body2Cam_; // from body frame to camera frame
  Eigen::Matrix4d body2CamColor_;

  // CAMERA ALIGNED DEPTH TO COLOR
  double fxC_, fyC_, cxC_, cyC_;

  // visualization
  void getDynamicPc(std::vector<Eigen::Vector3d>& dynamicPc);
  void publishUVImages(); 
  void publishYoloImages();
  void publishPoints(const std::vector<Eigen::Vector3d>& points, const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher);
  void publish3dBox(const std::vector<onboardDetector::box3D>& bboxes, const rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr publisher, double r, double g, double b);
  void publishHistoryTraj();
  void publishVelVis();

  // uv Detector Functions
  void transformUVBBoxes(std::vector<onboardDetector::box3D>& bboxes);

  // DBSCAN Detector Functions
  void projectDepthImage();
  void filterPoints(const std::vector<Eigen::Vector3d>& points, std::vector<Eigen::Vector3d>& filteredPoints);
  void clusterPointsAndBBoxes(const std::vector<Eigen::Vector3d>& points, std::vector<onboardDetector::box3D>& bboxes, std::vector<std::vector<Eigen::Vector3d>>& pcClusters, std::vector<Eigen::Vector3d>& pcClusterCenters, std::vector<Eigen::Vector3d>& pcClusterStds);
  void voxelFilter(const std::vector<Eigen::Vector3d>& points, std::vector<Eigen::Vector3d>& filteredPoints);
  void calcPcFeat(const std::vector<Eigen::Vector3d>& pcCluster, Eigen::Vector3d& pcClusterCenter, Eigen::Vector3d& pcClusterStd);

  // yolo helper functions
  void getYolo3DBBox(const vision_msgs::msg::Detection2D& detection, onboardDetector::box3D& bbox3D, cv::Rect& bboxVis); 
  void calculateMAD(std::vector<double>& depthValues, double& depthMedian, double& MAD);

  // detection helper functions
  double calBoxIOU(const onboardDetector::box3D& box1, const onboardDetector::box3D& box2);

};

} // namespace onboardDetector

#endif // DYNAMIC_DETECTOR_HPP_