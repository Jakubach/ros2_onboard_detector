/*
	FILE: detector_node.cpp
	--------------------------
	Run detector node
*/
#include <rclcpp/rclcpp.hpp>
#include <onboard_detector/dynamicDetector.h>

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<onboardDetector::DynamicDetector>();
    node->initDetector();
    rclcpp::spin(node);
    rclcpp::shutdown();
    
    return 0;
}