#ifndef _STEREO_CAMERA_HPP__
#define _STEREO_CAMERA_HPP__

#include <iostream>
#include <memory>
#include <sstream>
#include <fstream>

#include <opencv2/opencv.hpp>

#include "withrobot_camera.hpp"

struct StereoCameraData
{
    uint64_t timestamp_ms;

    cv::Mat frame_0; // left camera
    cv::Mat frame_1; // right camera
};

struct StereoCameraConfig
{
    std::string devPath = "/dev/video0";
    int width = 1280;
    int height = 960;
    int fps = 30;
    int exposure = 150;
    int gain = 100;
    int white_balance_blue = 180;
    int white_balance_red = 150;
    bool ae = false;
};

class StereoCamera
{
public:
    typedef std::shared_ptr<StereoCamera> Ptr;

    StereoCamera();
    StereoCamera(struct StereoCameraConfig config);
    ~StereoCamera();

    struct StereoCameraConfig getCurrentConfig() { return _cameraConfig; }
    void updateConfig(struct StereoCameraConfig config);

    bool getCamData(StereoCameraData &camData);

    inline bool checkCameraStarted() { return m_isCameraStarted; }

private:
    void setup();
    void reconfigCamera();
    void enum_device_list();

    std::shared_ptr<Withrobot::Camera> camera;
    Withrobot::camera_format camFormat;

    bool m_isCameraStarted;
    std::string _devPath;
    struct StereoCameraConfig _cameraConfig;
};

#endif //_STEREO_CAMERA_HPP__