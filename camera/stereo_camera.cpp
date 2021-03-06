#include "stereo_camera.hpp"

StereoCamera::StereoCamera()
    : camera(nullptr)
{
    this->setup();
}

StereoCamera::StereoCamera(struct StereoCameraConfig config) : _cameraConfig(config)
{
    this->setup();
}

StereoCamera::~StereoCamera()
{
    camera->stop();
}

void StereoCamera::setup()
{
    this->enum_device_list();

    camera = std::make_shared<Withrobot::Camera>(_devPath.c_str());

    camera->set_format(_cameraConfig.width, _cameraConfig.height, Withrobot::fourcc_to_pixformat('Y', 'U', 'Y', 'V'), 1, _cameraConfig.fps);

    camera->get_current_format(camFormat);
    camFormat.print();

    if (camera->start())
    {
        m_isCameraStarted = true;
        reconfigCamera();
    }
    else
    {
        m_isCameraStarted = false;
    }
}

void StereoCamera::enum_device_list()
{
    std::vector<Withrobot::usb_device_info> dev_list;
    int dev_num = Withrobot::get_usb_device_info_list(dev_list);

    if (dev_num < 1)
    {
        dev_list.clear();
        return;
    }
    for (uint8_t i = 0; i < dev_list.size(); ++i)
    {
        if (dev_list[i].product == "oCamS-1CGN-U")
        {
            _devPath = dev_list[i].dev_node;
            return;
        }
        else
        {
            // Workaround for calibration step
            _devPath = "";
        }
    }
    return;
}

void StereoCamera::reconfigCamera()
{
    camera->set_control("Exposure (Absolute)", _cameraConfig.exposure);
    camera->set_control("Gain", _cameraConfig.gain);
    camera->set_control("White Balance Blue Component", _cameraConfig.white_balance_blue);
    camera->set_control("White Balance Red Component", _cameraConfig.white_balance_red);
    if (_cameraConfig.ae)
        camera->set_control("Exposure, Auto", 0x3);
    else
        camera->set_control("Exposure, Auto", 0x1);
}

void StereoCamera::updateConfig(struct StereoCameraConfig config)
{
    _cameraConfig = config;
    this->reconfigCamera();
}

bool StereoCamera::getCamData(StereoCameraData &camData)
{
    cv::Mat temp(cv::Size(camFormat.width, camFormat.height), CV_8UC2);
    if (camera->get_frame(temp.data, camFormat.image_size, 1) != -1)
    {
        memcpy(&camData.timestamp_ms, temp.data, sizeof(uint32_t));

        cv::Mat stereo_raw[2];
        cv::split(temp, stereo_raw);

        cv::cvtColor(stereo_raw[1], camData.frame_0, CV_BayerGR2GRAY);
        cv::cvtColor(stereo_raw[0], camData.frame_1, CV_BayerGR2GRAY);

        return true;
    }
    return false;
}
