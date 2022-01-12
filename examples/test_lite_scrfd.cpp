//
// Created by DefTruth on 2021/12/30.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../hub/onnx/cv/scrfd_2.5g_bnkps_shape640x640.onnx";
  std::string test_img_path = "../resources/4.jpg";
  std::string save_img_path = "../logs/4.jpg";

  auto *scrfd = new lite::cv::face::detect::SCRFD(onnx_path);

  std::vector<lite::types::BoxfWithLandmarks> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  scrfd->detect(img_bgr, detected_boxes, 0.3f);

  lite::utils::draw_boxes_with_landmarks_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Default Version Done! Detected Face Num: " << detected_boxes.size() << std::endl;

  delete scrfd;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../hub/onnx/cv/scrfd_2.5g_bnkps_shape640x640.onnx";
  std::string test_img_path = "../resources/6.jpg";
  std::string save_img_path = "../logs/6.jpg";

  auto *scrfd = new lite::onnxruntime::cv::face::detect::SCRFD(onnx_path);

  std::vector<lite::types::BoxfWithLandmarks> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  scrfd->detect(img_bgr, detected_boxes, 0.3f);

  lite::utils::draw_boxes_with_landmarks_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "ONNXRuntime Version Done! Detected Face Num: " << detected_boxes.size() << std::endl;

  delete scrfd;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
  std::string mnn_path = "../hub/mnn/cv/scrfd_2.5g_bnkps_shape640x640.mnn";
  std::string test_img_path = "../resources/12.jpg";
  std::string save_img_path = "../logs/12.jpg";

  auto *scrfd = new lite::mnn::cv::face::detect::SCRFD(mnn_path);

  std::vector<lite::types::BoxfWithLandmarks> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  scrfd->detect(img_bgr, detected_boxes, 0.3f);

  lite::utils::draw_boxes_with_landmarks_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "MNN Version Done! Detected Face Num: " << detected_boxes.size() << std::endl;

  delete scrfd;
#endif
}

static void test_ncnn()
{
#ifdef ENABLE_NCNN
  std::string param_path = "../hub/ncnn/cv/scrfd_2.5g_bnkps_shape640x640.opt.param";
  std::string bin_path = "../hub/ncnn/cv/scrfd_2.5g_bnkps_shape640x640.opt.bin";
  std::string test_img_path = "../resources/1.jpg";
  std::string save_img_path = "../logs/1.jpg";

  auto *scrfd = new lite::ncnn::cv::face::detect::SCRFD(param_path, bin_path, 1, 640, 640);

  std::vector<lite::types::BoxfWithLandmarks> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  scrfd->detect(img_bgr, detected_boxes, 0.3f);

  lite::utils::draw_boxes_with_landmarks_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "NCNN Version Done! Detected Face Num: " << detected_boxes.size() << std::endl;

  delete scrfd;
#endif
}

static void test_tnn()
{
#ifdef ENABLE_TNN
  std::string proto_path = "../hub/tnn/cv/scrfd_2.5g_bnkps_shape640x640.opt.tnnproto";
  std::string model_path = "../hub/tnn/cv/scrfd_2.5g_bnkps_shape640x640.opt.tnnmodel";
  std::string test_img_path = "../resources/9.jpg";
  std::string save_img_path = "../logs/9.jpg";

  auto *scrfd = new lite::tnn::cv::face::detect::SCRFD(proto_path, model_path);

  std::vector<lite::types::BoxfWithLandmarks> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  scrfd->detect(img_bgr, detected_boxes, 0.3f);

  lite::utils::draw_boxes_with_landmarks_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "TNN Version Done! Detected Face Num: " << detected_boxes.size() << std::endl;

  delete scrfd;
#endif
}

static void test_lite()
{
  test_default();
  test_onnxruntime();
  test_mnn();
  test_ncnn();
  test_tnn();
}

int main(__unused int argc, __unused char *argv[])
{
  test_lite();
  return 0;
}