#pragma once

#include "visiontypes.h"
#include <QByteArray>
#include <functional>
#include <memory>
#include <opencv2/core.hpp>
#include <vector>

namespace vision
{
// Optional ONNX Runtime C API. Neither ORT nor CUDA is linked into the CPU app.
// One session belongs to one inference worker; output Mats own their pixels.
class OnnxCudaBackend final
{
  public:
    OnnxCudaBackend();
    ~OnnxCudaBackend();
    OnnxCudaBackend(const OnnxCudaBackend &) = delete;
    OnnxCudaBackend &operator=(const OnnxCudaBackend &) = delete;
    void load(const ModelConfig &config, const QByteArray &model);
    bool loaded() const noexcept;
    QString deviceName() const;
    std::vector<cv::Mat> infer(const cv::Mat &blob);
    void setCancellationCheck(std::function<bool()> check);

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
    std::function<bool()> m_cancellationCheck;
};
} // namespace vision
