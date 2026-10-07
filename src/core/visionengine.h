#pragma once

#include "visiontypes.h"
#include <functional>
#include <memory>
#include <opencv2/dnn.hpp>

namespace vision
{
class PtBackend;
class OnnxCudaBackend;

// Instances belong to one inference thread; cv::dnn::Net is not shared with the UI.
class VisionEngine final
{
  public:
    VisionEngine();
    ~VisionEngine();
    VisionEngine(const VisionEngine &) = delete;
    VisionEngine &operator=(const VisionEngine &) = delete;
    void load(const ModelConfig &config);
    void unload();
    InferenceResult infer(const QImage &image, const QString &source = {});
    bool loaded() const noexcept;
    QString backendName() const;
    void setCancellationCheck(std::function<bool()> check);
    const ModelConfig &config() const noexcept
    {
        return m_config;
    }

  private:
    cv::dnn::Net m_net;
    std::unique_ptr<PtBackend> m_pt;
    std::unique_ptr<OnnxCudaBackend> m_cuda;
    ModelConfig m_config;
    std::function<bool()> m_cancellationCheck;
};

} // namespace vision
