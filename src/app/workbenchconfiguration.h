// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include "modelsessioncontroller.h"

namespace vision::app
{
struct WorkbenchDraft
{
    // Editors and evaluation requests use doubles. ModelConfig's float fields
    // are derived only by compose, retaining the existing backend contract.
    double confidence = .25, iou = .45;
    ModelConfig parameters;
    QString classFilterText, labelsPath, exportDirectory;
    int cameraIndex = 0;
    StereoView stereoView = StereoView::Full;
    bool automaticExport = false;
};
struct ClassFilterStatus
{
    QVector<int> ids;
    QString error, hint;
};
class WorkbenchConfiguration final : public QObject
{
    Q_OBJECT
  public:
    explicit WorkbenchConfiguration(QObject *parent = nullptr) : QObject(parent) {}
    WorkbenchDraft snapshot() const { return draft_; }
    void replaceDraft(const WorkbenchDraft &draft) { draft_ = draft; emit changed(); }
    void setTask(ModelTask task);
    void setInputSize(int size);
    void setConfidence(double confidence);
    void setIou(double iou);
    void setScale(double scale);
    void setMeanR(double mean);
    void setMeanG(double mean);
    void setMeanB(double mean);
    void setInputColor(InputColorMode mode, bool swapRB);
    void setComputeDevice(ComputeDevice device);
    void setDeviceIndex(int index);
    void setClassFilterText(const QString &text);
    void setCameraIndex(int index);
    void setStereoView(StereoView view);
    void setAutomaticExport(bool enabled);
    void setLabels(const QStringList &labels, const QString &path)
    { draft_.parameters.labels = labels; draft_.labelsPath = path; emit changed(); }
    bool importLabels(const QString &path, QString *error = nullptr);
    void setExportDirectory(const QString &directory)
    { draft_.exportDirectory = directory; emit changed(); }
    ModelConfig compose(const ModelSessionSnapshot &model, bool convertedProfileAvailable) const;
    ClassFilterStatus classFilterStatus(const ModelSessionSnapshot &model, const QString &identity) const;
    static QString configurationFingerprint(ModelConfig config, const QString &modelIdentity);
  signals:
    void changed();
  private:
    WorkbenchDraft draft_;
};
}
