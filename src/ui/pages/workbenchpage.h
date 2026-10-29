// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

#include "core/visiontypes.h"
#include <QList>
#include <QHash>
#include <QJsonObject>
#include <QStringList>
#include <QWidget>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QLineEdit;
class QListWidget;
class QPushButton;
class QSpinBox;
class QTableWidget;
class QTabWidget;
class QTimer;
class EvaluationPanel;
class ImageCanvas;
class AsyncImagePreview;

// Owns the workbench widget tree and presentation-only interactions. It emits
// user intents; application state, files and workers stay out. The optional
// recording clock emits presentation ticks without owning recording state.
class WorkbenchPage final : public QWidget
{
    Q_OBJECT
  public:
    struct Controls
    {
        QLabel *countMetric = nullptr, *latencyMetric = nullptr, *classMetric = nullptr, *sizeMetric = nullptr,
               *sourceLabel = nullptr, *gpuDeviceLabel = nullptr, *deviceHint = nullptr,
               *stereoLabel = nullptr, *preprocessHint = nullptr, *modelName = nullptr, *modelMeta = nullptr,
               *classFilterHint = nullptr, *meanRLabel = nullptr, *backendBadge = nullptr,
               *rangeHint = nullptr, *canvasTitle = nullptr, *recordingStatus = nullptr,
               *zoomLabel = nullptr, *resultInfo = nullptr, *emptyResults = nullptr,
               *evaluationReviewSummaryLabel = nullptr, *evaluationTaskSummary = nullptr,
               *evaluationRangeHint = nullptr;
        QPushButton *datasetButton = nullptr, *videoButton = nullptr, *cameraButton = nullptr,
                    *modelButton = nullptr, *labelButton = nullptr, *openExportDirectory = nullptr,
                    *runButton = nullptr, *stopButton = nullptr, *recordButton = nullptr,
                    *clearQueue = nullptr, *evaluationReviewPreviousButton = nullptr,
                    *evaluationReviewNextButton = nullptr, *evaluationReviewButton = nullptr,
                    *evaluationStartButton = nullptr, *evaluationStopButton = nullptr;
        QComboBox *computeDevice = nullptr, *gpuDeviceIndex = nullptr, *inputColorMode = nullptr,
                  *stereoView = nullptr, *taskBox = nullptr, *detectionRange = nullptr,
                  *evaluationPreviewFilter = nullptr, *evaluationRange = nullptr;
        QSpinBox *cameraIndex = nullptr, *inputSize = nullptr;
        QDoubleSpinBox *confidence = nullptr, *iou = nullptr, *scale = nullptr,
                       *meanR = nullptr, *meanG = nullptr, *meanB = nullptr;
        QCheckBox *autoExport = nullptr, *showBoxes = nullptr, *showLabels = nullptr, *showGroundTruth = nullptr;
        QLineEdit *classFilter = nullptr;
        QWidget *detectionRangePanel = nullptr, *evaluationReviewBar = nullptr;
        QListWidget *queue = nullptr;
        QTableWidget *predictionTable = nullptr;
        QTabWidget *workbenchTabs = nullptr;
        QTimer *recordingClock = nullptr;
        ImageCanvas *canvas = nullptr;
        EvaluationPanel *evaluation = nullptr;
        QList<QWidget *> lockedControls;
    };
    struct DevicePresentation {
        bool known = false;
        vision::ComputeDevice device = vision::ComputeDevice::CPU;
        int index = -1;
        QString name, notice, backend;
    };
    const vision::InferenceResult &displayedResult() const { return displayedResult_; }
    const vision::ModelConfig &displayedConfig() const { return displayedConfig_; }
    const DevicePresentation &devicePresentation() const { return device_; }
    void clearDisplayedResult() { displayedResult_ = {}; }
    void setDisplayedConfig(const vision::ModelConfig &config) { displayedConfig_ = config; }
    void setDisplayedClassCount(int count) { displayedConfig_.modelClassCount = count; }
    void clearDeviceVerification(bool clearBackend = false)
    { device_.known = false; device_.notice.clear(); if (clearBackend) device_.backend.clear(); }
    void showResolvedDevice(const QString &backend, const vision::ModelConfig &config);
    void showResult(const vision::InferenceResult &result, vision::SourceKind sourceKind,
                    const QString &streamPath, bool taskActive);
    explicit WorkbenchPage(const QString &exportDirectory, QWidget *parent = nullptr);
    const Controls &controls() const { return controls_; }
    struct ConfigurationProjection {
        vision::ModelConfig parameters;
        double confidence = .25, iou = .45;
        QString classFilterText;
        int cameraIndex = 0;
        vision::StereoView stereoView = vision::StereoView::Full;
        bool automaticExport = false;
    };
    // A write-only projection of the application draft. Signals remain blocked
    // while rendering; rounded editor values never become a second authority.
    void showConfiguration(const ConfigurationProjection &configuration);
    void selectFirstReviewOnNextFilter(bool enabled = true) { selectFirstReview_ = enabled; }
    bool takeSelectFirstReview() { const bool value = selectFirstReview_; selectFirstReview_ = false; return value; }
    void setExportDirectory(const QString &directory);
    struct InputPresentation { bool taskActive = false, failed = false, replacing = false; };
    InputPresentation inputPresentation() const { return input_; }
    void beginInputPresentation() { input_ = {true, false, false}; }
    void endInputPresentation() { input_.taskActive = false; }
    void markInputFailed() { input_.failed = true; }
    void markInputReplacing() { input_.replacing = true; }
    // These are transient projections of controller-owned batches. The page
    // never adds dataset members or advances the controller generation.
    void projectQueue(const QStringList &paths, int firstRow, quint64 generation,
                      const QHash<QString, QJsonObject> &samples,
                      vision::InputColorMode colorMode, bool datasetContext);
    void cancelQueueProjection();
    bool isQueueProjectionActive() const { return queueProjectionActive_; }
    void cancelImagePreview();
    void requestImagePreview(const QString &primary, const QString &fallback, vision::InputColorMode colorMode);
    quint64 previewGeneration() const { return previewGeneration_; }
    // Normal shutdown also cancels the local queue timer; no GUI-thread wait.
    void requestPreviewShutdown();
    bool isPreviewShutdownReady() const;

  signals:
    // Field edits precede the corresponding legacy user-intent signal so its
    // consumer always observes the patched application configuration.
    void taskEdited(vision::ModelTask task);
    void inputSizeEdited(int value);
    void confidenceEdited(double value);
    void iouEdited(double value);
    void scaleEdited(double value);
    void meanREdited(double value);
    void meanGEdited(double value);
    void meanBEdited(double value);
    void inputColorEdited(vision::InputColorMode mode, bool swapRB);
    void computeDeviceEdited(vision::ComputeDevice device);
    void deviceIndexEdited(int value);
    void classFilterEdited(const QString &value);
    void cameraIndexEdited(int value);
    void stereoViewEdited(vision::StereoView view);
    void automaticExportEdited(bool value);
    void videoRequested();
    void cameraRequested();
    void deviceChanged();
    void importModelRequested();
    void importLabelsRequested();
    void defaultLabelsRequested();
    void classFilterChanged(const QString &text);
    void parametersChanged();
    void taskChanged(int index);
    void inputColorChanged();
    void stereoChanged();
    void cameraIndexChanged();
    void rangeChanged();
    void startRequested();
    void stopRequested();
    void openExportDirectoryRequested();
    void recordRequested();
    void recordingTick();
    void filesDropped(const QStringList &paths);
    void clearQueueRequested();
    void queueRowChanged(int row);
    void queueSelectionChanged(int row, bool selected);
    void reviewFilterRequested(const QString &filter);
    void reviewNavigateRequested(int step);
    void reviewErrorsRequested();
    void workbenchTabChanged(int index);
    void queueProjectionFinished(quint64 generation, int projectedCount);
    void projectionChanged();
    void previewReady(quint64 generation, QString path, QImage original, QImage preview,
                      bool primaryUsed, QString error);

  private:
    Controls controls_;
    bool selectFirstReview_ = false;
    vision::InferenceResult displayedResult_;
    vision::ModelConfig displayedConfig_;
    DevicePresentation device_;
    InputPresentation input_;
    QTimer *queueProjectionTimer_ = nullptr;
    bool queueProjectionActive_ = false, inputViewShuttingDown_ = false;
    AsyncImagePreview *previewLoader_ = nullptr;
    quint64 previewGeneration_ = 0;
};
