// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

#include <QList>
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
    explicit WorkbenchPage(const QString &exportDirectory, QWidget *parent = nullptr);
    const Controls &controls() const { return controls_; }
    void setExportDirectory(const QString &directory);

  signals:
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

  private:
    Controls controls_;
};
