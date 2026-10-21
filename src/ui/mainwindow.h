#pragma once
#include "core/visiontypes.h"
#include <QElapsedTimer>
#include <QHash>
#include <QJsonArray>
#include <QJsonObject>
#include <QMainWindow>
#include <QVariantMap>
#include <memory>
#include <atomic>
#include <functional>
class QComboBox;
class QLineEdit;
class QDoubleSpinBox;
class QSpinBox;
class QCheckBox;
class QLabel;
class QPushButton;
class QListWidget;
class QTableWidget;
class QStackedWidget;
class QProgressBar;
class QThread;
class QThreadPool;
class QSettings;
class QTimer;
class QProcess;
class QPlainTextEdit;
class QTemporaryDir;
class ImageCanvas;
class ModelViewer;
class ModelConversionPage;
class EvaluationPanel;
class QTabWidget;
class AsyncImagePreview;
namespace cv
{
class VideoCapture;
}
namespace vision
{
class InferenceWorker;
class EvaluationWorker;
}
class MainWindow : public QMainWindow
{
    Q_OBJECT
    friend class UiTests;
    friend class BatchResponsivenessTests;
    friend class ScopeFilterUiTests;
    friend class EvaluationIntegrationTests;
    friend class DatasetSessionUiTests;
    friend class InputDropUiTests;
  public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;
    void saveScreenshot(const QString &path);
    void runSmoke(const QString &outputDir);
    void runDemo();
    void runPtDemo();
    void runPtSmoke(const QString &outputDir);
    void runModelSmoke(const QString &outputDir, const QString &model = {});
    void runEvaluationSmoke(const QString &outputDir, const QString &requestPath);
    void runDatasetSessionSmoke(const QString &outputDir);
    void runPreviewDropSmoke(const QString &outputDir, const QString &requestPath);
    void showModelStructure(const QString &model = {});
    void setComputeDevice(vision::ComputeDevice device, int index = 0);
    void restoreDatasetSession();
    QString datasetSessionFilePath() const;
  signals:
    void startRequested(vision::JobRequest request);

  protected:
    void closeEvent(QCloseEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;
    bool eventFilter(QObject *watched, QEvent *event) override;

  private:
    QWidget *buildSidebar();
    QWidget *buildWorkbench();
    QWidget *buildModels();
    QWidget *buildModelDisplay();
    QWidget *buildHistory();
    QWidget *buildGuide();
    QWidget *buildRecordings();
    QWidget *buildMore();
    void updateInputPreview();
    void toggleRecording();
    void updateRecordingUi();
    void refreshRecordings();
    void selectRecording();
    void toggleRecordingPlayback();
    void readRecordingFrame();
    void stopRecordingPlayback();
    void exportRecording();
    void setupStyle();
    void connectWorker();
    void chooseVideo();
    void loadVideo(const QString &path);
    bool handlePreviewDrop(const QStringList &paths);
    bool previewDropLocked() const;
    void importDroppedImages(const QStringList &paths, bool append);
    void cancelDroppedImageImport();
    void chooseCamera();
    void importModel();
    void importLabels();
    void setModel(const QString &path, bool force = false);
    bool rememberConvertedModel(const QString &path, const QJsonObject &metadata);
    QJsonObject convertedModelProfile(const QString &path) const;
    bool applyConvertedModelProfile(const QString &path);
    void addFiles(const QStringList &files);
    void startInference();
    void startWorkbenchTask();
    void populateDatasetQueue(const QJsonObject &manifest, const std::function<void()> &completed);
    void stopInference();
    void onResult(const vision::InferenceResult &result, quint64 ticket = 0);
    void displayResult(const vision::InferenceResult &result);
    void selectQueueImage(int row);
    void cacheImageResult(const vision::InferenceResult &result);
    void clearImageResults();
    void saveImageResultInBackground(const vision::InferenceResult &result, quint64 ticket);
    void onFinished(bool cancelled);
    void setBusy(bool busy);
    void refreshModelLibrary();
    void showModelContextMenu(const QPoint &position);
    void openModelLocation(const QString &path);
    void editModelNote(const QString &path);
    void renameLibraryModel(const QString &path);
    void removeLibraryModel(const QString &path);
    QString modelDisplayName(const QString &path) const;
    void refreshHistory();
    void recordResult(const vision::InferenceResult &result, bool writeToDisk = true);
    void exportResult();
    void connectEvaluation();
    void loadDataset(const QString &yaml, const QString &split, const QString &metadata = {});
    void beginDatasetImport(const QString &source, const QString &split, const QString &metadata);
    void cancelPendingDatasetSessionRestore();
    void clearDatasetContext();
    void startEvaluation();
    void loadEvaluationReport(const QString &directory);
    void acceptEvaluationReport(const QJsonObject &report, const QString &directory, bool record);
    void reviewEvaluationImage(int row);
    void applyEvaluationFilter(const QString &filter);
    void showEvaluationErrors();
    void navigateEvaluationReview(int step);
    void updateEvaluationReviewUi();
    void resetEvaluationReviewNavigation();
    void applyEvaluationProfile(double threshold);
    void updateEvaluationModel();
    QString evaluationFingerprint() const;
    void attachGroundTruth(vision::InferenceResult &result) const;
    struct ResultExport
    {
        QString base;
        vision::InferenceResult result;
        QByteArray json, csv;
    };
    ResultExport prepareResultExport(const vision::InferenceResult &result, const QString &directory,
                                    const vision::ModelConfig &config);
    static bool writePreparedResult(const ResultExport &result, QString *error = nullptr);
    bool writeResult(const vision::InferenceResult &result, const QString &directory,
                     QString *error = nullptr);
    void showNotice(const QString &text, bool error = false);
    void persist();
    void selectRoute(int index);
    void updateSourceUi();
    void updateRangeUi();
    void updateClassFilterUi();
    void updateStartEligibility();
    QStringList selectedImageRange(QString *error = nullptr) const;
    QStringList selectedEvaluationRange(QString *error = nullptr) const;
    void inspectSelectedModel();
    QString inferenceFingerprint(const QString &modelIdentity = {}) const;
    QString modelFileIdentity() const;
    void invalidateConfiguredResults();
    void updateTaskUi();
    void updateDeviceUi();
    void checkGpuEnvironment();
    void prepareGpuEnvironment();
    void cancelGpuPreparation();
    void readGpuSetupOutput();
    void appendGpuLog(const QString &message);
    QString gpuLauncherPython() const;
    void updateModelMeta(const QString &state);
    void displayModelStructure();
    vision::ModelConfig currentConfig() const;
    QString projectRoot_, dataRoot_, modelPath_, labelsPath_, streamPath_, exportDir_, smokeDir_;
    QStringList files_, labels_, models_;
    QVariantMap modelDisplayNames_, modelNotes_, convertedModelProfiles_;
    QVariantMap modelEvaluationProfiles_;
    EvaluationPanel *evaluation_ = nullptr;
    QTabWidget *workbenchTabs_ = nullptr;
    QComboBox *evaluationRange_ = nullptr;
    QPushButton *evaluationStartButton_ = nullptr, *evaluationStopButton_ = nullptr;
    QLabel *evaluationTaskSummary_ = nullptr, *evaluationRangeHint_ = nullptr;
    QWidget *evaluationReviewBar_ = nullptr;
    QLabel *evaluationReviewSummaryLabel_ = nullptr;
    QComboBox *evaluationPreviewFilter_ = nullptr;
    QPushButton *evaluationReviewButton_ = nullptr, *evaluationReviewPreviousButton_ = nullptr,
                *evaluationReviewNextButton_ = nullptr;
    QVector<int> evaluationReviewRows_, evaluationErrorRows_;
    int evaluationEvaluatedCount_ = 0, evaluationFalsePositiveCount_ = 0,
        evaluationFalseNegativeCount_ = 0, evaluationNegativeFalsePositiveCount_ = 0;
    bool evaluationReviewPending_ = false, evaluationReviewAvailable_ = false,
        evaluationReviewSelectFirst_ = false;
    QString evaluationReviewError_;
    QTimer *queuePopulationTimer_ = nullptr;
    bool queuePopulating_ = false;
    bool droppedImageImportActive_ = false, droppedImageAppend_ = false, droppedImageQueueChanged_ = false;
    int droppedImageOriginalCount_ = 0, droppedImageOriginalRow_ = -1;
    quint64 droppedImageImportGeneration_ = 0;
    std::shared_ptr<std::atomic_bool> droppedImageImportCancellation_;
    quint64 queuePopulationGeneration_ = 0, imagePreviewGeneration_ = 0;
    AsyncImagePreview *imagePreviewLoader_ = nullptr;
    vision::EvaluationWorker *evaluationWorker_ = nullptr;
    QJsonObject datasetManifest_, evaluationReport_;
    QString evaluationDirectory_, evaluationRunFingerprint_, evaluationReviewFilter_ = "all";
    vision::ModelConfig evaluationRunConfig_;
    QHash<QString, int> evaluationSampleIndices_;
    QHash<QString, QJsonObject> datasetSamples_;
    bool loadingDataset_ = false, evaluationReviewActive_ = false;
    bool datasetSessionRestoreRequested_ = false, datasetSessionUserInteracted_ = false;
    quint64 datasetSessionRestoreGeneration_ = 0, datasetImportGeneration_ = 0;
    std::shared_ptr<std::atomic_bool> evaluationPreviewCancellation_, evaluationFilterCancellation_, datasetImportCancellation_;
    quint64 evaluationPreviewGeneration_ = 0, evaluationFilterGeneration_ = 0;
    QCheckBox *showGroundTruth_ = nullptr;
    QStringList nativeLabels_;
    vision::SourceKind sourceKind_ = vision::SourceKind::Images;
    vision::InferenceResult lastResult_;
    vision::ModelConfig lastConfig_;
    struct CachedImageResult
    {
        vision::InferenceResult result;
        vision::ModelConfig config;
        QString snapshotPath;
        QString sourceIdentity;
    };
    QHash<QString, CachedImageResult> imageResults_;
    std::shared_ptr<QTemporaryDir> imageResultDirectory_;
    quint64 imageResultGeneration_ = 0;
    int resultIoPending_ = 0;
    QThreadPool *resultIoPool_ = nullptr;
    bool imageResultCacheWarning_ = false;
    bool modelListReordering_ = false;
    QString lastError_;
    QString actualBackend_, actualDeviceName_, actualDeviceNotice_;
    vision::ComputeDevice actualDevice_ = vision::ComputeDevice::CPU;
    bool actualDeviceKnown_ = false;
    int actualDeviceIndex_ = -1;
    QJsonObject gpuEnvironment_;
    QByteArray gpuProbeOutput_, gpuProbeErrors_, gpuSetupOutput_;
    bool gpuProbeKnown_ = false, gpuSetupCancelling_ = false;
    QProcess *gpuProbeProcess_ = nullptr, *gpuSetupProcess_ = nullptr;
    QJsonArray history_;
    bool busy_ = false, closing_ = false, failed_ = false;
    bool recordingRequested_ = false, recordingActive_ = false, recordingStopping_ = false;
    bool inferenceStopping_ = false;
    QElapsedTimer recordingElapsed_;
    QTimer *recordingClock_ = nullptr, *playbackTimer_ = nullptr;
    std::unique_ptr<cv::VideoCapture> playbackCapture_;
    QString playbackPath_;
    QSize playbackContentSize_;
    int completed_ = 0, modelInputChannels_ = 0;
    int modelClassCount_ = 0, taskRangeFirst_ = 0, taskRangeLast_ = 0;
    bool initializing_ = true, modelInspectionPending_ = false;
    QString classFilterError_, modelInspectionError_, inspectedModelIdentity_, imageConfigFingerprint_;
    QStringList taskRangeFiles_;
    QString activeModelIdentity_, activeImageFingerprint_;
    QHash<QString, QString> taskSourceIdentities_;
    quint64 modelInspectionGeneration_ = 0;
    std::shared_ptr<std::atomic_bool> modelInspectionCancellation_;
    QSettings *settings_ = nullptr;
    QThread *workerThread_ = nullptr;
    vision::InferenceWorker *worker_ = nullptr;
    QStackedWidget *pages_ = nullptr;
    QList<QPushButton *> navButtons_;
    QLabel *pageTitle_ = nullptr, *pageSubtitle_ = nullptr, *statusLabel_ = nullptr, *modelName_ = nullptr,
           *modelMeta_ = nullptr, *sourceLabel_ = nullptr;
    QLabel *countMetric_ = nullptr, *latencyMetric_ = nullptr, *classMetric_ = nullptr,
           *sizeMetric_ = nullptr, *canvasTitle_ = nullptr, *zoomLabel_ = nullptr, *resultInfo_ = nullptr,
           *emptyResults_ = nullptr, *modelCount_ = nullptr;
    QLabel *backendBadge_ = nullptr, *backendFooter_ = nullptr;
    QLabel *deviceHint_ = nullptr, *gpuEnvironmentStatus_ = nullptr;
    QPushButton *gpuCheckButton_ = nullptr, *gpuPrepareButton_ = nullptr, *gpuCancelButton_ = nullptr;
    QPlainTextEdit *gpuLog_ = nullptr;
    QProgressBar *gpuSetupProgress_ = nullptr;
    QLabel *preprocessHint_ = nullptr, *meanRLabel_ = nullptr, *stereoLabel_ = nullptr;
    QPushButton *runButton_ = nullptr, *stopButton_ = nullptr, *exportButton_ = nullptr,
                *modelButton_ = nullptr, *labelButton_ = nullptr;
    QPushButton *recordButton_ = nullptr, *recordingPlayButton_ = nullptr, *exportRecordingButton_ = nullptr;
    QLabel *recordingStatus_ = nullptr, *recordingsCount_ = nullptr, *recordingDetails_ = nullptr;
    QListWidget *queue_ = nullptr, *modelList_ = nullptr;
    QTableWidget *predictionTable_ = nullptr, *historyTable_ = nullptr;
    QTableWidget *recordingsTable_ = nullptr;
    QComboBox *taskBox_ = nullptr, *inputColorMode_ = nullptr, *stereoView_ = nullptr;
    QComboBox *detectionRange_ = nullptr;
    QLineEdit *classFilter_ = nullptr;
    QWidget *detectionRangePanel_ = nullptr;
    QLabel *rangeHint_ = nullptr, *classFilterHint_ = nullptr;
    QComboBox *computeDevice_ = nullptr, *gpuDeviceIndex_ = nullptr;
    QLabel *gpuDeviceLabel_ = nullptr;
    QSpinBox *inputSize_ = nullptr, *cameraIndex_ = nullptr;
    QDoubleSpinBox *confidence_ = nullptr, *iou_ = nullptr, *scale_ = nullptr, *meanR_ = nullptr,
                   *meanG_ = nullptr, *meanB_ = nullptr;
    QCheckBox *autoExport_ = nullptr, *showBoxes_ = nullptr, *showLabels_ = nullptr;
    QProgressBar *progress_ = nullptr;
    ImageCanvas *canvas_ = nullptr;
    ImageCanvas *recordingCanvas_ = nullptr;
    ModelViewer *modelViewer_ = nullptr;
    ModelConversionPage *modelConversion_ = nullptr;
    QLabel *structureModelName_ = nullptr, *structureModelMeta_ = nullptr, *structureHint_ = nullptr,
           *structureStatus_ = nullptr;
    QList<QWidget *> lockedControls_;
};
