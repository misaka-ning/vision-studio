#pragma once
#include "core/visiontypes.h"
#include "route.h"
#include "app/resultwriter.h"
#include "app/imageresultstore.h"
#include "app/detectioncontroller.h"
#include "app/evaluationcontroller.h"
#include "app/operationcoordinator.h"
#include "app/preferencesrepository.h"
#include "app/historyrepository.h"
#include "app/modelsessioncontroller.h"
#include "app/inputsessioncontroller.h"
#include "app/workbenchconfiguration.h"
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
class QTimer;
class QPlainTextEdit;
class ImageCanvas;
class ModelViewer;
class ModelConversionPage;
class EvaluationPanel;
class QTabWidget;
class ModelsLibraryPage;
class ModelDisplayPage;
class HistoryPage;
class RecordingsPage;
class MorePage;
class GuidePage;
namespace vision::app { class RecordingController; }
class WorkbenchPage;
class WorkspaceDropFilter;
namespace vision::app { class GpuEnvironmentService; }
class MainWindow : public QMainWindow
{
    Q_OBJECT
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
    void runModelLibraryDropSmoke(const QString &directory);
    void showModelStructure(const QString &model = {});
    void setComputeDevice(vision::ComputeDevice device, int index = 0);
    void restoreDatasetSession();
    QString datasetSessionFilePath() const;
    QJsonArray historySnapshot() const { return historyRepository_ ? historyRepository_->snapshot() : QJsonArray{}; }
    vision::app::EvaluationState evaluationState() const { return evaluationController_ ? evaluationController_->snapshot() : vision::app::EvaluationState{}; }
    const vision::app::ImageResultStore &imageResults() const { return detection_->results()->cache(); }
    vision::app::EvaluationController *evaluations() const { return evaluationController_; }
    vision::app::DetectionTaskSnapshot detectionState() const { return detection_->snapshot(); }
    int pendingResultWrites() const { return detection_ ? detection_->results()->pendingCount() : 0; }
    void requestExit(int code);
    void enableDiagnostics() { diagnosticRun_ = true; }
    vision::app::DetectionController *detections() const { return detection_; }
    bool isWorkbenchBusy() const;
    bool isClosing() const { return operationCoordinator_ && operationCoordinator_->isClosing(); }
    bool isShutdownReady() const { return operationCoordinator_ && operationCoordinator_->isShutdownReady(); }
    int completedCount() const;
    QStringList taskFiles() const;
    int taskFirst() const;
    int taskLast() const;
    bool isEvaluationBusy() const { return evaluationController_ && evaluationController_->isBusy(); }
    vision::app::ModelSessionSnapshot modelState() const { return modelSession_->snapshot(); }
    void setModelEvaluationProfile(const QString &path, const QVariantMap &profile)
    { modelSession_->setEvaluationProfile(path, profile); }
    vision::app::InputSessionSnapshot inputState() const;
    vision::app::WorkbenchDraft configurationState() const { return configuration_->snapshot(); }
    QString exportDirectory() const { return configurationState().exportDirectory; }
    vision::app::ClassFilterStatus classFilterStatus() const;
    vision::ModelConfig currentConfig() const;
    vision::ModelConfig displayedConfig() const;
    vision::InferenceResult displayedResult() const;
    vision::app::InputSessionController *inputs() const { return inputSession_; }
    vision::app::WorkbenchConfiguration *configuration() const { return configuration_; }
    vision::app::ModelSessionController *models() const { return modelSession_; }
    QString projectRoot() const { return projectRoot_; }
    QString dataRoot() const { return dataRoot_; }
    QString preferencesFilePath() const { return settings_->fileName(); }
    void selectModel(const QString &path) { setModel(path); }
    void openDataset(const QString &source, const QString &split, const QString &metadata = {})
    { loadDataset(source, split, metadata); }
    bool exportDisplayedResult(const QString &directory, QString *base = nullptr, QString *error = nullptr);
    void persistPreferences() { persist(); }
    void refreshModelPresentation() { refreshModelLibrary(); }
    void reviewEvaluationRow(int row) { reviewEvaluationImage(row); }
    void presentEvaluationReport(const QJsonObject &report, const QString &directory, bool record)
    { acceptEvaluationReport(report, directory, record); }
    QString inferenceFingerprint(const QString &modelIdentity = {}) const;
  signals:
    void startRequested(vision::JobRequest request);

  protected:
    void closeEvent(QCloseEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;

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
    void stopRecordingPlayback();
    void exportRecording();
    void setupStyle();
    void initializeOperationCoordinator();
    void connectWorker();
    void chooseVideo();
    void loadVideo(const QString &path);
    bool handlePreviewDrop(const QStringList &paths);
    bool previewDropLocked() const;
    void importDroppedImages(const QStringList &paths, bool append);
    void cancelDroppedImageImport();
    void chooseCamera();
    void importModel();
    bool importModelFiles(const QStringList &paths);
    bool validateModelImport(const QStringList &input, QStringList *paths, QString *error) const;
    void showModelImportHint(const QString &message, bool error = false);
    void importLabels();
    void setModel(const QString &path, bool force = false);
    void presentSelectedModel(bool convertedBgr = false);
    bool rememberConvertedModel(const QString &path, const QJsonObject &metadata);
    QJsonObject convertedModelProfile(const QString &path) const;
    bool applyConvertedModelProfile(const QString &path);
    void addFiles(const QStringList &files);
    void connectInputSession();
    void projectInputQueue(const QStringList &files, int firstRow, quint64 generation);
    void connectConfiguration();
    void projectConfigurationToView();
    void startInference();
    void startWorkbenchTask();
    void populateDatasetQueue(const QJsonObject &manifest, const std::function<void()> &completed);
    void stopInference();
    void onResult(const vision::InferenceResult &result, quint64 ticket = 0);
    void displayResult(const vision::InferenceResult &result);
    void selectQueueImage(int row);
    void clearImageResults();
    void onFinished(bool cancelled);
    void setBusy(bool busy);
    void refreshModelLibrary();
    void removeLibraryModel(const QString &path);
    QString modelDisplayName(const QString &path) const;
    void refreshHistory();
    void exportResult();
    void connectEvaluation();
    void loadDataset(const QString &yaml, const QString &split, const QString &metadata = {});
    void beginDatasetImport(const QString &source, const QString &split, const QString &metadata);
    void cancelPendingDatasetSessionRestore();
    void clearDatasetContext();
    void startEvaluation();
    void reanalyzeEvaluation(const QJsonObject &options);
    void importEvaluationReport(const QString &source, const QJsonObject &options);
    void showEvaluationSceneErrors(const QString &group);
    void loadEvaluationReport(const QString &directory, bool record = false);
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
    using ResultExport = vision::app::PreparedResultExport;
    ResultExport prepareResultExport(const vision::InferenceResult &result, const QString &directory,
                                    const vision::ModelConfig &config);
    static bool writePreparedResult(const ResultExport &result, QString *error = nullptr);
    bool writeResult(const vision::InferenceResult &result, const QString &directory,
                     QString *error = nullptr);
    void showNotice(const QString &text, bool error = false);
    void persist();
    void selectRoute(int index);
    void selectRoute(ui::Route route) { selectRoute(ui::routeIndex(route)); }
    void updateSourceUi();
    void updateRangeUi();
    void updateClassFilterUi();
    void updateStartEligibility();
    QStringList selectedImageRange(QString *error = nullptr) const;
    QStringList selectedEvaluationRange(QString *error = nullptr) const;
    void inspectSelectedModel();
    QString modelFileIdentity() const;
    void invalidateConfiguredResults();
    void updateTaskUi();
    void updateDeviceUi();
    void checkGpuEnvironment();
    void prepareGpuEnvironment();
    void cancelGpuPreparation();
    void appendGpuLog(const QString &message);
    void updateModelMeta(const QString &state);
    void displayModelStructure();
    QString projectRoot_, dataRoot_;
    vision::app::WorkbenchConfiguration *configuration_ = nullptr;
    vision::app::ModelSessionController *modelSession_ = nullptr;
    WorkbenchPage *workbenchPage_ = nullptr;
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
    vision::app::InputSessionController *inputSession_ = nullptr;
    vision::app::EvaluationController *evaluationController_ = nullptr;
    QCheckBox *showGroundTruth_ = nullptr;
    vision::app::GpuEnvironmentService *gpuService_ = nullptr;
    std::unique_ptr<vision::app::HistoryRepository> historyRepository_;
    vision::app::RecordingController *recordingsController_ = nullptr;
    ModelsLibraryPage *modelsLibraryPage_ = nullptr;
    ModelDisplayPage *modelDisplayPage_ = nullptr;
    HistoryPage *historyPage_ = nullptr;
    RecordingsPage *recordingsPage_ = nullptr;
    MorePage *morePage_ = nullptr;
    GuidePage *guidePage_ = nullptr;
    bool initializing_ = true;
    bool diagnosticRun_ = false;
    int requestedExitCode_ = 0;
    bool exitRequested_ = false;
    vision::app::PreferencesRepository *settings_ = nullptr;
    vision::app::DetectionController *detection_ = nullptr;
    vision::app::OperationCoordinator *operationCoordinator_ = nullptr;
    QStackedWidget *pages_ = nullptr;
    QWidget *workspaceHeader_ = nullptr;
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
    WorkspaceDropFilter *previewDropFilter_ = nullptr, *modelLibraryDropFilter_ = nullptr;
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
    QCheckBox *restoreDatasetOnStartup_ = nullptr;
    QProgressBar *progress_ = nullptr;
    ImageCanvas *canvas_ = nullptr;
    ImageCanvas *recordingCanvas_ = nullptr;
    ModelViewer *modelViewer_ = nullptr;
    ModelConversionPage *modelConversion_ = nullptr;
    QLabel *structureModelName_ = nullptr, *structureModelMeta_ = nullptr, *structureHint_ = nullptr,
           *structureStatus_ = nullptr;
    QList<QWidget *> lockedControls_;
};
