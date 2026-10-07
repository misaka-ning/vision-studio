#pragma once
#include "core/visiontypes.h"
#include <QElapsedTimer>
#include <QJsonArray>
#include <QJsonObject>
#include <QMainWindow>
#include <memory>
class QComboBox;
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
class QSettings;
class QTimer;
class QProcess;
class QPlainTextEdit;
class ImageCanvas;
class ModelViewer;
namespace cv
{
class VideoCapture;
}
namespace vision
{
class InferenceWorker;
}
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
    void showModelStructure(const QString &model = {});
    void setComputeDevice(vision::ComputeDevice device, int index = 0);
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
    void readRecordingFrame();
    void stopRecordingPlayback();
    void exportRecording();
    void setupStyle();
    void connectWorker();
    void chooseImages();
    void chooseFolder();
    void chooseVideo();
    void chooseCamera();
    void importModel();
    void importLabels();
    void setModel(const QString &path);
    void addFiles(const QStringList &files);
    void startInference();
    void stopInference();
    void onResult(const vision::InferenceResult &result);
    void onFinished(bool cancelled);
    void setBusy(bool busy);
    void refreshModelLibrary();
    void refreshHistory();
    void recordResult(const vision::InferenceResult &result);
    void exportResult();
    bool writeResult(const vision::InferenceResult &result, const QString &directory,
                     QString *error = nullptr);
    void showNotice(const QString &text, bool error = false);
    void persist();
    void selectRoute(int index);
    void updateSourceUi();
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
    QStringList nativeLabels_;
    vision::SourceKind sourceKind_ = vision::SourceKind::Images;
    vision::InferenceResult lastResult_;
    vision::ModelConfig lastConfig_;
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
                *demoButton_ = nullptr, *modelButton_ = nullptr, *labelButton_ = nullptr;
    QPushButton *recordButton_ = nullptr, *recordingPlayButton_ = nullptr, *exportRecordingButton_ = nullptr;
    QLabel *recordingStatus_ = nullptr, *recordingsCount_ = nullptr, *recordingDetails_ = nullptr;
    QListWidget *queue_ = nullptr, *modelList_ = nullptr;
    QTableWidget *predictionTable_ = nullptr, *historyTable_ = nullptr;
    QTableWidget *recordingsTable_ = nullptr;
    QComboBox *taskBox_ = nullptr, *inputColorMode_ = nullptr, *stereoView_ = nullptr;
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
    QLabel *structureModelName_ = nullptr, *structureModelMeta_ = nullptr, *structureHint_ = nullptr,
           *structureStatus_ = nullptr;
    QList<QWidget *> lockedControls_;
};
