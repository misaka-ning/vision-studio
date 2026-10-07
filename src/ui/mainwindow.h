#pragma once
#include "core/visiontypes.h"
#include <QJsonArray>
#include <QMainWindow>
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
class ImageCanvas;
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
  signals:
    void startRequested(vision::JobRequest request);

  protected:
    void closeEvent(QCloseEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;

  private:
    QWidget *buildSidebar();
    QWidget *buildWorkbench();
    QWidget *buildModels();
    QWidget *buildHistory();
    QWidget *buildGuide();
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
    vision::ModelConfig currentConfig() const;
    QString projectRoot_, dataRoot_, modelPath_, labelsPath_, streamPath_, exportDir_, smokeDir_;
    QStringList files_, labels_, models_;
    QStringList nativeLabels_;
    vision::SourceKind sourceKind_ = vision::SourceKind::Images;
    vision::InferenceResult lastResult_;
    vision::ModelConfig lastConfig_;
    QString lastError_;
    QJsonArray history_;
    bool busy_ = false, closing_ = false, failed_ = false;
    int completed_ = 0;
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
    QPushButton *runButton_ = nullptr, *stopButton_ = nullptr, *exportButton_ = nullptr,
                *demoButton_ = nullptr, *modelButton_ = nullptr, *labelButton_ = nullptr;
    QListWidget *queue_ = nullptr, *modelList_ = nullptr;
    QTableWidget *predictionTable_ = nullptr, *historyTable_ = nullptr;
    QComboBox *taskBox_ = nullptr;
    QSpinBox *inputSize_ = nullptr, *cameraIndex_ = nullptr;
    QDoubleSpinBox *confidence_ = nullptr, *iou_ = nullptr, *scale_ = nullptr, *meanR_ = nullptr,
                   *meanG_ = nullptr, *meanB_ = nullptr;
    QCheckBox *swapRB_ = nullptr, *autoExport_ = nullptr, *showBoxes_ = nullptr, *showLabels_ = nullptr;
    QProgressBar *progress_ = nullptr;
    ImageCanvas *canvas_ = nullptr;
    QList<QWidget *> lockedControls_;
};
