#include "ui/modelconversionpage.h"
#include <QApplication>
#include <QComboBox>
#include <QDir>
#include <QFile>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QTest>
#include <QTimer>
#include <memory>

class ConversionUiTests : public QObject
{
    Q_OBJECT
  private:
    std::unique_ptr<QTemporaryDir> temporary_;
    std::unique_ptr<ModelConversionPage> page_;
    QString source_, helper_;
    QByteArray previousHelper_, previousPython_;
    bool hadHelper_, hadPython_;
    template<typename T> T *control(const char *name) { return page_->findChild<T *>(name); }
    void write(const QString &path, const QByteArray &bytes) {
        QFile file(path); QVERIFY(file.open(QIODevice::WriteOnly)); QCOMPARE(file.write(bytes), bytes.size());
    }
    void worker(const QString &mode) {
        const QByteArray code = R"PY(import argparse,json,os,signal,sys,time
p=argparse.ArgumentParser()
p.add_argument('--source');p.add_argument('--format');p.add_argument('--output')
p.add_argument('--image-size');p.add_argument('--opset');p.add_argument('--work-dir')
a=p.parse_args()
def emit(**event): print(json.dumps(dict(protocol=1,**event)),flush=True)
def stop(*unused):
 emit(type='error',code='cancelled',message='已取消')
 sys.exit(130)
signal.signal(signal.SIGTERM,stop)
emit(type='status',message='prepared',job_temp=a.work_dir)
emit(type='progress',message='exporting',percent=25)
)PY" + ("mode=" + QString("'%1'\n").arg(mode)).toUtf8() + R"PY(
if mode=='redirect':
 emit(type='status',message='bad temp',job_temp=os.path.join(os.path.dirname(a.output),'.vision-studio-convert-unrelated'))
 time.sleep(60)
if mode=='wait': time.sleep(60)
time.sleep(0.15)
with open(a.output,'xb') as f: f.write(b'validated test artifact')
emit(type='result',message='done',output=a.output,format=a.format,input_channels=3)
if mode=='result_hang': time.sleep(60)
)PY";
        write(helper_, code);
    }
  private slots:
    void init() {
        temporary_ = std::make_unique<QTemporaryDir>(); QVERIFY(temporary_->isValid());
        source_ = temporary_->path() + "/first.pt";
        helper_ = temporary_->path() + "/worker.py";
        write(source_, "original checkpoint"); worker("success");
        hadHelper_=qEnvironmentVariableIsSet("VISION_STUDIO_CONVERSION_WORKER");
        hadPython_=qEnvironmentVariableIsSet("VISION_STUDIO_PYTHON");
        previousHelper_=qgetenv("VISION_STUDIO_CONVERSION_WORKER"); previousPython_=qgetenv("VISION_STUDIO_PYTHON");
        qputenv("VISION_STUDIO_CONVERSION_WORKER",helper_.toUtf8());
        qputenv("VISION_STUDIO_PYTHON","/usr/bin/python3.10");
        page_=std::make_unique<ModelConversionPage>(temporary_->path());
        page_->resize(1100,850); page_->show(); page_->setSourceModel(source_,"实验模型");
    }
    void cleanup() {
        if (page_->isBusy()) { page_->cancel(); QTRY_VERIFY_WITH_TIMEOUT(!page_->isBusy(),4000); }
        page_.reset();
        if(hadHelper_)qputenv("VISION_STUDIO_CONVERSION_WORKER",previousHelper_);else qunsetenv("VISION_STUDIO_CONVERSION_WORKER");
        if(hadPython_)qputenv("VISION_STUDIO_PYTHON",previousPython_);else qunsetenv("VISION_STUDIO_PYTHON");
        temporary_.reset();
    }
    void formatAndUnsupportedSource() {
        QCOMPARE(control<QLabel>("conversionSourceName")->text(),QString("实验模型"));
        QVERIFY(control<QPushButton>("conversionStartButton")->isEnabled());
        control<QComboBox>("conversionFormat")->setCurrentIndex(1);
        QVERIFY(!control<QComboBox>("conversionOpset")->isEnabled());
        QVERIFY(control<QLineEdit>("conversionOutputFilename")->text().endsWith(".torchscript"));
        const QString onnx=temporary_->path()+"/graph.onnx";write(onnx,"onnx");
        page_->setSourceModel(onnx);
        QVERIFY(!control<QPushButton>("conversionStartButton")->isEnabled());
        QVERIFY(control<QLabel>("conversionSupportHint")->text().contains("首版"));
    }
    void successKeepsJobSourceAndUsesLatestSelectionNextTime() {
        QSignalSpy busy(page_.get(),&ModelConversionPage::busyChanged);
        QSignalSpy imported(page_.get(),&ModelConversionPage::convertedModelReady);
        QTest::mouseClick(control<QPushButton>("conversionStartButton"),Qt::LeftButton);
        QVERIFY(page_->isBusy());
        const QString other=temporary_->path()+"/second.pt";write(other,"other original");
        page_->setSourceModel(other,"另一模型");
        QCOMPARE(control<QLabel>("conversionSourceName")->text(),QString("实验模型"));
        QTRY_VERIFY_WITH_TIMEOUT(!page_->isBusy(),5000);
        QCOMPARE(busy.size(),2);
        QCOMPARE(control<QLabel>("conversionStatus")->text(),QString("转换完成"));
        QCOMPARE(control<QLabel>("conversionSourceName")->text(),QString("另一模型"));
        QVERIFY(control<QLineEdit>("conversionOutputFilename")->text().startsWith("second-"));
        QTest::mouseClick(control<QPushButton>("conversionAddModelButton"),Qt::LeftButton);
        QCOMPARE(imported.size(),1);
        QCOMPARE(imported[0][0].toString(),temporary_->path()+"/converted-models/first-640.onnx");
        QFile original(source_);QVERIFY(original.open(QIODevice::ReadOnly));QCOMPARE(original.readAll(),QByteArray("original checkpoint"));
    }
    void existingOutputIsNeverOverwritten() {
        QDir().mkpath(temporary_->path()+"/converted-models");
        const QString output=temporary_->path()+"/converted-models/first-640.onnx"; write(output,"keep me");
        QSignalSpy busy(page_.get(),&ModelConversionPage::busyChanged);
        QTest::mouseClick(control<QPushButton>("conversionStartButton"),Qt::LeftButton);
        QVERIFY(!page_->isBusy()); QCOMPARE(busy.size(),0);
        QVERIFY(control<QLabel>("conversionSummary")->text().contains("已存在"));
        QFile file(output);QVERIFY(file.open(QIODevice::ReadOnly));QCOMPARE(file.readAll(),QByteArray("keep me"));
    }
    void cancelIsResponsiveAndKeepsUnrelatedDirectories() {
        worker("wait");
        const QString unrelated=temporary_->path()+"/converted-models/.vision-studio-convert-unrelated";
        QDir().mkpath(unrelated); write(unrelated+"/user-file","keep");
        int heartbeats=0; QTimer timer;timer.setInterval(10);connect(&timer,&QTimer::timeout,this,[&]{++heartbeats;});timer.start();
        QTest::mouseClick(control<QPushButton>("conversionStartButton"),Qt::LeftButton);
        QTRY_VERIFY(control<QLabel>("conversionStatus")->text()=="exporting");
        QTest::mouseClick(control<QPushButton>("conversionCancelButton"),Qt::LeftButton);
        QTRY_VERIFY_WITH_TIMEOUT(!page_->isBusy(),4000);
        QVERIFY(heartbeats>0);
        QCOMPARE(control<QLabel>("conversionStatus")->text(),QString("已取消转换"));
        QVERIFY(QFileInfo(unrelated+"/user-file").isFile());
        QVERIFY(!QFileInfo(temporary_->path()+"/converted-models/first-640.onnx").exists());
    }
    void completedArtifactCanStillCancelStuckWorker() {
        worker("result_hang");
        QTest::mouseClick(control<QPushButton>("conversionStartButton"),Qt::LeftButton);
        QTRY_VERIFY(control<QLabel>("conversionSummary")->text().contains("FP32"));
        QVERIFY(control<QPushButton>("conversionCancelButton")->isEnabled());
        QTest::mouseClick(control<QPushButton>("conversionCancelButton"),Qt::LeftButton);
        QTRY_VERIFY_WITH_TIMEOUT(!page_->isBusy(),4000);
        QVERIFY(control<QPushButton>("conversionAddModelButton")->isEnabled());
    }
    void workerCannotRedirectCleanupToAnotherDirectory() {
        worker("redirect");
        const QString unrelated=temporary_->path()+"/converted-models/.vision-studio-convert-unrelated";
        QDir().mkpath(unrelated);write(unrelated+"/keep","user data");
        QTest::mouseClick(control<QPushButton>("conversionStartButton"),Qt::LeftButton);
        QTRY_VERIFY_WITH_TIMEOUT(!page_->isBusy(),4000);
        QCOMPARE(control<QLabel>("conversionStatus")->text(),QString("转换失败"));
        QVERIFY(QFileInfo(unrelated+"/keep").isFile());
    }
};
QTEST_MAIN(ConversionUiTests)
#include "conversion_ui_tests.moc"
