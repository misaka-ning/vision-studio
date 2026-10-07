#include "core/visionengine.h"
#include <QCryptographicHash>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QProcess>
#include <QTemporaryDir>
#include <QtTest>
#include <atomic>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <thread>

class PtBackendTests final : public QObject
{
    Q_OBJECT
  private:
    struct SavedEnvironment
    {
        QByteArray name;
        QByteArray value;
        bool existed;
    };
    QList<SavedEnvironment> savedEnvironment;
    QTemporaryDir temporary;
    QString project, python, modelPath, imagePath;

    vision::ModelConfig config(const QString &path = {}) const
    {
        vision::ModelConfig value;
        value.modelPath = path.isEmpty() ? modelPath : path;
        value.task = vision::ModelTask::YoloV8;
        value.inputSize = 640;
        value.confidence = .25f;
        value.iou = .45f;
        return value;
    }

    QString failureMessage(vision::VisionEngine &engine, const vision::ModelConfig &value)
    {
        try
        {
            engine.load(value);
        }
        catch (const std::exception &error)
        {
            return QString::fromUtf8(error.what());
        }
        return {};
    }

    QString helper(const QString &name, const QByteArray &body)
    {
        const QString path = temporary.path() + "/" + name + ".py";
        QFile file(path);
        if (!file.open(QIODevice::WriteOnly) || file.write(body) != body.size())
            throw std::runtime_error("Cannot write process protocol fixture");
        return path;
    }

    QByteArray fileSha256(const QString &path)
    {
        QFile file(path);
        if (!file.open(QIODevice::ReadOnly))
            return {};
        return QCryptographicHash::hash(file.readAll(), QCryptographicHash::Sha256).toHex();
    }

    void verifyDetection(const vision::InferenceResult &result, const QSize &size)
    {
        QCOMPARE(result.image.size(), size);
        QVERIFY(!result.demonstration);
        QVERIFY(result.inferenceMs > 0);
        QVERIFY(result.totalMs >= result.inferenceMs);
        QVERIFY(!result.predictions.isEmpty());
        bool person = false, bus = false;
        for (const auto &prediction : result.predictions)
        {
            person = person || (prediction.classId == 0 && prediction.label == "person");
            bus = bus || (prediction.classId == 5 && prediction.label == "bus");
            QVERIFY(std::isfinite(prediction.confidence));
            QVERIFY(prediction.confidence >= .25f && prediction.confidence <= 1);
            QVERIFY(prediction.box.width() > 0 && prediction.box.height() > 0);
            QVERIFY(prediction.box.left() >= -.01 && prediction.box.top() >= -.01);
            QVERIFY(prediction.box.right() <= size.width() + .01);
            QVERIFY(prediction.box.bottom() <= size.height() + .01);
        }
        QVERIFY(person);
        QVERIFY(bus);
    }

  private slots:
    void initTestCase()
    {
        QVERIFY(temporary.isValid());
        project = QStringLiteral(VISION_PROJECT_DIR);
        for (const char *key : {"VISION_STUDIO_HOME", "VISION_STUDIO_PYTHON", "VISION_STUDIO_PT_WORKER",
                                "VISION_STUDIO_PT_LOAD_TIMEOUT_MS", "VISION_STUDIO_PT_INFER_TIMEOUT_MS"})
            savedEnvironment.append({QByteArray(key), qgetenv(key), qEnvironmentVariableIsSet(key)});
        qputenv("VISION_STUDIO_HOME", project.toUtf8());
        qunsetenv("VISION_STUDIO_PT_WORKER");
        python = qEnvironmentVariable("VISION_STUDIO_PYTHON");
        if (python.isEmpty())
        {
            const QString local = project + "/runtime/bin/python";
            python = QFileInfo(local).isExecutable() ? local : "/home/misaka/VisionStudio/runtime/bin/python";
        }
        QVERIFY2(QFileInfo(python).isExecutable(), qPrintable("Missing prepared Python runtime: " + python));
        qputenv("VISION_STUDIO_PYTHON", python.toUtf8());
        modelPath = project + "/models/yolov8n.pt";
        imagePath = project + "/assets/bus.jpg";
        QVERIFY2(QFileInfo(modelPath).isFile(), "The real official YOLOv8n .pt model is required.");
        QVERIFY(!QImage(imagePath).isNull());
    }

    void cleanupTestCase()
    {
        for (const auto &entry : savedEnvironment)
        {
            if (entry.existed)
                qputenv(entry.name.constData(), entry.value);
            else
                qunsetenv(entry.name.constData());
        }
    }

    void init()
    {
        qputenv("VISION_STUDIO_PYTHON", python.toUtf8());
        qunsetenv("VISION_STUDIO_PT_WORKER");
        qunsetenv("VISION_STUDIO_PT_LOAD_TIMEOUT_MS");
        qunsetenv("VISION_STUDIO_PT_INFER_TIMEOUT_MS");
    }

    void realCheckpointLoadsNamesAndProcessesConsecutiveImages()
    {
        vision::VisionEngine engine;
        auto value = config();
        // The Qt preprocessing spin box stores eight decimal places.
        value.scale = .00392157;
        engine.load(value);
        QVERIFY(engine.loaded());
        QVERIFY(engine.backendName().contains("PyTorch", Qt::CaseInsensitive));
        QCOMPARE(engine.config().labels.size(), 80);
        QCOMPARE(engine.config().labels[0], QString("person"));
        QCOMPARE(engine.config().labels[5], QString("bus"));
        const QImage original(imagePath);
        const auto first = engine.infer(original, "first-pt-request");
        verifyDetection(first, original.size());
        QCOMPARE(first.source, QString("first-pt-request"));
        const QImage secondImage =
            original.scaledToWidth(640, Qt::SmoothTransformation).convertToFormat(QImage::Format_RGBA8888);
        const auto second = engine.infer(secondImage, "second-pt-request");
        verifyDetection(second, secondImage.size());
        QCOMPARE(second.source, QString("second-pt-request"));
        QVERIFY(engine.loaded());
    }

    void missingExplicitPythonReportsFailure()
    {
        qputenv("VISION_STUDIO_PYTHON", (temporary.path() + "/missing-python").toUtf8());
        vision::VisionEngine engine;
        const QString error = failureMessage(engine, config());
        QVERIFY2(!error.isEmpty(),
                 "An invalid explicit interpreter must fail instead of silently falling back.");
        QVERIFY(error.contains("Python", Qt::CaseInsensitive) || error.contains("解释器"));
        QVERIFY(!engine.loaded());
    }

    void officialLegacyYoloV5CheckpointRunsWithBundledFramework()
    {
        const QString original = project + "/models/yolov5n.pt";
        const QByteArray expectedHash("649e089f59b78ac021025de035b2d9c45dc26e544ea252955d0ffcefc1099e2f");
        QCOMPARE(fileSha256(original), expectedHash);
        const QString isolated = temporary.path() + "/legacy-isolation";
        QVERIFY(QDir().mkpath(isolated));
        const QString legacy = isolated + "/yolov5n.pt";
        const QString decoy = isolated + "/yolov5nu.pt";
        const QString renamed = isolated + "/best.pt";
        QVERIFY(QFile::copy(original, legacy));
        // A modern checkpoint beside the selected legacy file must never replace it.
        QVERIFY(QFile::copy(modelPath, decoy));
        QVERIFY(QFile::copy(original, renamed));
        const QDir directory(isolated);
        const QStringList before = directory.entryList(QDir::Files, QDir::Name);
        vision::VisionEngine engine;
        const QImage image(imagePath);
        for (const QString &selected : {legacy, renamed})
        {
            engine.load(config(selected));
            QVERIFY(engine.backendName().contains("YOLOv5", Qt::CaseInsensitive));
            QCOMPARE(engine.config().task, vision::ModelTask::YoloV5);
            QCOMPARE(engine.config().labels.size(), 80);
            QCOMPARE(QFileInfo(engine.config().modelPath).canonicalFilePath(),
                     QFileInfo(selected).canonicalFilePath());
            verifyDetection(engine.infer(image, selected), image.size());
            QCOMPARE(fileSha256(selected), expectedHash);
        }
        QCOMPARE(directory.entryList(QDir::Files, QDir::Name), before);
        QCOMPARE(fileSha256(original), expectedHash);
    }

    void invalidCheckpointCannotLeavePreviousModelLoaded()
    {
        const QString invalid = temporary.path() + "/invalid.pt";
        QFile file(invalid);
        QVERIFY(file.open(QIODevice::WriteOnly));
        QCOMPARE(file.write("not a pytorch checkpoint"), qint64(24));
        file.close();
        vision::VisionEngine engine;
        engine.load(config());
        QVERIFY(engine.loaded());
        const QString error = failureMessage(engine, config(invalid));
        QVERIFY(!error.isEmpty());
        QVERIFY(!engine.loaded());
    }

    void bareStateDictionaryReportsMissingArchitecture()
    {
        const QString stateDictionary = temporary.path() + "/state-dictionary.pt";
        QProcess generator;
        generator.start(
            python,
            {"-c",
             "import sys, torch; torch.save({'state_dict': {'weight': torch.zeros(2, 3)}}, sys.argv[1])",
             stateDictionary});
        QVERIFY2(generator.waitForFinished(60000),
                 "The local runtime could not create a state_dict fixture.");
        QCOMPARE(generator.exitStatus(), QProcess::NormalExit);
        QVERIFY2(generator.exitCode() == 0, generator.readAllStandardError().constData());
        QVERIFY(QFileInfo(stateDictionary).isFile());
        vision::VisionEngine engine;
        const QString error = failureMessage(engine, config(stateDictionary));
        QVERIFY(!error.isEmpty());
        QVERIFY2(error.contains("state_dict", Qt::CaseInsensitive) || error.contains("架构") ||
                     error.contains("结构"),
                 qPrintable(error));
        QVERIFY(!engine.loaded());
    }

    void metadataBearingTorchScriptClassificationUsesTopK()
    {
        const QString scripted = temporary.path() + "/classification.pt";
        const QString script = QStringLiteral(
            "import sys, json, torch\n"
            "class Classifier(torch.nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__(); self.register_buffer('logits', torch.tensor([[-1., 0., 2.]]))\n"
            "    def forward(self, x):\n"
            "        return self.logits + x.mean(dim=(1,2,3)).unsqueeze(1) * 0\n"
            "model = torch.jit.trace(Classifier().eval(), torch.zeros(1,3,64,64))\n"
            "metadata = {'task':'classify', 'names':['low','mid','high']}\n"
            "torch.jit.save(model, sys.argv[1], _extra_files={'config.txt': json.dumps(metadata)})\n");
        QProcess generator;
        generator.start(python, {"-c", script, scripted});
        QVERIFY(generator.waitForFinished(60000));
        QCOMPARE(generator.exitStatus(), QProcess::NormalExit);
        QVERIFY2(generator.exitCode() == 0, generator.readAllStandardError().constData());
        auto value = config(scripted);
        value.inputSize = 64;
        vision::VisionEngine engine;
        engine.load(value);
        QCOMPARE(engine.config().task, vision::ModelTask::Classification);
        QCOMPARE(engine.config().labels, QStringList({"low", "mid", "high"}));
        const auto result = engine.infer(QImage(imagePath), "scripted-classification");
        QCOMPARE(result.task, vision::ModelTask::Classification);
        QCOMPARE(result.predictions.size(), 3);
        QCOMPARE(result.predictions[0].classId, 2);
        QCOMPARE(result.predictions[0].label, QString("high"));
        const double expected = std::exp(2.0) / (std::exp(-1.0) + 1.0 + std::exp(2.0));
        QVERIFY(std::abs(result.predictions[0].confidence - expected) < .002);
        for (const auto &prediction : result.predictions)
            QVERIFY(prediction.box.isEmpty());
    }

    void customOnnxPreprocessingCannotSilentlyChangePtInference()
    {
        auto value = config();
        value.meanR = 10;
        vision::VisionEngine engine;
        const QString error = failureMessage(engine, value);
        QVERIFY2(!error.isEmpty(), "The .pt backend must reject unsupported custom preprocessing.");
        QVERIFY(error.contains("预处理") || error.contains("均值"));
        QVERIFY(!engine.loaded());
    }

    void cancellationInterruptsBlockedModelInitialization()
    {
        const QString path = helper("blocked-load", "import time\ntime.sleep(30)\n");
        qputenv("VISION_STUDIO_PT_WORKER", path.toUtf8());
        std::atomic_bool cancelled{false};
        vision::VisionEngine engine;
        engine.setCancellationCheck([&cancelled] { return cancelled.load(); });
        std::thread trigger(
            [&cancelled]
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(300));
                cancelled.store(true);
            });
        QElapsedTimer timer;
        timer.start();
        const QString error = failureMessage(engine, config());
        trigger.join();
        QVERIFY2(timer.elapsed() < 2000, "Model initialization cancellation must promptly stop the helper.");
        QVERIFY(error.contains("取消") || error.contains("停止"));
        QVERIFY(!engine.loaded());
    }

    void cancellationInterruptsBlockedInference()
    {
        const QString path =
            helper("blocked-infer",
                   "import sys, time, os, json\n"
                   "model_path = os.path.realpath(sys.argv[sys.argv.index('--model') + 1])\n"
                   "print(json.dumps({'ok':True,'event':'ready','protocol':1,'task':'detect','layout':'v8',"
                   "'labels':['probe'],'backend':'PyTorch fixture','model_path':model_path}), flush=True)\n"
                   "sys.stdin.readline()\ntime.sleep(30)\n");
        qputenv("VISION_STUDIO_PT_WORKER", path.toUtf8());
        std::atomic_bool cancelled{false};
        vision::VisionEngine engine;
        engine.setCancellationCheck([&cancelled] { return cancelled.load(); });
        engine.load(config());
        QVERIFY(engine.loaded());
        std::thread trigger(
            [&cancelled]
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(300));
                cancelled.store(true);
            });
        QElapsedTimer timer;
        timer.start();
        QString error;
        try
        {
            engine.infer(QImage(imagePath), "cancelled-inference");
        }
        catch (const std::exception &exception)
        {
            error = QString::fromUtf8(exception.what());
        }
        trigger.join();
        QVERIFY2(timer.elapsed() < 2000, "Inference cancellation must promptly stop the helper.");
        QVERIFY(error.contains("取消") || error.contains("停止"));
        QVERIFY(!engine.loaded());
    }

    void invalidProcessProtocolResetsBackend()
    {
        const QString path =
            helper("invalid-protocol", "import time\nprint('not valid JSON', flush=True)\ntime.sleep(30)\n");
        qputenv("VISION_STUDIO_PT_WORKER", path.toUtf8());
        vision::VisionEngine engine;
        const QString error = failureMessage(engine, config());
        QVERIFY(error.contains("协议") || error.contains("JSON", Qt::CaseInsensitive));
        QVERIFY(!engine.loaded());
    }

    void substitutedModelHandshakeIsRejected()
    {
        const QString path =
            helper("substituted-model",
                   "import sys, time, os, json\n"
                   "selected = os.path.realpath(sys.argv[sys.argv.index('--model') + 1])\n"
                   "other = os.path.join(os.path.dirname(selected), 'yolov5n.pt')\n"
                   "print(json.dumps({'ok':True,'event':'ready','protocol':1,'task':'detect','layout':'v8',"
                   "'labels':['probe'],'backend':'PyTorch fixture','model_path':other}), flush=True)\n"
                   "time.sleep(30)\n");
        qputenv("VISION_STUDIO_PT_WORKER", path.toUtf8());
        vision::VisionEngine engine;
        const QString error = failureMessage(engine, config());
        QVERIFY2(error.contains("不一致") || error.contains("所选模型"), qPrintable(error));
        QVERIFY(!engine.loaded());
    }

    void shortenedInitializationDeadlineStopsBackend()
    {
        const QString path = helper("load-timeout", "import time\ntime.sleep(30)\n");
        qputenv("VISION_STUDIO_PT_WORKER", path.toUtf8());
        qputenv("VISION_STUDIO_PT_LOAD_TIMEOUT_MS", "100");
        vision::VisionEngine engine;
        QElapsedTimer timer;
        timer.start();
        const QString error = failureMessage(engine, config());
        QVERIFY2(timer.elapsed() < 2000, "The shorter deadline must stop a stalled model initialization.");
        QVERIFY(error.contains("超时"));
        QVERIFY(!engine.loaded());
    }
};

QTEST_GUILESS_MAIN(PtBackendTests)
#include "pt_backend_tests.moc"
