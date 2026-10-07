#include "core/visionengine.h"
#include <QColor>
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

    QString channelFixture(int channels, bool detection = false, const QString &metadata = {})
    {
        const QString path = temporary.path() + QString("/channels-%1-%2-%3.pt")
                                                  .arg(channels)
                                                  .arg(detection ? "detect" : "classify")
                                                  .arg(metadata.isEmpty() ? "inferred" : "metadata");
        const QString script = QStringLiteral(
            "import sys,json,torch\n"
            "channels=int(sys.argv[2]); detect=sys.argv[3]=='detect'\n"
            "class ChannelProbe(torch.nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__(); self.conv=torch.nn.Conv2d(channels,3,1,bias=False)\n"
            "        with torch.no_grad():\n"
            "            self.conv.weight.zero_()\n"
            "            if channels==3:\n"
            "                for i in range(3): self.conv.weight[i,i,0,0]=1.\n"
            "            else: self.conv.weight[1,0,0,0]=1.\n"
            "        boxes=torch.zeros(1,6,8); boxes[0,:4,0]=torch.tensor([32.,32.,32.,16.])\n"
            "        self.register_buffer('boxes',boxes)\n"
            "    def forward(self,x):\n"
            "        means=self.conv(x).mean(dim=(2,3))\n"
            "        if detect:\n"
            "            output=self.boxes.clone(); output[:,4,0]=means[:,1]; return output\n"
            "        return means*6.\n"
            "model=torch.jit.trace(ChannelProbe().eval(),torch.zeros(1,channels,64,64))\n"
            "info={'task':'detect' if detect else 'classify','names':['probe','other'] if detect else ['red','green','blue']}\n"
            "info.update(json.loads(sys.argv[4]))\n"
            "torch.jit.save(model,sys.argv[1],_extra_files={'config.txt':json.dumps(info)})\n");
        QProcess generator;
        generator.start(python, {"-c", script, path, QString::number(channels),
                                 detection ? "detect" : "classify", metadata.isEmpty() ? "{}" : metadata});
        if (!generator.waitForFinished(60000) || generator.exitStatus() != QProcess::NormalExit ||
            generator.exitCode() != 0)
            throw std::runtime_error(generator.readAllStandardError().constData());
        return path;
    }

    float classConfidence(const vision::InferenceResult &result, int classId) const
    {
        for (const auto &prediction : result.predictions)
            if (prediction.classId == classId)
                return prediction.confidence;
        return -1;
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
        QCOMPARE(engine.config().inputChannels, 3);
        QCOMPARE(engine.config().colorMode, vision::InputColorMode::Color);
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

    void realModernAndLegacyModelsAcceptGrayscaleWithoutChangingPreview()
    {
        const QImage original(imagePath);
        for (const QString &path : {modelPath, project + "/models/yolov5n.pt"})
        {
            auto value = config(path);
            value.colorMode = vision::InputColorMode::Grayscale;
            vision::VisionEngine engine;
            engine.load(value);
            QCOMPARE(engine.config().inputChannels, 3);
            QCOMPARE(engine.config().colorMode, vision::InputColorMode::Grayscale);
            const auto result = engine.infer(original, "real-grayscale");
            verifyDetection(result, original.size());
            QCOMPARE(result.image, original);
            const QImage resized = original.scaledToWidth(640, Qt::SmoothTransformation);
            const auto second = engine.infer(resized, "resized-grayscale");
            verifyDetection(second, resized.size());
            QCOMPARE(second.image, resized);
        }
    }

    void grayscaleThreeChannelsAreEqualAndUseBgrToGrayWeights()
    {
        const QString path = channelFixture(3);
        QImage original(128, 64, QImage::Format_RGB888);
        original.fill(QColor(240, 80, 20));
        auto value = config(path);
        value.inputSize = 64;
        vision::VisionEngine engine;
        engine.load(value);
        QCOMPARE(engine.config().inputChannels, 3);
        const auto color = engine.infer(original);
        QCOMPARE(color.predictions[0].classId, 0);
        QVERIFY(classConfidence(color, 0) > classConfidence(color, 1));
        QVERIFY(classConfidence(color, 1) > classConfidence(color, 2));
        value.colorMode = vision::InputColorMode::Grayscale;
        value.swapRB = false; // Gray planes are equal; channel exchange has no effect.
        engine.load(value);
        const auto gray = engine.infer(original);
        QCOMPARE(gray.image, original);
        for (int classId = 0; classId < 3; ++classId)
            QVERIFY(std::abs(classConfidence(gray, classId) - 1.0f / 3) < .0001f);
    }

    void grayscaleSingleChannelIsInferredFromTorchScriptConv()
    {
        const QString path = channelFixture(1);
        QImage original(128, 64, QImage::Format_RGB888);
        original.fill(QColor(240, 80, 20));
        auto value = config(path);
        value.inputSize = 64;
        vision::VisionEngine engine;
        const QString colorError = failureMessage(engine, value);
        QVERIFY2(colorError.contains("单通道") || colorError.contains("灰度"), qPrintable(colorError));
        QVERIFY(!engine.loaded());
        value.colorMode = vision::InputColorMode::Grayscale;
        engine.load(value);
        QCOMPARE(engine.config().inputChannels, 1);
        const auto result = engine.infer(original);
        QCOMPARE(result.image, original);
        QCOMPARE(result.predictions[0].classId, 1);
        // OpenCV rounds BGR2GRAY(20,80,240) to 121 before normalization.
        const double activation = 6.0 * 121 / 255;
        const double expected = std::exp(activation) / (2 + std::exp(activation));
        QVERIFY(std::abs(classConfidence(result, 1) - expected) < .0001);
    }

    void nativeClassificationSupportsOneAndThreeGrayChannels()
    {
        const QString one = temporary.path() + "/native-classification-1.pt";
        const QString three = temporary.path() + "/native-classification-3.pt";
        const QString script = QStringLiteral(
            "import sys,torch,copy\n"
            "from ultralytics.nn.tasks import ClassificationModel\n"
            "cfg={'nc':3,'depth_multiple':1.,'width_multiple':1.,'backbone':[[-1,1,'Conv',[8,3,1]]],'head':[[-1,1,'Classify',[3]]]}\n"
            "for channels,path in [(1,sys.argv[1]),(3,sys.argv[2])]:\n"
            "    model=ClassificationModel(copy.deepcopy(cfg),ch=channels,nc=3,verbose=False).eval()\n"
            "    model.names={0:'a',1:'b',2:'c'}\n"
            "    torch.save({'model':model,'train_args':{'task':'classify'}},path)\n");
        QProcess generator;
        generator.start(python, {"-c", script, one, three});
        QVERIFY(generator.waitForFinished(60000));
        QCOMPARE(generator.exitStatus(), QProcess::NormalExit);
        QVERIFY2(generator.exitCode() == 0, generator.readAllStandardError().constData());
        QImage image(32, 16, QImage::Format_RGB888);
        image.fill(QColor(240, 80, 20));
        for (const auto &pair : {qMakePair(one, 1), qMakePair(three, 3)})
        {
            auto value = config(pair.first);
            value.colorMode = vision::InputColorMode::Grayscale;
            value.inputSize = 16; // Classification is valid below the detector's stride.
            vision::VisionEngine engine;
            engine.load(value);
            QCOMPARE(engine.config().inputChannels, pair.second);
            QCOMPARE(engine.config().task, vision::ModelTask::Classification);
            const auto result = engine.infer(image);
            QCOMPARE(result.image, image);
            QCOMPARE(result.predictions.size(), 3);
            double sum = 0;
            for (const auto &prediction : result.predictions)
            {
                QVERIFY(prediction.box.isEmpty());
                QVERIFY(std::isfinite(prediction.confidence));
                sum += prediction.confidence;
            }
            QVERIFY(std::abs(sum - 1) < .001);
        }
    }

    void grayscaleTorchScriptDetectionRestoresLetterboxCoordinates()
    {
        QImage original(128, 64, QImage::Format_RGB888);
        original.fill(QColor(240, 80, 20));
        for (int channels : {1, 3})
        {
            auto value = config(channelFixture(channels, true));
            value.inputSize = 64;
            value.colorMode = vision::InputColorMode::Grayscale;
            vision::VisionEngine engine;
            engine.load(value);
            QCOMPARE(engine.config().inputChannels, channels);
            const auto result = engine.infer(original);
            QCOMPARE(result.image, original);
            QCOMPARE(result.predictions.size(), 1);
            const auto &prediction = result.predictions[0];
            QCOMPARE(prediction.classId, 0);
            QVERIFY(std::abs(prediction.confidence - (121.0 + 114) / (2 * 255)) < .0001);
            QCOMPARE(prediction.box, QRectF(32, 16, 64, 32));
        }
    }

    void realYoloSingleChannelStemsPreserveGrayscalePredictions()
    {
        const QString modern = temporary.path() + "/single-channel-modern.pt";
        const QString legacy = temporary.path() + "/single-channel-legacy.pt";
        const QString focus = temporary.path() + "/focus-legacy.pt";
        const QString script = QStringLiteral(
            "import sys,torch\n"
            "sys.path.insert(0,sys.argv[1]+'/vendor/yolov5')\n"
            "from models.common import Focus\n"
            "for source,target in [(sys.argv[2],sys.argv[4]),(sys.argv[3],sys.argv[5])]:\n"
            "    checkpoint=torch.load(source,map_location='cpu',weights_only=False)\n"
            "    model=checkpoint.get('ema') or checkpoint['model']; model.float().eval()\n"
            "    conv=next(m for m in model.modules() if isinstance(m,torch.nn.Conv2d))\n"
            "    conv.weight=torch.nn.Parameter(conv.weight.sum(dim=1,keepdim=True)); conv.in_channels=1\n"
            "    checkpoint['model']=model; checkpoint['ema']=None\n"
            "    torch.save(checkpoint,target)\n"
            "checkpoint=torch.load(sys.argv[3],map_location='cpu',weights_only=False)\n"
            "model=checkpoint.get('ema') or checkpoint['model']; model.float().eval(); old=model.model[0]\n"
            "stem=Focus(3,old.conv.out_channels,3); stem.eval(); stem.conv.bn.load_state_dict(old.bn.state_dict())\n"
            "with torch.no_grad():\n"
            "    for index,(y,x) in enumerate([(0,0),(1,0),(0,1),(1,1)]):\n"
            "        stem.conv.conv.weight[:,index*3:(index+1)*3]=old.conv.weight[:,:,y::2,x::2]\n"
            "for name in ('i','f','type','np'): setattr(stem,name,getattr(old,name))\n"
            "model.model[0]=stem; checkpoint['model']=model; checkpoint['ema']=None\n"
            "torch.save(checkpoint,sys.argv[6])\n");
        QProcess generator;
        generator.start(python, {"-c", script, project, modelPath, project + "/models/yolov5n.pt",
                                 modern, legacy, focus});
        QVERIFY(generator.waitForFinished(60000));
        QCOMPARE(generator.exitStatus(), QProcess::NormalExit);
        QVERIFY2(generator.exitCode() == 0, generator.readAllStandardError().constData());
        const QImage image(imagePath);
        for (const auto &pair : {qMakePair(modelPath, modern),
                                 qMakePair(project + "/models/yolov5n.pt", legacy)})
        {
            auto referenceConfig = config(pair.first);
            referenceConfig.colorMode = vision::InputColorMode::Grayscale;
            vision::VisionEngine reference, single;
            reference.load(referenceConfig);
            const auto expected = reference.infer(image);
            auto singleConfig = config(pair.second);
            const QString colorError = failureMessage(single, singleConfig);
            QVERIFY2(colorError.contains("单通道") || colorError.contains("灰度"), qPrintable(colorError));
            singleConfig.colorMode = vision::InputColorMode::Grayscale;
            single.load(singleConfig);
            QCOMPARE(single.config().inputChannels, 1);
            const auto actual = single.infer(image);
            verifyDetection(actual, image.size());
            QCOMPARE(actual.image, image);
            QCOMPARE(actual.predictions.size(), expected.predictions.size());
            for (qsizetype index = 0; index < actual.predictions.size(); ++index)
            {
                QCOMPARE(actual.predictions[index].classId, expected.predictions[index].classId);
                QVERIFY(std::abs(actual.predictions[index].confidence - expected.predictions[index].confidence) < .001);
                const QRectF a = actual.predictions[index].box, b = expected.predictions[index].box;
                QVERIFY(std::abs(a.x() - b.x()) < .1 && std::abs(a.y() - b.y()) < .1);
                QVERIFY(std::abs(a.width() - b.width()) < .1 && std::abs(a.height() - b.height()) < .1);
            }
        }
        auto focusConfig = config(focus);
        focusConfig.colorMode = vision::InputColorMode::Grayscale;
        vision::VisionEngine engine;
        engine.load(focusConfig);
        QCOMPARE(engine.config().inputChannels, 3);
        QCOMPARE(engine.config().task, vision::ModelTask::YoloV5);
        verifyDetection(engine.infer(image), image.size());
    }

    void invalidTorchScriptChannelMetadataIsRejected()
    {
        for (const QString &metadata : {QString("{\"input_channels\":3}"),
                                        QString("{\"input_channels\":4}")})
        {
            auto value = config(channelFixture(1, false, metadata));
            value.inputSize = 64;
            value.colorMode = vision::InputColorMode::Grayscale;
            vision::VisionEngine engine;
            const QString error = failureMessage(engine, value);
            QVERIFY2(error.contains("通道") || error.contains("input_channels"), qPrintable(error));
            QVERIFY(!engine.loaded());
        }
    }

    void detectionInputSizeMustBeMultipleOf32AtLoad()
    {
        for (const auto mode : {vision::InputColorMode::Color, vision::InputColorMode::Grayscale})
        {
            auto value = config();
            value.inputSize = 650;
            value.colorMode = mode;
            vision::VisionEngine engine;
            const QString error = failureMessage(engine, value);
            QVERIFY2(error.contains("32") && error.contains("尺寸"), qPrintable(error));
            QVERIFY(!engine.loaded());
        }
    }

    void torchScriptWithoutInputConstraintsRequiresActionableMetadata()
    {
        const QString path = temporary.path() + "/unconstrained-input.pt";
        const QString script = QStringLiteral(
            "import sys,json,torch\n"
            "class Unconstrained(torch.nn.Module):\n"
            "    def forward(self,x): return x.mean(dim=(1,2,3)).unsqueeze(1).repeat(1,3)\n"
            "model=torch.jit.trace(Unconstrained(),torch.zeros(1,3,64,64))\n"
            "torch.jit.save(model,sys.argv[1],_extra_files={'config.txt':json.dumps({'task':'classify','names':['a','b','c']})})\n");
        QProcess generator;
        generator.start(python, {"-c", script, path});
        QVERIFY(generator.waitForFinished(60000));
        QCOMPARE(generator.exitStatus(), QProcess::NormalExit);
        QVERIFY2(generator.exitCode() == 0, generator.readAllStandardError().constData());
        auto value = config(path);
        value.inputSize = 64;
        vision::VisionEngine engine;
        const QString error = failureMessage(engine, value);
        QVERIFY2(error.contains("config.txt") && error.contains("input_channels"), qPrintable(error));
        QVERIFY(!engine.loaded());
    }

    void processHandshakeCannotGuessMissingOrInvalidInputChannels()
    {
        for (const QByteArray &value : {QByteArray("None"), QByteArray("4"), QByteArray("True")})
        {
            const QByteArray script =
                "import sys,os,json,time\n"
                "path=os.path.realpath(sys.argv[sys.argv.index('--model')+1])\n"
                "print(json.dumps({'ok':True,'event':'ready','protocol':1,'task':'detect','layout':'v8',"
                "'labels':['probe'],'model_path':path,'input_channels':" + value +
                "}),flush=True)\ntime.sleep(30)\n";
            qputenv("VISION_STUDIO_PT_WORKER", helper("invalid-input-channels", script).toUtf8());
            vision::VisionEngine engine;
            const QString error = failureMessage(engine, config());
            QVERIFY2(error.contains("通道"), qPrintable(error));
            QVERIFY(!engine.loaded());
        }
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
            "metadata = {'task':'classify', 'names':['low','mid','high'], 'input_channels':3}\n"
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
                   "'labels':['probe'],'backend':'PyTorch fixture','model_path':model_path,'input_channels':3}), flush=True)\n"
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
