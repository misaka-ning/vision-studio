#include "core/gpuruntime.h"
#include "core/visionengine.h"

#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcess>
#include <QScopeGuard>
#include <QTemporaryDir>
#include <QtTest>
#include <atomic>
#include <cmath>
#include <stdexcept>
#include <thread>

// This suite exercises a real NVIDIA GPU. It is never a mocked substitute for
// GPU acceptance, and intentionally skips unless explicitly enabled by QA.
class GpuBackendTests final : public QObject
{
    Q_OBJECT
    QTemporaryDir temporary;
    QString project, python;

    vision::ModelConfig config(const QString &path, vision::ComputeDevice device,
                               vision::InputColorMode color = vision::InputColorMode::Color) const
    {
        vision::ModelConfig value;
        value.modelPath = path;
        value.device = device;
        value.colorMode = color;
        value.task = path.contains("yolov5") ? vision::ModelTask::YoloV5 : vision::ModelTask::YoloV8;
        return value;
    }

    static double overlap(const QRectF &a, const QRectF &b)
    {
        const QRectF intersection = a.intersected(b);
        const double area = intersection.width() * intersection.height();
        return area / (a.width() * a.height() + b.width() * b.height() - area);
    }

    void comparable(const vision::InferenceResult &cpu, const vision::InferenceResult &gpu)
    {
        QCOMPARE(gpu.device, vision::ComputeDevice::CUDA);
        QCOMPARE(gpu.deviceIndex, 0);
        QVERIFY(!gpu.deviceName.isEmpty());
        QVERIFY(gpu.deviceNotice.isEmpty());
        QVERIFY(gpu.backend.contains("CUDA"));
        QVERIFY(gpu.inferenceMs > 0 && gpu.totalMs >= gpu.inferenceMs);
        QCOMPARE(gpu.image, cpu.image);
        QCOMPARE(gpu.originalImage, cpu.originalImage);
        QVERIFY(!gpu.predictions.isEmpty());
        // Match detections away from the threshold so floating-point rounding
        // around NMS/confidence boundaries does not require bitwise equality.
        for (const auto &expected : cpu.predictions)
        {
            if (expected.confidence < .30f)
                continue;
            bool matched = false;
            for (const auto &actual : gpu.predictions)
                if (actual.classId == expected.classId &&
                    std::abs(actual.confidence - expected.confidence) < .03 &&
                    (cpu.task == vision::ModelTask::Classification || overlap(actual.box, expected.box) > .95))
                {
                    matched = true;
                    break;
                }
            QVERIFY2(matched, qPrintable(QString("Missing CUDA match for class %1, confidence %2")
                                             .arg(expected.classId).arg(expected.confidence)));
        }
    }

    QString scriptFixture(int channels, bool detection)
    {
        const QString path = temporary.path() + QString("/gpu-%1-%2.pt").arg(channels).arg(detection);
        const QString code =
            "import sys,json,torch\n"
            "c=int(sys.argv[2]); detect=sys.argv[3]=='1'\n"
            "class Probe(torch.nn.Module):\n"
            " def __init__(self):\n"
            "  super().__init__(); self.conv=torch.nn.Conv2d(c,3,1,bias=False)\n"
            "  with torch.no_grad(): self.conv.weight.fill_(1./c)\n"
            "  boxes=torch.zeros(1,6,1); boxes[0,:4,0]=torch.tensor([32.,32.,16.,16.])\n"
            "  self.register_buffer('boxes',boxes)\n"
            "  self.register_buffer('multipliers',torch.tensor([[1.,2.,3.]]))\n"
            " def forward(self,x):\n"
            "  means=self.conv(x).mean(dim=(2,3))\n"
            "  if detect:\n"
            "   out=self.boxes.clone(); out[:,4,0]=.75+means[:,0]*.01; return out\n"
            "  return means*self.multipliers\n"
            "m=torch.jit.trace(Probe().eval(),torch.zeros(1,c,64,64))\n"
            "info={'task':'detect' if detect else 'classify','input_channels':c,'names':['probe','other'] if detect else ['a','b','c']}\n"
            "torch.jit.save(m,sys.argv[1],_extra_files={'config.txt':json.dumps(info)})\n";
        QProcess process;
        process.start(python, {"-B", "-c", code, path, QString::number(channels), detection ? "1" : "0"});
        if (!process.waitForFinished(60000) || process.exitCode() != 0)
            throw std::runtime_error(process.readAllStandardError().constData());
        return path;
    }

  private slots:
    void initTestCase()
    {
        if (qEnvironmentVariable("VISION_STUDIO_GPU_TESTS") != "1")
            QSKIP("Real CUDA acceptance requires VISION_STUDIO_GPU_TESTS=1 and a prepared NVIDIA runtime.");
        QVERIFY(temporary.isValid());
        project = QStringLiteral(VISION_PROJECT_DIR);
        const auto runtime = vision::gpuRuntimePaths();
        QVERIFY2(runtime.prepared, qPrintable(runtime.error));
        python = runtime.python;
    }

    void realModernAndLegacyPtColorAndGrayscale()
    {
        const QImage image(project + "/assets/bus.jpg");
        QVERIFY(!image.isNull());
        for (const QString &model : {QString("yolov8n.pt"), QString("yolov5n.pt")})
            for (auto color : {vision::InputColorMode::Color, vision::InputColorMode::Grayscale})
            {
                auto value = config(project + "/models/" + model, vision::ComputeDevice::CPU, color);
                vision::VisionEngine cpu, gpu;
                cpu.load(value);
                const auto expected = cpu.infer(image);
                value.device = vision::ComputeDevice::CUDA;
                gpu.load(value);
                QCOMPARE(gpu.config().resolvedDevice, vision::ComputeDevice::CUDA);
                comparable(expected, gpu.infer(image));
                // A second request must reuse the device session and maintain
                // geometry for a differently sized selected source ROI.
                const auto resized = image.scaledToWidth(640, Qt::SmoothTransformation);
                comparable(cpu.infer(resized), gpu.infer(resized));
            }
    }

    void torchScriptC1AndC3GrayscaleDetectionAndClassification()
    {
        QImage image(128, 64, QImage::Format_RGB888);
        image.fill(QColor(240, 80, 20));
        for (int channels : {1, 3})
            for (bool detection : {true, false})
            {
                const QString path = scriptFixture(channels, detection);
                auto value = config(path, vision::ComputeDevice::CPU, vision::InputColorMode::Grayscale);
                value.inputSize = 64;
                value.task = detection ? vision::ModelTask::YoloV8 : vision::ModelTask::Classification;
                vision::VisionEngine cpu, gpu;
                cpu.load(value);
                const auto expected = cpu.infer(image);
                value.device = vision::ComputeDevice::CUDA;
                gpu.load(value);
                QCOMPARE(gpu.config().inputChannels, channels);
                comparable(expected, gpu.infer(image));
            }
    }

    void nativeGrayC1C3Classification()
    {
        const QString one = temporary.path() + "/native-classification-1.pt";
        const QString three = temporary.path() + "/native-classification-3.pt";
        const QString code =
            "import sys,torch,copy\n"
            "from ultralytics.nn.tasks import ClassificationModel\n"
            "torch.manual_seed(123)\n"
            "cfg={'nc':3,'depth_multiple':1.,'width_multiple':1.,'backbone':[[-1,1,'Conv',[8,3,1]]],'head':[[-1,1,'Classify',[3]]]}\n"
            "for channels,path in [(1,sys.argv[1]),(3,sys.argv[2])]:\n"
            " m=ClassificationModel(copy.deepcopy(cfg),ch=channels,nc=3,verbose=False).eval()\n"
            " m.names={0:'a',1:'b',2:'c'}; torch.save({'model':m,'train_args':{'task':'classify'}},path)\n";
        QProcess generator;
        generator.start(python, {"-B", "-c", code, one, three});
        QVERIFY(generator.waitForFinished(60000));
        QVERIFY2(generator.exitCode() == 0, generator.readAllStandardError().constData());
        QImage image(128, 64, QImage::Format_RGB888);
        image.fill(QColor(240, 80, 20));
        for (const QString &path : {one, three})
        {
            auto value = config(path, vision::ComputeDevice::CPU, vision::InputColorMode::Grayscale);
            value.inputSize = 64;
            vision::VisionEngine cpu, gpu;
            cpu.load(value);
            const auto expected = cpu.infer(image);
            value.device = vision::ComputeDevice::CUDA;
            gpu.load(value);
            QCOMPARE(gpu.config().inputChannels, path == three ? 3 : 1);
            comparable(expected, gpu.infer(image));
        }
    }

    void modernAndLegacyRealSingleChannelStems()
    {
        const QString modern = temporary.path() + "/single-channel-modern.pt";
        const QString legacy = temporary.path() + "/single-channel-legacy.pt";
        QProcess generator;
        const QString code =
            "import sys,torch\n"
            "sys.path.insert(0,sys.argv[1]+'/vendor/yolov5')\n"
            "for source,target in [(sys.argv[2],sys.argv[4]),(sys.argv[3],sys.argv[5])]:\n"
            " ckpt=torch.load(source,map_location='cpu',weights_only=False)\n"
            " m=ckpt.get('ema') or ckpt['model']; m.float().eval()\n"
            " conv=next(v for v in m.modules() if isinstance(v,torch.nn.Conv2d))\n"
            " conv.weight=torch.nn.Parameter(conv.weight.sum(dim=1,keepdim=True)); conv.in_channels=1\n"
            " ckpt['model']=m; ckpt['ema']=None; torch.save(ckpt,target)\n";
        generator.start(python, {"-B", "-c", code, project, project + "/models/yolov8n.pt",
                                 project + "/models/yolov5n.pt", modern, legacy});
        QVERIFY(generator.waitForFinished(60000));
        QVERIFY2(generator.exitCode() == 0, generator.readAllStandardError().constData());
        const QImage image(project + "/assets/bus.jpg");
        for (const QString &path : {modern, legacy})
        {
            auto value = config(path, vision::ComputeDevice::CPU, vision::InputColorMode::Grayscale);
            vision::VisionEngine cpu, gpu;
            cpu.load(value);
            const auto expected = cpu.infer(image);
            value.device = vision::ComputeDevice::CUDA;
            gpu.load(value);
            QCOMPARE(gpu.config().inputChannels, 1);
            comparable(expected, gpu.infer(image));
        }
    }

    void realOnnxCudaMatchesCpuAndExecutesCudaNodes()
    {
        const QByteArray saved = qgetenv("VISION_STUDIO_ORT_PROFILE_PREFIX");
        const bool existed = qEnvironmentVariableIsSet("VISION_STUDIO_ORT_PROFILE_PREFIX");
        const auto restore = qScopeGuard([&] {
            if (existed) qputenv("VISION_STUDIO_ORT_PROFILE_PREFIX", saved);
            else qunsetenv("VISION_STUDIO_ORT_PROFILE_PREFIX");
        });
        const QImage image(project + "/assets/bus.jpg");
        for (auto color : {vision::InputColorMode::Color, vision::InputColorMode::Grayscale})
        {
            auto value = config(project + "/models/yolov5n.onnx", vision::ComputeDevice::CPU, color);
            vision::VisionEngine cpu, gpu;
            cpu.load(value);
            const auto expected = cpu.infer(image);
            value.device = vision::ComputeDevice::Auto;
            qputenv("VISION_STUDIO_ORT_PROFILE_PREFIX",
                    (temporary.path() + QString("/onnx-cuda-%1").arg(int(color))).toUtf8());
            gpu.load(value);
            comparable(expected, gpu.infer(image));
            QCOMPARE(gpu.config().device, vision::ComputeDevice::Auto);
            gpu.unload(); // Flush the ORT profile before reading actual nodes.
        }
        const QStringList profiles = QDir(temporary.path()).entryList({"onnx-cuda*.json"}, QDir::Files);
        QCOMPARE(profiles.size(), 2);
        for (const QString &name : profiles)
        {
            QFile profile(temporary.filePath(name));
            QVERIFY(profile.open(QIODevice::ReadOnly));
            const auto events = QJsonDocument::fromJson(profile.readAll()).array();
            bool cudaNode = false;
            for (const auto &event : events)
                cudaNode |= event.toObject().value("args").toObject().value("provider").toString() ==
                            "CUDAExecutionProvider";
            QVERIFY2(cudaNode, "No real CUDA execution-provider node was recorded in an ONNX profile.");
        }
    }

    void onnxC1AndC3GrayPixelsMatchModelInput()
    {
        // Encode tiny real Conv -> Pool -> Flatten ONNX graphs using Python's
        // standard library, without adding an ONNX exporter dependency.
        const QString code =
            "import sys,struct\n"
            "def v(n):\n"
            " out=b''\n"
            " while n>127: out+=bytes([(n&127)|128]); n>>=7\n"
            " return out+bytes([n])\n"
            "def i(f,n): return v(f<<3)+v(n)\n"
            "def b(f,d): return v((f<<3)|2)+v(len(d))+d\n"
            "def s(f,t): return b(f,t.encode())\n"
            "def info(n,d): return s(1,n)+b(2,b(1,i(1,1)+b(2,b''.join(b(1,i(1,x)) for x in d))))\n"
            "def tensor(n,d,a): return b''.join(i(1,x) for x in d)+i(2,1)+s(8,n)+b(9,struct.pack('<'+'f'*len(a),*a))\n"
            "def node(op,ins,out,attrs=b''): return b''.join(s(1,n) for n in ins)+s(2,out)+s(4,op)+attrs\n"
            "c=int(sys.argv[2]); weights=[0.]*c+[6./c]*c+[0.]*c\n"
            "attr=b(5,s(1,'kernel_shape')+i(8,1)+i(8,1)+i(20,7))\n"
            "g=b(1,node('Conv',['images','w','bias'],'conv',attr))+b(1,node('GlobalAveragePool',['conv'],'pool'))+b(1,node('Flatten',['pool'],'output'))\n"
            "g+=s(2,'gray_probe')+b(5,tensor('w',[3,c,1,1],weights))+b(5,tensor('bias',[3],[0.,0.,0.]))\n"
            "g+=b(11,info('images',[1,c,64,64]))+b(12,info('output',[1,3]))\n"
            "open(sys.argv[1],'wb').write(i(1,7)+s(2,'Vision Studio GPU tests')+b(7,g)+b(8,i(2,11)))\n";
        QImage image(64, 64, QImage::Format_RGB888);
        image.fill(QColor(240, 80, 20));
        for (int channels : {1, 3})
        {
            const QString path = temporary.path() + QString("/gray-%1.onnx").arg(channels);
            QProcess generator;
            generator.start(python, {"-B", "-c", code, path, QString::number(channels)});
            QVERIFY(generator.waitForFinished(60000));
            QVERIFY2(generator.exitCode() == 0, generator.readAllStandardError().constData());
            auto value = config(path, vision::ComputeDevice::CPU, vision::InputColorMode::Grayscale);
            value.task = vision::ModelTask::Classification;
            value.inputSize = 64;
            value.labels = {"red", "green", "blue"};
            vision::VisionEngine cpu, gpu;
            cpu.load(value);
            const auto expected = cpu.infer(image);
            value.device = vision::ComputeDevice::CUDA;
            gpu.load(value);
            QCOMPARE(gpu.config().inputChannels, channels);
            const auto actual = gpu.infer(image);
            comparable(expected, actual);
            QCOMPARE(actual.image.pixelColor(0, 0), QColor(121, 121, 121));
            QCOMPARE(actual.predictions[0].classId, 1);
            const double activation = std::exp(6.0 * 121 / 255);
            QVERIFY(std::abs(actual.predictions[0].confidence - activation / (2 + activation)) < .0001);
        }
    }

    void onnxCudaCancellationTerminatesRunAndSessionCanRecover()
    {
        std::atomic_bool cancelled{false};
        const auto owner = std::this_thread::get_id();
        vision::VisionEngine engine;
        engine.load(config(project + "/models/yolov5n.onnx", vision::ComputeDevice::CUDA));
        QCOMPARE(engine.config().resolvedDevice, vision::ComputeDevice::CUDA);
        engine.setCancellationCheck([&] {
            // Cancel precisely when the backend's watcher polls. No fixed
            // delay, artificial heavy network or unrelated process is used.
            if (std::this_thread::get_id() != owner)
                cancelled.store(true);
            return cancelled.load();
        });
        bool threw = false;
        try
        {
            engine.infer(QImage(project + "/assets/bus.jpg"));
        }
        catch (const std::exception &)
        {
            threw = true;
        }
        QVERIFY(threw);
        QVERIFY(cancelled.load());
        QVERIFY(engine.loaded());
        QCOMPARE(engine.config().resolvedDevice, vision::ComputeDevice::CUDA);
        QVERIFY(engine.backendName().contains("CUDA"));
        engine.setCancellationCheck({});
        const auto resumed = engine.infer(QImage(project + "/assets/bus.jpg"));
        QCOMPARE(resumed.device, vision::ComputeDevice::CUDA);
        QVERIFY(!resumed.predictions.isEmpty());
    }

    void ptCudaCancellationStopsChildAndCudaReloadWorks()
    {
        std::atomic_int polls{0};
        vision::VisionEngine engine;
        const auto value = config(project + "/models/yolov8n.pt", vision::ComputeDevice::CUDA);
        engine.load(value);
        QCOMPARE(engine.config().resolvedDevice, vision::ComputeDevice::CUDA);
        engine.setCancellationCheck([&] {
            // infer() checks before encoding, before sending, then in receive().
            // The third poll therefore cancels a submitted real CUDA request.
            return ++polls >= 3;
        });
        bool threw = false;
        try
        {
            engine.infer(QImage(project + "/assets/bus.jpg"));
        }
        catch (const std::exception &)
        {
            threw = true;
        }
        QVERIFY(threw);
        QVERIFY(polls.load() >= 3);
        QVERIFY(!engine.loaded());
        QCOMPARE(engine.config().resolvedDevice, vision::ComputeDevice::CUDA);
        engine.setCancellationCheck({});
        engine.load(value);
        const auto resumed = engine.infer(QImage(project + "/assets/bus.jpg"));
        QCOMPARE(resumed.device, vision::ComputeDevice::CUDA);
        QVERIFY(!resumed.predictions.isEmpty());
    }
};

QTEST_GUILESS_MAIN(GpuBackendTests)
#include "gpu_backend_tests.moc"
