#include "ui/modelviewer.h"

#include <QApplication>
#include <QDir>
#include <QEventLoop>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QPointer>
#include <QProcess>
#include <QRegularExpression>
#include <QSignalSpy>
#include <QStackedWidget>
#include <QTemporaryDir>
#include <QTest>
#include <QTimer>
#include <QVBoxLayout>
#include <QWebEnginePage>
#include <QWebEngineProfile>
#include <QWebEngineView>
#include <memory>

namespace
{
constexpr int deadline = 30000;

class EnvironmentOverride
{
  public:
    EnvironmentOverride(const char *name, const QByteArray &value)
        : m_name(name), m_wasSet(qEnvironmentVariableIsSet(name)), m_previous(qgetenv(name))
    {
        qputenv(name, value);
    }
    ~EnvironmentOverride()
    {
        if (m_wasSet)
            qputenv(m_name.constData(), m_previous);
        else
            qunsetenv(m_name.constData());
    }

    EnvironmentOverride(const EnvironmentOverride &) = delete;
    EnvironmentOverride &operator=(const EnvironmentOverride &) = delete;

  private:
    QByteArray m_name;
    bool m_wasSet;
    QByteArray m_previous;
};

QJsonObject runJson(QWebEngineView *browser, const QString &expression)
{
    auto result = std::make_shared<QJsonObject>();
    QEventLoop loop;
    QPointer<QEventLoop> guard(&loop);
    browser->page()->runJavaScript(
        expression,
        [result, guard](const QVariant &value)
        {
            *result = QJsonDocument::fromJson(value.toString().toUtf8()).object();
            if (guard)
                guard->quit();
        });
    QTimer::singleShot(3000, &loop, &QEventLoop::quit);
    loop.exec();
    return *result;
}

QJsonObject graphSnapshot(QWebEngineView *browser)
{
    return runJson(browser, QStringLiteral(R"JS(JSON.stringify((() => {
            const view = window.__view__;
            const graph = view && view.activeTarget;
            return { model: !!(view && view.model), active: !!graph,
                nodes: graph && Array.isArray(graph.nodes) ? graph.nodes.length : 0,
                rendered: !!document.querySelector('#target svg'),
                bodyDefault: document.body.classList.contains('default'),
                format: view && view.model ? String(view.model.format || '') : '' };
        })()))JS"));
}

void clickBrowserPoint(QWebEngineView *browser, const QJsonObject &position)
{
    const QPoint point(qRound(position.value("x").toDouble() * browser->zoomFactor()),
                       qRound(position.value("y").toDouble() * browser->zoomFactor()));
    QWidget *receiver = browser->focusProxy() ? browser->focusProxy() : browser;
    QTest::mouseClick(receiver, Qt::LeftButton, Qt::NoModifier, receiver->mapFrom(browser, point));
}

bool containsChinese(const QString &value)
{
    return value.contains(QRegularExpression(QStringLiteral("[\\x{4e00}-\\x{9fff}]")));
}
} // namespace

class ModelViewerTests : public QObject
{
    Q_OBJECT
  private slots:
    void initTestCase()
    {
        m_temporary = std::make_unique<QTemporaryDir>();
        QVERIFY(m_temporary->isValid());
        const QString project = QString::fromUtf8(VISION_PROJECT_DIR);
        const QString portablePython = project + "/runtime/bin/python";
        const QString defaultPython = QFileInfo(portablePython).isFile()
                                          ? portablePython
                                          : project + "/build/release-runtime/bin/python";
        const QString python = qEnvironmentVariable("VISION_STUDIO_PYTHON", defaultPython);
        QVERIFY2(QFileInfo(python).isExecutable(), qPrintable("Missing Netron runtime: " + python));
        const QString helper = project + "/scripts/netron_server.py";
        QVERIFY(QFileInfo(helper).isFile());
        m_python = std::make_unique<EnvironmentOverride>("VISION_STUDIO_PYTHON", python.toUtf8());
        m_helper = std::make_unique<EnvironmentOverride>("VISION_STUDIO_NETRON_WORKER", helper.toUtf8());

        QProcess probe;
        probe.start(python, {"-c", "import netron; print(netron.__version__)"});
        QVERIFY(probe.waitForFinished(5000));
        QCOMPARE(probe.exitStatus(), QProcess::NormalExit);
        QCOMPARE(probe.exitCode(), 0);
        QCOMPARE(probe.readAllStandardOutput().trimmed(), QByteArray("9.3.1"));

        const QString fixtures = m_temporary->path() + QStringLiteral("/中文 模型目录");
        QVERIFY(QDir().mkpath(fixtures));
        m_onnx = fixtures + QStringLiteral("/单目 检测.onnx");
        m_pt = fixtures + QStringLiteral("/YOLO 中文模型.pt");
        QVERIFY(QFile::copy(project + "/models/yolov5n.onnx", m_onnx));
        QVERIFY(QFile::copy(project + "/models/yolov8n.pt", m_pt));
        m_final = fixtures + QStringLiteral("/最终 模型.onnx");
        QVERIFY(QFile::copy(m_onnx, m_final));
        m_corrupt = fixtures + "/corrupt.onnx";
        m_unknown = fixtures + "/model.unsupported-vision-format";
        for (const QString &path : {m_corrupt, m_unknown})
        {
            QFile file(path);
            QVERIFY(file.open(QIODevice::WriteOnly));
            QVERIFY(file.write("not a neural network protobuf or checkpoint\n") > 0);
        }
    }

    void realModelsRenderGraphs_data()
    {
        QTest::addColumn<QString>("path");
        QTest::addColumn<QString>("format");
        QTest::newRow("onnx-structure") << m_onnx << QStringLiteral("ONNX");
        QTest::newRow("pt-module-hierarchy") << m_pt << QStringLiteral("PyTorch");
    }

    void realModelsRenderGraphs()
    {
        QFETCH(QString, path);
        QFETCH(QString, format);
        ModelViewer viewer;
        viewer.resize(1100, 760);
        viewer.show();
        QCOMPARE(viewer.state(), ModelViewer::State::Empty);
        QVERIFY(viewer.findChildren<QProcess *>().isEmpty());
        QVERIFY(viewer.findChildren<QWebEngineView *>().isEmpty());
        QSignalSpy loaded(&viewer, &ModelViewer::modelLoaded);
        QSignalSpy failed(&viewer, &ModelViewer::loadFailed);
        viewer.openModel(path);
        QTRY_VERIFY_WITH_TIMEOUT(viewer.state() != ModelViewer::State::Loading, deadline);
        QVERIFY2(viewer.state() == ModelViewer::State::Ready, qPrintable(viewer.errorString()));
        QCOMPARE(loaded.count(), 1);
        QCOMPARE(failed.count(), 0);
        QCOMPARE(loaded.first().first().toString(), QFileInfo(path).absoluteFilePath());
        QVERIFY(viewer.graphNodeCount() > 0);
        auto *browser = viewer.findChild<QWebEngineView *>("netronWebView");
        QVERIFY(browser);
        const QJsonObject graph = graphSnapshot(browser);
        QVERIFY(graph.value("model").toBool());
        QVERIFY(graph.value("active").toBool());
        QVERIFY(graph.value("rendered").toBool());
        QVERIFY(graph.value("bodyDefault").toBool());
        QVERIFY(graph.value("nodes").toInt() > 0);
        QCOMPARE(viewer.graphNodeCount(), graph.value("nodes").toInt());
        QVERIFY2(graph.value("format").toString().contains(format, Qt::CaseInsensitive),
                 qPrintable(graph.value("format").toString()));
        if (format == QStringLiteral("ONNX"))
        {
            // Hit-test an actual visible operator header, then send a native
            // Qt click to Chromium's render widget at those DOM coordinates.
            const QJsonObject node = runJson(browser, QStringLiteral(R"JS(JSON.stringify((() => {
                for (const header of document.querySelectorAll('#target .graph-node .node-item-type')) {
                    const rect = header.getBoundingClientRect();
                    const x = rect.left + rect.width / 2;
                    const y = rect.top + rect.height / 2;
                    if (x < 8 || x > innerWidth - 8 || y < 48 || y > innerHeight - 8) continue;
                    const hit = document.elementFromPoint(x, y);
                    if (!hit || hit.closest('.node-item-type') !== header) continue;
                    const container = header.closest('.graph-node');
                    const entry = Array.from(window.__view__.target.nodes.values())
                        .find(entry => entry.label && entry.label.element === container);
                    const value = entry && entry.label.value;
                    if (value && value.name && value.type) return { x, y,
                        name: value.name, type: value.type.identifier || value.type.name };
                }
                return {};
            })()))JS"));
            QVERIFY2(node.contains("x"), "No visible, unobstructed ONNX operator header was found");
            QVERIFY(!node.value("name").toString().isEmpty());
            QVERIFY(!node.value("type").toString().isEmpty());
            clickBrowserPoint(browser, node);
            auto sidebar = [browser]
            {
                return runJson(browser, QStringLiteral(R"JS(JSON.stringify((() => {
                    const sidebar = document.getElementById('sidebar');
                    const rect = sidebar.getBoundingClientRect();
                    return { visible: getComputedStyle(sidebar).opacity === '1' &&
                        rect.width > 0 && rect.left < innerWidth && rect.right > 0,
                        title: document.getElementById('sidebar-title').innerText,
                        text: document.getElementById('sidebar-content').innerText };
                })()))JS"));
            };
            QTRY_VERIFY_WITH_TIMEOUT(sidebar().value("visible").toBool(), 5000);
            const QJsonObject properties = sidebar();
            QCOMPARE(properties.value("title").toString().toCaseFolded(), QStringLiteral("node properties"));
            const QString text = properties.value("text").toString();
            QVERIFY2(text.contains(node.value("type").toString()), qPrintable(text));
            QVERIFY2(text.contains(node.value("name").toString()), qPrintable(text));
            QVERIFY2(text.contains(QStringLiteral("Inputs"), Qt::CaseInsensitive) &&
                         text.contains(QStringLiteral("Outputs"), Qt::CaseInsensitive),
                     qPrintable(text));

            const QJsonObject before = runJson(browser, QStringLiteral(R"JS(JSON.stringify((() => {
                const canvas = document.getElementById('canvas');
                const button = document.getElementById('zoom-in-button');
                const rect = button.getBoundingClientRect();
                const transform = canvas.getScreenCTM();
                const x = rect.left + rect.width / 2;
                const y = rect.top + rect.height / 2;
                const hit = document.elementFromPoint(x, y);
                window.__visionStudioTestClicks = [];
                document.addEventListener('click', event => {
                    window.__visionStudioTestClicks.push({ id: event.target.id,
                        button: event.target.closest('button')?.id || '',
                        x: event.clientX, y: event.clientY });
                }, { capture: true });
                return { x, y, rect: { left: rect.left, top: rect.top,
                        width: rect.width, height: rect.height },
                    hit: hit?.closest('button')?.id || '',
                    viewport: { width: innerWidth, height: innerHeight, dpr: devicePixelRatio },
                    scale: transform.a, zoom: window.__view__.target.zoom,
                    width: canvas.style.width };
            })()))JS"));
            QVERIFY(before.value("scale").toDouble() > 0);
            QVERIFY(before.value("zoom").toDouble() < 1.4);
            QCOMPARE(before.value("hit").toString(), QStringLiteral("zoom-in-button"));
            clickBrowserPoint(browser, before);
            // Netron scales the SVG through its CSS dimensions, so screen CTM
            // verifies the real rendering transform rather than a style label.
            QJsonObject after;
            const bool zoomed = QTest::qWaitFor(
                [&]
                {
                    after = runJson(browser, QStringLiteral(R"JS(JSON.stringify((() => {
                        const canvas = document.getElementById('canvas');
                        return { scale: canvas.getScreenCTM().a,
                            zoom: window.__view__.target.zoom, width: canvas.style.width,
                            clicks: window.__visionStudioTestClicks };
                    })()))JS"));
                    return after.value("zoom").toDouble() > before.value("zoom").toDouble();
                },
                3000);
            const QString diagnostic = QStringLiteral("before=%1 after=%2 browser=%3x%4 zoomFactor=%5")
                                           .arg(QString::fromUtf8(QJsonDocument(before).toJson(QJsonDocument::Compact)),
                                                QString::fromUtf8(QJsonDocument(after).toJson(QJsonDocument::Compact)))
                                           .arg(browser->width()).arg(browser->height()).arg(browser->zoomFactor());
            QVERIFY2(zoomed, qPrintable(diagnostic));
            QVERIFY(after.value("scale").toDouble() > before.value("scale").toDouble());
            QVERIFY(after.value("width").toString() != before.value("width").toString());
        }
    }

    void browserParsingFailuresAreChinese_data()
    {
        QTest::addColumn<QString>("path");
        QTest::newRow("unknown-format") << m_unknown;
        QTest::newRow("corrupt-onnx") << m_corrupt;
    }

    void browserParsingFailuresAreChinese()
    {
        QFETCH(QString, path);
        ModelViewer viewer;
        viewer.resize(1100, 760);
        viewer.show();
        QSignalSpy loaded(&viewer, &ModelViewer::modelLoaded);
        QSignalSpy failed(&viewer, &ModelViewer::loadFailed);
        viewer.openModel(path);
        QTRY_VERIFY_WITH_TIMEOUT(viewer.state() != ModelViewer::State::Loading, deadline);
        QCOMPARE(viewer.state(), ModelViewer::State::Error);
        QCOMPARE(loaded.count(), 0);
        QCOMPARE(failed.count(), 1);
        QVERIFY2(containsChinese(viewer.errorString()), qPrintable(viewer.errorString()));
        QVERIFY(viewer.errorString().contains(QStringLiteral("解析")) ||
                viewer.errorString().contains(QStringLiteral("格式")));
        QCOMPARE(viewer.graphNodeCount(), 0);
        QVERIFY(viewer.findChildren<QWebEngineView *>().isEmpty());
        QTRY_VERIFY_WITH_TIMEOUT(viewer.findChildren<QProcess *>().isEmpty(), 3000);
    }

    void rapidSwitchOnlyCompletesLatestModel()
    {
        ModelViewer viewer;
        viewer.resize(1100, 760);
        viewer.show();
        QSignalSpy loaded(&viewer, &ModelViewer::modelLoaded);
        QSignalSpy failed(&viewer, &ModelViewer::loadFailed);
        viewer.openModel(m_onnx);
        QTRY_VERIFY_WITH_TIMEOUT(viewer.findChild<QWebEngineView *>("netronWebView"), deadline);
        QCOMPARE(viewer.state(), ModelViewer::State::Loading);
        QPointer<QWebEngineView> oldBrowser = viewer.findChild<QWebEngineView *>("netronWebView");
        QPointer<QProcess> oldService = viewer.findChild<QProcess *>();
        QVERIFY(oldService && oldService->state() == QProcess::Running);

        viewer.openModel(m_pt);
        viewer.openModel(m_corrupt);
        viewer.openModel(m_final);
        QVERIFY(oldBrowser.isNull());
        QTRY_VERIFY_WITH_TIMEOUT(viewer.state() != ModelViewer::State::Loading, deadline);
        QVERIFY2(viewer.state() == ModelViewer::State::Ready, qPrintable(viewer.errorString()));
        QCOMPARE(viewer.modelPath(), QFileInfo(m_final).absoluteFilePath());
        QVERIFY(viewer.graphNodeCount() > 0);
        QCOMPARE(loaded.count(), 1);
        QCOMPARE(loaded.first().first().toString(), QFileInfo(m_final).absoluteFilePath());
        QCOMPARE(failed.count(), 0);
        QTRY_VERIFY_WITH_TIMEOUT(oldService.isNull(), 3000);
        // Give retired browser/process callbacks time to arrive; they cannot
        // overwrite the latest model's state or emit stale completion signals.
        QTest::qWait(600);
        QCOMPARE(viewer.state(), ModelViewer::State::Ready);
        QCOMPARE(loaded.count(), 1);
        QCOMPARE(failed.count(), 0);
    }

    void hiddenStackedPagePreloadsAndKeepsCachedGraph()
    {
        QStackedWidget pages;
        pages.resize(1100, 760);
        auto *workspace = new QWidget(&pages);
        auto *modelPage = new QWidget(&pages);
        auto *layout = new QVBoxLayout(modelPage);
        auto *viewer = new ModelViewer(modelPage);
        layout->addWidget(viewer);
        pages.addWidget(workspace);
        pages.addWidget(modelPage);
        pages.setCurrentWidget(workspace);
        pages.show();
        QSignalSpy loaded(viewer, &ModelViewer::modelLoaded);
        QSignalSpy failed(viewer, &ModelViewer::loadFailed);
        QVERIFY(!viewer->isVisible());
        viewer->openModel(m_onnx);
        QTRY_VERIFY_WITH_TIMEOUT(viewer->state() != ModelViewer::State::Loading, deadline);
        QVERIFY2(viewer->state() == ModelViewer::State::Ready, qPrintable(viewer->errorString()));
        QCOMPARE(pages.currentWidget(), workspace);
        QVERIFY(!viewer->isVisible());
        QVERIFY(viewer->graphNodeCount() > 0);
        QCOMPARE(loaded.count(), 1);
        QCOMPARE(failed.count(), 0);
        QPointer<QProcess> service = viewer->findChild<QProcess *>();
        QPointer<QWebEngineView> browser = viewer->findChild<QWebEngineView *>("netronWebView");
        QVERIFY(service && service->state() == QProcess::Running);
        QVERIFY(browser);
        const qint64 processId = service->processId();
        const int nodes = viewer->graphNodeCount();
        QSignalSpy pageLoads(browser->page(), &QWebEnginePage::loadStarted);
        const QJsonObject graph = graphSnapshot(browser);
        QVERIFY(graph.value("rendered").toBool());
        QVERIFY(graph.value("nodes").toInt() > 0);

        pages.setCurrentWidget(modelPage);
        QTest::qWait(300);
        QVERIFY(viewer->isVisible());
        pages.setCurrentWidget(workspace);
        QTest::qWait(100);
        pages.setCurrentWidget(modelPage);
        QTest::qWait(300);
        QCOMPARE(viewer->state(), ModelViewer::State::Ready);
        QCOMPARE(viewer->graphNodeCount(), nodes);
        QCOMPARE(viewer->findChild<QProcess *>(), service.data());
        QCOMPARE(viewer->findChild<QWebEngineView *>("netronWebView"), browser.data());
        QCOMPARE(service->processId(), processId);
        QCOMPARE(pageLoads.count(), 0);
        QCOMPARE(loaded.count(), 1);
        QCOMPARE(failed.count(), 0);
    }

    void stopDuringLoadingReleasesStartedBrowserAndService()
    {
        ModelViewer viewer;
        viewer.resize(1100, 760);
        viewer.show();
        QSignalSpy loaded(&viewer, &ModelViewer::modelLoaded);
        QSignalSpy failed(&viewer, &ModelViewer::loadFailed);
        viewer.openModel(m_onnx);
        QTRY_VERIFY_WITH_TIMEOUT(viewer.findChild<QWebEngineView *>("netronWebView"), deadline);
        QCOMPARE(viewer.state(), ModelViewer::State::Loading);
        QPointer<QWebEngineView> browser = viewer.findChild<QWebEngineView *>("netronWebView");
        QPointer<QWebEngineProfile> profile = browser->page()->profile();
        QPointer<QProcess> service = viewer.findChild<QProcess *>();
        QVERIFY(service && service->state() == QProcess::Running);
        viewer.stop();
        QCOMPARE(viewer.state(), ModelViewer::State::Empty);
        QCOMPARE(viewer.graphNodeCount(), 0);
        QVERIFY(viewer.errorString().isEmpty());
        QVERIFY(browser.isNull());
        QVERIFY(profile.isNull());
        QTRY_VERIFY_WITH_TIMEOUT(service.isNull(), 3000);
        QTest::qWait(400);
        QCOMPARE(viewer.state(), ModelViewer::State::Empty);
        QCOMPARE(loaded.count(), 0);
        QCOMPARE(failed.count(), 0);
    }

    void missingHelperOrRuntimeHasActionableChineseError()
    {
        ModelViewer viewer;
        {
            EnvironmentOverride missing("VISION_STUDIO_NETRON_WORKER",
                                        (m_temporary->path() + "/missing netron_server.py").toUtf8());
            viewer.openModel(m_onnx);
            QCOMPARE(viewer.state(), ModelViewer::State::Error);
            QVERIFY(viewer.errorString().contains("netron_server.py"));
            QVERIFY(containsChinese(viewer.errorString()));
            QVERIFY(viewer.findChildren<QProcess *>().isEmpty());
        }
        {
            EnvironmentOverride missing("VISION_STUDIO_PYTHON",
                                        (m_temporary->path() + "/missing python").toUtf8());
            viewer.openModel(m_onnx);
            QCOMPARE(viewer.state(), ModelViewer::State::Error);
            QVERIFY(viewer.errorString().contains("VISION_STUDIO_PYTHON"));
            QVERIFY(viewer.errorString().contains(QStringLiteral("运行环境")));
            QVERIFY(viewer.findChildren<QWebEngineView *>().isEmpty());
            QVERIFY(viewer.findChildren<QProcess *>().isEmpty());
        }
    }

    void cleanupTestCase()
    {
        m_helper.reset();
        m_python.reset();
        m_temporary.reset();
    }

  private:
    std::unique_ptr<QTemporaryDir> m_temporary;
    std::unique_ptr<EnvironmentOverride> m_python;
    std::unique_ptr<EnvironmentOverride> m_helper;
    QString m_onnx;
    QString m_pt;
    QString m_final;
    QString m_corrupt;
    QString m_unknown;
};

int main(int argc, char **argv)
{
    QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
    QApplication application(argc, argv);
    ModelViewerTests tests;
    return QTest::qExec(&tests, argc, argv);
}

#include "modelviewer_tests.moc"
