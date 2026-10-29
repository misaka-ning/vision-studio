// SPDX-License-Identifier: AGPL-3.0-only
#include "app/gpuenvironmentservice.h"

#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QPointer>
#include <QProcess>
#include <QSignalSpy>
#include <QStandardPaths>
#include <QTemporaryDir>
#include <QTimer>
#include <QtTest>

using vision::app::GpuEnvironmentService;

// Real, isolated helper processes test lifecycle/protocol handling. These are
// not CUDA acceptance tests and do not install or change any runtime.
class GpuEnvironmentServiceTests final : public QObject
{
    Q_OBJECT
    QTemporaryDir directory_;
    QString python_;
    int scriptNumber_ = 0;

    QString script(const QByteArray &body)
    {
        const auto path = directory_.filePath(QString("helper-%1.py").arg(++scriptNumber_));
        QFile file(path);
        if (!file.open(QIODevice::WriteOnly) || file.write(body) != body.size()) return {};
        return path;
    }

    GpuEnvironmentService::Options options(const QByteArray &probe, const QByteArray &setup = "pass\n")
    {
        GpuEnvironmentService::Options value;
        value.projectRoot = directory_.path();
        value.launcherPython = python_;
        value.runtimeDirectory = directory_.filePath("never-created-runtime");
        value.probeScript = script(probe);
        value.setupScript = script(setup);
        value.probeTimeoutMs = 3000;
        value.preparationKillDelayMs = 80;
        value.shutdownKillDelayMs = 80;
        return value;
    }

    static QStringList logs(const QSignalSpy &spy)
    {
        QStringList result;
        for (const auto &entry : spy) result.append(entry[0].toString());
        return result;
    }

  private slots:
    void initTestCase()
    {
        QVERIFY(directory_.isValid());
        python_ = QStandardPaths::findExecutable("python3");
        if (python_.isEmpty()) python_ = QStandardPaths::findExecutable("python");
        QVERIFY2(!python_.isEmpty(), "A Python interpreter is required for isolated helper fixtures.");
    }

    void probePreservesPayloadArgumentsAndBusyGates()
    {
        const auto value = options(
            "import json,os,sys,time\n"
            "time.sleep(.05)\n"
            "print(json.dumps({'ok':True,'prepared':True,'cuda_available':True,"
            "'devices':[{'index':2,'name':'Fixture GPU','total_memory_mb':8192}],"
            "'argv':sys.argv[1:],'dont_write':os.environ.get('PYTHONDONTWRITEBYTECODE')}))\n");
        GpuEnvironmentService service(value);
        QSignalSpy changed(&service, &GpuEnvironmentService::environmentChanged);
        QSignalSpy busy(&service, &GpuEnvironmentService::busyChanged);
        QSignalSpy shutdown(&service, &GpuEnvironmentService::shutdownReady);
        QVERIFY(!service.probeKnown());
        QVERIFY(service.checkEnvironment());
        QVERIFY(service.isChecking());
        QVERIFY(!service.prepareEnvironment());
        QVERIFY(!service.checkEnvironment());
        QTRY_COMPARE_WITH_TIMEOUT(changed.size(), 1, 5000);
        QVERIFY(!service.isBusy()); QVERIFY(service.probeKnown());
        const auto environment = service.environment();
        QVERIFY(environment.value("cuda_available").toBool());
        QCOMPARE(environment.value("argv").toArray(), QJsonArray({"--runtime-dir", value.runtimeDirectory}));
        QCOMPARE(environment.value("dont_write").toString(), QString("1"));
        QVERIFY(service.status().contains("GPU 2 · Fixture GPU · 8.0 GB"));
        QCOMPARE(busy.size(), 2); QCOMPARE(busy[0][0].toBool(), true); QCOMPARE(busy[1][0].toBool(), false);
        service.requestShutdown();
        QTRY_COMPARE_WITH_TIMEOUT(shutdown.size(), 1, 2000);
        QVERIFY(service.isShutdownReady());
        service.requestShutdown(); QTest::qWait(20); QCOMPARE(shutdown.size(), 1);
        QVERIFY(!QFileInfo::exists(value.runtimeDirectory));
    }

    void setupParsesChunkedJsonLogsProgressAndAutomaticRecheck()
    {
        auto value = options(
            "import json\nprint(json.dumps({'ok':True,'prepared':True,'cuda_available':False,'devices':[]}))\n");
        // Build setup separately so its exact runtime argument is checked too.
        const QByteArray setup =
            "import json,os,sys,time\n"
            "assert sys.argv[1]=='--runtime-dir' and sys.argv[2].endswith('never-created-runtime')\n"
            "assert os.environ.get('PYTHONUNBUFFERED')=='1'\n"
            "assert os.environ.get('PYTHONDONTWRITEBYTECODE')=='1'\n"
            "sys.stdout.write('{\"message\":\"stage one\",'); sys.stdout.flush(); time.sleep(.02)\n"
            "print('\"percent\":37}')\n"
            "print(json.dumps({'event':'fallback event','progress':101}))\n"
            "print(json.dumps({'message':'stage two','progress':-4}))\n"
            "print('raw log')\n"
            "print('stderr log',file=sys.stderr)\n"
            "print('x'*17000)\n"
            "sys.stdout.write('final tail')\n";
        value.setupScript = script(setup);
        GpuEnvironmentService service(value);
        QSignalSpy log(&service, &GpuEnvironmentService::logAppended);
        QSignalSpy reset(&service, &GpuEnvironmentService::logReset);
        QSignalSpy progress(&service, &GpuEnvironmentService::progressChanged);
        QSignalSpy completed(&service, &GpuEnvironmentService::preparationFinished);
        QSignalSpy probe(&service, &GpuEnvironmentService::environmentChanged);
        QSignalSpy shutdown(&service, &GpuEnvironmentService::shutdownReady);
        QVERIFY(service.prepareEnvironment());
        QVERIFY(service.isPreparing()); QVERIFY(!service.checkEnvironment());
        QTRY_COMPARE_WITH_TIMEOUT(completed.size(), 1, 5000);
        QCOMPARE(completed[0][0].toBool(), true); QCOMPARE(completed[0][1].toBool(), false);
        QTRY_COMPARE_WITH_TIMEOUT(probe.size(), 1, 5000);
        QCOMPARE(reset.size(), 1);
        const auto messages = logs(log);
        for (const auto &message : messages) QVERIFY(message.size() <= 16000);
        QVERIFY(messages.contains("stage one")); QVERIFY(messages.contains("fallback event"));
        QVERIFY(messages.contains("stage two")); QVERIFY(messages.contains("raw log"));
        QVERIFY(messages.contains("stderr log")); QVERIFY(messages.contains("final tail"));
        QVector<int> percentages;
        for (const auto &event : progress)
            if (event[1].toInt() == 100 && event[3].toBool()) percentages.append(event[2].toInt());
        QCOMPARE(percentages, QVector<int>({37, 100, 0}));
        QVERIFY(!progress.last()[3].toBool());
        service.requestShutdown(); QTRY_COMPARE_WITH_TIMEOUT(shutdown.size(), 1, 2000);
    }

    void missingToolsRemainRecoverableWithoutProcesses()
    {
        auto value = options("pass\n");
        value.probeScript = directory_.filePath("missing-probe.py");
        value.setupScript = directory_.filePath("missing-setup.py");
        GpuEnvironmentService service(value);
        QSignalSpy changed(&service, &GpuEnvironmentService::environmentChanged);
        QSignalSpy shutdown(&service, &GpuEnvironmentService::shutdownReady);
        QVERIFY(!service.checkEnvironment()); QVERIFY(service.probeKnown());
        QCOMPARE(changed.size(), 1); QVERIFY(!service.environment().value("ok").toBool());
        QVERIFY(service.status().contains("CPU 仍可使用"));
        QVERIFY(!service.prepareEnvironment()); QVERIFY(!service.isBusy());
        QVERIFY(service.findChildren<QProcess *>().isEmpty());
        service.requestShutdown(); QCOMPARE(shutdown.size(), 0);
        QTRY_COMPARE_WITH_TIMEOUT(shutdown.size(), 1, 1000);
        QVERIFY(!service.checkEnvironment()); QVERIFY(!service.prepareEnvironment());
    }

    void failedLaunchRetiresOnceAndCompletesShutdown()
    {
        auto value = options("pass\n");
        value.launcherPython = directory_.filePath("missing-python");
        GpuEnvironmentService service(value);
        QSignalSpy changed(&service, &GpuEnvironmentService::environmentChanged);
        QSignalSpy shutdown(&service, &GpuEnvironmentService::shutdownReady);
        QVERIFY(service.checkEnvironment());
        QTRY_COMPARE_WITH_TIMEOUT(changed.size(), 1, 2000);
        QVERIFY(!service.isBusy()); QVERIFY(!service.environment().value("ok").toBool());
        service.requestShutdown(); QTRY_COMPARE_WITH_TIMEOUT(shutdown.size(), 1, 2000);
        QTest::qWait(30); QCOMPARE(changed.size(), 1); QCOMPARE(shutdown.size(), 1);
    }

    void probeTimeoutKeepsEventLoopResponsive()
    {
        auto value = options("import time\ntime.sleep(30)\n");
        value.probeTimeoutMs = 120;
        GpuEnvironmentService service(value);
        QSignalSpy changed(&service, &GpuEnvironmentService::environmentChanged);
        QSignalSpy shutdown(&service, &GpuEnvironmentService::shutdownReady);
        int ticks = 0; QTimer timer;
        connect(&timer, &QTimer::timeout, this, [&] { ++ticks; }); timer.start(5);
        QVERIFY(service.checkEnvironment());
        QTRY_COMPARE_WITH_TIMEOUT(changed.size(), 1, 3000);
        QVERIFY(ticks >= 3);
        QVERIFY(!service.environment().value("ok").toBool());
        QVERIFY(service.environment().value("reason").toString().contains("超时"));
        service.requestShutdown(); QTRY_COMPARE_WITH_TIMEOUT(shutdown.size(), 1, 2000);
    }

    void oversizedProbeOutputIsNotAcceptedAsEnvironment()
    {
        const auto value = options("import sys,time\nsys.stdout.write('x'*600000);sys.stdout.flush()\ntime.sleep(30)\n");
        GpuEnvironmentService service(value);
        QSignalSpy changed(&service, &GpuEnvironmentService::environmentChanged);
        QSignalSpy shutdown(&service, &GpuEnvironmentService::shutdownReady);
        QVERIFY(service.checkEnvironment());
        QTRY_COMPARE_WITH_TIMEOUT(changed.size(), 1, 3000);
        QVERIFY(!service.environment().value("ok").toBool());
        QVERIFY(!service.environment().contains("cuda_available"));
        service.requestShutdown(); QTRY_COMPARE_WITH_TIMEOUT(shutdown.size(), 1, 2000);
    }

    void largePartialSetupOutputStaysBoundedAndFailureDoesNotRecheck()
    {
        const auto value = options("print('{\"cuda_available\":false}')\n",
            "import sys,time\nsys.stdout.write('x'*300000);sys.stdout.flush();time.sleep(.05)\n"
            "print('\\n')\nprint('{\"message\":\"after large output\",\"percent\":42}')\nsys.exit(2)\n");
        GpuEnvironmentService service(value);
        QSignalSpy log(&service, &GpuEnvironmentService::logAppended);
        QSignalSpy completed(&service, &GpuEnvironmentService::preparationFinished);
        QSignalSpy probe(&service, &GpuEnvironmentService::environmentChanged);
        QSignalSpy shutdown(&service, &GpuEnvironmentService::shutdownReady);
        QVERIFY(service.prepareEnvironment());
        QTRY_COMPARE_WITH_TIMEOUT(completed.size(), 1, 3000);
        QCOMPARE(completed[0][0].toBool(), false); QCOMPARE(completed[0][1].toBool(), false);
        const auto messages = logs(log);
        for (const auto &message : messages) QVERIFY(message.size() <= 16000);
        QVERIFY(messages.contains("after large output"));
        QTest::qWait(30); QCOMPARE(probe.size(), 0);
        service.requestShutdown(); QTRY_COMPARE_WITH_TIMEOUT(shutdown.size(), 1, 2000);
    }

    void shutdownFromStateSignalDoesNotLaunchAHelperAfterShutdown()
    {
        const auto value = options("import time\ntime.sleep(30)\n");
        GpuEnvironmentService service(value);
        QSignalSpy shutdown(&service, &GpuEnvironmentService::shutdownReady);
        QSignalSpy probe(&service, &GpuEnvironmentService::environmentChanged);
        bool started = false;
        connect(&service, &GpuEnvironmentService::stateChanged, this, [&] {
            if (service.isChecking() && !service.shutdownRequested()) {
                auto *process = service.findChild<QProcess *>();
                QVERIFY(process);
                connect(process, &QProcess::started, this, [&] { started = true; });
                service.requestShutdown();
            }
        });
        QVERIFY(!service.checkEnvironment());
        QTRY_COMPARE_WITH_TIMEOUT(shutdown.size(), 1, 2000);
        QVERIFY(!started); QCOMPARE(probe.size(), 0);
    }

    void cancellationAndShutdownKillOnlyOwnedHelperAsynchronously()
    {
        auto value = options("print('{\"cuda_available\":false}')\n",
            "import signal,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
            "print('helper ready',flush=True)\ntime.sleep(30)\n");
        GpuEnvironmentService service(value);
        QSignalSpy log(&service, &GpuEnvironmentService::logAppended);
        QSignalSpy completed(&service, &GpuEnvironmentService::preparationFinished);
        QSignalSpy probe(&service, &GpuEnvironmentService::environmentChanged);
        QSignalSpy shutdown(&service, &GpuEnvironmentService::shutdownReady);
        QVERIFY(service.prepareEnvironment());
        QTRY_VERIFY_WITH_TIMEOUT(logs(log).contains("helper ready"), 3000);
        int ticks = 0; QTimer timer;
        connect(&timer, &QTimer::timeout, this, [&] { ++ticks; }); timer.start(5);
        QElapsedTimer elapsed; elapsed.start();
        service.cancelPreparation();
        QVERIFY(elapsed.elapsed() < 100);
        QVERIFY(service.isCancelling());
        service.cancelPreparation();
        service.requestShutdown();
        QVERIFY(!service.checkEnvironment()); QVERIFY(!service.prepareEnvironment());
        QTRY_COMPARE_WITH_TIMEOUT(shutdown.size(), 1, 3000);
        QCOMPARE(completed.size(), 1);
        QCOMPARE(completed[0][0].toBool(), false); QCOMPARE(completed[0][1].toBool(), true);
        QCOMPARE(probe.size(), 0); QVERIFY(ticks > 0);
        QVERIFY(service.findChildren<QProcess *>().isEmpty());
    }

    void probeShutdownAwaitsReapingAndSuppressesNewWork()
    {
        auto value = options("import time\ntime.sleep(30)\n");
        GpuEnvironmentService service(value);
        QSignalSpy shutdown(&service, &GpuEnvironmentService::shutdownReady);
        QVERIFY(service.checkEnvironment());
        QElapsedTimer elapsed; elapsed.start(); service.requestShutdown();
        QVERIFY(elapsed.elapsed() < 100);
        QVERIFY(!service.checkEnvironment()); QVERIFY(!service.prepareEnvironment());
        QTRY_COMPARE_WITH_TIMEOUT(shutdown.size(), 1, 3000);
        QVERIFY(service.isShutdownReady());
        QVERIFY(service.findChildren<QProcess *>().isEmpty());
    }

    void earlyOwnerDestructionNeverWaitsForStubbornHelper()
    {
        auto value = options("import signal,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\ntime.sleep(30)\n");
        auto *service = new GpuEnvironmentService(value);
        QVERIFY(service->checkEnvironment());
        const QPointer<QProcess> process(service->findChild<QProcess *>());
        QVERIFY(process);
        QTRY_VERIFY_WITH_TIMEOUT(process->state() == QProcess::Running, 2000);
        QTest::qWait(50);
        QElapsedTimer elapsed; elapsed.start(); delete service;
        QVERIFY(elapsed.elapsed() < 100);
        QTRY_VERIFY_WITH_TIMEOUT(process.isNull(), 3000);
    }
};

QTEST_GUILESS_MAIN(GpuEnvironmentServiceTests)
#include "gpu_environment_service_tests.moc"
