// SPDX-License-Identifier: AGPL-3.0-only
#include "core/cpuruntime.h"
#include "ui/runtimesetup.h"
#include <QApplication>
#include <QCryptographicHash>
#include <QDialog>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QJsonDocument>
#include <QLockFile>
#include <QPointer>
#include <QPushButton>
#include <QTcpServer>
#include <QTcpSocket>
#include <QTemporaryDir>
#include <QTimer>
#include <QtTest>
#ifdef Q_OS_UNIX
#include <cerrno>
#include <csignal>
#endif
#ifdef Q_OS_WIN
#include <qt_windows.h>
#endif

namespace {
QString package;
QJsonObject manifest;
bool runtimeReady = false;
constexpr char Payload[] = "isolated-runtime-archive";

bool processExited(qint64 pid)
{
#ifdef Q_OS_WIN
    HANDLE process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, DWORD(pid));
    if (!process) return true;
    DWORD code = STILL_ACTIVE;
    const bool exited = GetExitCodeProcess(process, &code) && code != STILL_ACTIVE;
    CloseHandle(process);
    return exited;
#else
    return ::kill(pid_t(pid), 0) == -1 && errno == ESRCH;
#endif
}

int helperMain(int argc, char **argv)
{
    QCoreApplication app(argc, argv);
    const auto arguments = app.arguments();
    const auto stage = arguments.value(arguments.indexOf("-Destination") + 1);
    if (!QDir().mkpath(stage)) return 3;
    QFile marker(stage + "/helper.pid");
    if (!marker.open(QIODevice::WriteOnly)) return 4;
    marker.write(QByteArray::number(app.applicationPid()));
    marker.close();
    const auto mode = qgetenv("VISION_RUNTIME_HELPER_MODE");
    if (mode == "success") return 0;
    if (mode == "failure") return 7;
    // Bound a failed test helper without depending on its parent for cleanup.
    QTimer::singleShot(30000, &app, &QCoreApplication::quit);
    return app.exec();
}
}

// Isolate the UI lifecycle from platform package discovery and real downloads.
namespace vision {
QString packageRoot() { return package; }
QString cpuRuntimeCacheRoot() { return package + "/personal/runtimes/cpu"; }
QJsonObject cpuRuntimeDescriptor() { return manifest; }
QString cpuRuntimeDirectory(const QJsonObject &descriptor) { return cpuRuntimeCacheRoot() + "/" + descriptor.value("sha256").toString(); }
QString cpuPython(const QString &) { return runtimeReady ? package + "/python" : QString(); }
bool cpuRuntimeDownloadNeeded() { return !runtimeReady; }
}

class RuntimeSetupShutdownTests final : public QObject
{
    Q_OBJECT
    QTemporaryDir temp;
    QTcpServer server;
    QByteArray previousSystemRoot;
    bool hadSystemRoot = false;
    int serial = 0;
    bool servePayload = false;

    QDialog *dialog() const
    {
        for (auto *widget : QApplication::topLevelWidgets())
            if (widget->objectName() == "cpuRuntimeSetupDialog") return qobject_cast<QDialog *>(widget);
        return nullptr;
    }
    QString stage() const
    {
        const auto entries = QDir(vision::cpuRuntimeCacheRoot()).entryList({".staging-*"}, QDir::Dirs | QDir::Hidden | QDir::NoDotAndDotDot);
        return entries.isEmpty() ? QString() : QDir(vision::cpuRuntimeCacheRoot()).filePath(entries.front());
    }
    qint64 childPid() const
    {
        QFile marker(stage() + "/helper.pid");
        return marker.open(QIODevice::ReadOnly) ? marker.readAll().toLongLong() : 0;
    }
    void openAndStart(QWidget &owner, std::function<void()> retry = [] {})
    {
        QVERIFY(!ui::ensureCpuRuntime(&owner, std::move(retry)));
        QVERIFY(dialog());
        dialog()->findChild<QPushButton *>("prepareCpuRuntimeButton")->click();
    }

private slots:
    void initTestCase()
    {
        QVERIFY(temp.isValid());
        qApp->setQuitOnLastWindowClosed(false);
        hadSystemRoot = qEnvironmentVariableIsSet("SystemRoot");
        previousSystemRoot = qgetenv("SystemRoot");
        connect(&server, &QTcpServer::newConnection, this, [this] {
            while (auto *socket = server.nextPendingConnection()) {
                connect(socket, &QTcpSocket::readyRead, socket, [this, socket] {
                    socket->readAll();
                    if (!servePayload) return;
                    const QByteArray payload(Payload);
                    socket->write("HTTP/1.1 200 OK\r\nContent-Length: " + QByteArray::number(payload.size())
                        + "\r\nConnection: close\r\n\r\n" + payload);
                    socket->disconnectFromHost();
                });
                connect(socket, &QTcpSocket::disconnected, socket, &QObject::deleteLater);
            }
        });
    }
    void init()
    {
        package = temp.path() + "/case-" + QString::number(++serial);
        const auto systemRoot = package + "/system";
        const auto helper = systemRoot + "/System32/WindowsPowerShell/v1.0/powershell.exe";
        QVERIFY(QDir().mkpath(QFileInfo(helper).absolutePath()));
        QVERIFY(QFile::copy(QCoreApplication::applicationFilePath(), helper));
        QVERIFY(QFile::setPermissions(helper, QFile::ReadOwner | QFile::WriteOwner | QFile::ExeOwner));
        qputenv("SystemRoot", systemRoot.toUtf8());
        qputenv("VISION_RUNTIME_HELPER_MODE", "hold");
        runtimeReady = false;
        servePayload = false;
        QVERIFY(server.listen(QHostAddress::LocalHost));
        manifest = {{"schema_version", 1},
            {"sha256", QString::fromLatin1(QCryptographicHash::hash(QByteArray(Payload), QCryptographicHash::Sha256).toHex())},
            {"bytes", int(sizeof(Payload) - 1)}, {"uncompressed_bytes", 1024},
            {"url", QString("http://127.0.0.1:%1/runtime.zip").arg(server.serverPort())}};
    }
    void cleanup()
    {
        auto *setup = ui::runtimeSetup();
        setup->requestShutdown();
        QTRY_VERIFY(setup->isShutdownReady());
        QVERIFY(!dialog());
        delete setup;
        server.close();
        for (auto *socket : server.findChildren<QTcpSocket *>()) delete socket;
        qunsetenv("VISION_RUNTIME_HELPER_MODE");
    }
    void cleanupTestCase()
    {
        if (hadSystemRoot) qputenv("SystemRoot", previousSystemRoot); else qunsetenv("SystemRoot");
    }
    void idleShutdownIsQueuedIdempotentAndClosesAdmission()
    {
        auto *setup = ui::runtimeSetup();
        QSignalSpy ready(setup, &ui::RuntimeSetup::shutdownReady);
        QVERIFY(setup->isShutdownReady());
        setup->requestShutdown(); setup->requestShutdown();
        QCOMPARE(ready.count(), 0);
        QTRY_COMPARE(ready.count(), 1);
        QWidget owner;
        QVERIFY(!ui::ensureCpuRuntime(&owner, [] { QFAIL("shutdown admitted a retry"); }));
        QVERIFY(!dialog());
    }
    void promptShutdownDoesNotDownloadOrRetry()
    {
        QWidget owner;
        int retries = 0;
        QVERIFY(!ui::ensureCpuRuntime(&owner, [&] { ++retries; }));
        QVERIFY(dialog());
        auto *setup = ui::runtimeSetup();
        QVERIFY(!setup->isShutdownReady());
        QSignalSpy ready(setup, &ui::RuntimeSetup::shutdownReady);
        setup->requestShutdown();
        QTRY_COMPARE(ready.count(), 1);
        QCOMPARE(retries, 0);
        QVERIFY(!QFileInfo::exists(vision::cpuRuntimeCacheRoot()));
    }
    void shutdownSuppressesAlreadyQueuedSuccessfulRetry()
    {
        QWidget owner;
        int retries = 0;
        QVERIFY(!ui::ensureCpuRuntime(&owner, [&] { ++retries; }));
        runtimeReady = true;
        dialog()->findChild<QPushButton *>("prepareCpuRuntimeButton")->click();
        ui::runtimeSetup()->requestShutdown();
        QTRY_VERIFY(ui::runtimeSetup()->isShutdownReady());
        QCOMPARE(retries, 0);
    }
    void cancellationSuppressesAlreadyQueuedRetryAndStillAllowsNewPreparation()
    {
        QWidget owner;
        int retries = 0;
        auto *setup = ui::runtimeSetup();
        QVERIFY(!ui::ensureCpuRuntime(&owner, [&] { ++retries; }));
        runtimeReady = true;
        // Actual success accepts the dialog and extracts its retries into
        // queued callbacks. Cancel in this same turn before they can execute.
        dialog()->findChild<QPushButton *>("prepareCpuRuntimeButton")->click();
        setup->cancelPendingPreparations();
        QTRY_VERIFY(setup->isShutdownReady());
        QCOMPARE(retries, 0);

        runtimeReady = false;
        QVERIFY(!ui::ensureCpuRuntime(&owner, [&] { ++retries; }));
        QVERIFY(dialog());
        runtimeReady = true;
        dialog()->findChild<QPushButton *>("prepareCpuRuntimeButton")->click();
        QTRY_COMPARE(retries, 1);
        QTRY_VERIFY(setup->isShutdownReady());
    }
    void downloadingShutdownReleasesOwnLock()
    {
        QWidget owner;
        openAndStart(owner);
        const auto lockPath = vision::cpuRuntimeDirectory(manifest) + ".lock";
        QVERIFY(QFileInfo::exists(lockPath));
        auto *setup = ui::runtimeSetup();
        setup->requestShutdown();
        QTRY_VERIFY(setup->isShutdownReady());
        QLockFile probe(lockPath);
        QVERIFY(probe.tryLock(0));
        QVERIFY(stage().isEmpty());
    }
    void waitingShutdownKeepsAnotherPreparationsLock()
    {
        QVERIFY(QDir().mkpath(vision::cpuRuntimeCacheRoot()));
        const auto lockPath = vision::cpuRuntimeDirectory(manifest) + ".lock";
        QLockFile other(lockPath);
        QVERIFY(other.tryLock(0));
        QWidget owner;
        openAndStart(owner);
        ui::runtimeSetup()->requestShutdown();
        QTRY_VERIFY(ui::runtimeSetup()->isShutdownReady());
        QLockFile probe(lockPath);
        QVERIFY(!probe.tryLock(0));
        QVERIFY(QFileInfo::exists(lockPath));
    }
    void extractingShutdownReapsChildBeforeReadyAndCleansOnlyOwnStage()
    {
        servePayload = true;
        QWidget owner;
        int retries = 0;
        openAndStart(owner, [&] { ++retries; });
        QTRY_VERIFY(childPid() > 0);
        const auto pid = childPid();
        const auto ownStage = stage();
        const auto foreignStage = vision::cpuRuntimeCacheRoot() + "/.staging-unrelated";
        QVERIFY(QDir().mkpath(foreignStage));
        auto *setup = ui::runtimeSetup();
        QSignalSpy ready(setup, &ui::RuntimeSetup::shutdownReady);
        QElapsedTimer elapsed; elapsed.start();
        setup->requestShutdown();
        QVERIFY(elapsed.elapsed() < 100);
        QVERIFY(!setup->isShutdownReady());
        QTRY_COMPARE(ready.count(), 1);
        QVERIFY(processExited(pid));
        QVERIFY(!QFileInfo::exists(ownStage));
        QVERIFY(QFileInfo(foreignStage).isDir());
        QLockFile probe(vision::cpuRuntimeDirectory(manifest) + ".lock");
        QVERIFY(probe.tryLock(0));
        QCOMPARE(retries, 0);
    }
    void directDialogDeletionStillDrainsResources()
    {
        servePayload = true;
        QWidget owner;
        openAndStart(owner);
        QTRY_VERIFY(childPid() > 0);
        const auto pid = childPid();
        const auto ownStage = stage();
        QElapsedTimer elapsed; elapsed.start();
        delete dialog();
        QVERIFY(elapsed.elapsed() < 100);
        auto *setup = ui::runtimeSetup();
        QVERIFY(!setup->isShutdownReady());
        setup->requestShutdown();
        QTRY_VERIFY(setup->isShutdownReady());
        QVERIFY(processExited(pid));
        QVERIFY(!QFileInfo::exists(ownStage));
    }
    void failedExtractorCanRetryAfterCleanupThenPublish()
    {
        servePayload = true;
        qputenv("VISION_RUNTIME_HELPER_MODE", "failure");
        QWidget owner;
        int retries = 0;
        openAndStart(owner, [&] { ++retries; });
        QPointer<QPushButton> button(dialog()->findChild<QPushButton *>("prepareCpuRuntimeButton"));
        QTRY_VERIFY(button && button->text() == "重试" && button->isEnabled());
        QVERIFY(stage().isEmpty());
        QLockFile probe(vision::cpuRuntimeDirectory(manifest) + ".lock");
        QVERIFY(probe.tryLock(0)); probe.unlock();
        qputenv("VISION_RUNTIME_HELPER_MODE", "success");
        button->click();
        QTRY_COMPARE(retries, 1);
        QTRY_VERIFY(ui::runtimeSetup()->isShutdownReady());
        QVERIFY(QFileInfo::exists(vision::cpuRuntimeDirectory(manifest) + "/ready.json"));
    }
    void failedToStartExtractorStillCleansAndCanShutdown()
    {
        servePayload = true;
        const auto helper = qEnvironmentVariable("SystemRoot") + "/System32/WindowsPowerShell/v1.0/powershell.exe";
        QVERIFY(QFile::remove(helper));
        QWidget owner;
        openAndStart(owner);
        QPointer<QPushButton> button(dialog()->findChild<QPushButton *>("prepareCpuRuntimeButton"));
        QTRY_VERIFY(button && button->text() == "重试" && button->isEnabled());
        QVERIFY(stage().isEmpty());
        QLockFile probe(vision::cpuRuntimeDirectory(manifest) + ".lock");
        QVERIFY(probe.tryLock(0)); probe.unlock();
        ui::runtimeSetup()->requestShutdown();
        QTRY_VERIFY(ui::runtimeSetup()->isShutdownReady());
    }
};

int main(int argc, char **argv)
{
    for (int i = 1; i < argc; ++i)
        if (QByteArray(argv[i]) == "-Destination") return helperMain(argc, argv);
    QApplication app(argc, argv);
    RuntimeSetupShutdownTests tests;
    return QTest::qExec(&tests, argc, argv);
}
#include "runtime_setup_shutdown_tests.moc"
