// SPDX-License-Identifier: AGPL-3.0-only
#include "core/cpuruntime.h"
#include "ui/runtimesetup.h"
#include <QApplication>
#include <QCryptographicHash>
#include <QDialog>
#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QLabel>
#include <QLockFile>
#include <QPushButton>
#include <QTcpServer>
#include <QTcpSocket>
#include <QTemporaryDir>
#include <QtTest>

class CpuRuntimeTests : public QObject {
    Q_OBJECT
    QTemporaryDir temp;
    QMap<QByteArray,QByteArray> saved;
    QSet<QByteArray> existed;
    QString home, data;
    QJsonObject descriptor;
    void write(const QString &path, const QByteArray &bytes) {
        QDir().mkpath(QFileInfo(path).absolutePath()); QFile f(path);
        QVERIFY(f.open(QIODevice::WriteOnly)); QCOMPARE(f.write(bytes), bytes.size());
    }
    void marker() { write(home + "/cpu-runtime.json", QJsonDocument(descriptor).toJson()); }
    void fakePython(const QString &path) {
        QDir().mkpath(QFileInfo(path).absolutePath());
        QVERIFY(QFile::copy(qEnvironmentVariable("SystemRoot") + "/System32/cmd.exe", path));
    }
    QDialog *dialog() { return qApp->activeModalWidget() ? qobject_cast<QDialog *>(qApp->activeModalWidget()) : nullptr; }
    void closeDialog() { if (auto *d = dialog()) { d->reject(); QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete); } }
private slots:
    void initTestCase() {
        for (const QByteArray name : {QByteArray("VISION_STUDIO_HOME"), QByteArray("VISION_STUDIO_DATA_DIR"),
             QByteArray("VISION_STUDIO_PYTHON"), QByteArray("VISION_STUDIO_SYSTEM_INSTALL")}) {
            if (qEnvironmentVariableIsSet(name)) existed.insert(name);
            saved[name] = qgetenv(name);
        }
    }
    void init() {
        QVERIFY(temp.isValid());
        home = temp.path() + "/package-" + QString::number(QDateTime::currentMSecsSinceEpoch());
        data = home + "/personal";
        QDir().mkpath(home);
        qputenv("VISION_STUDIO_HOME", home.toUtf8()); qputenv("VISION_STUDIO_DATA_DIR", data.toUtf8());
        qunsetenv("VISION_STUDIO_PYTHON"); qputenv("VISION_STUDIO_SYSTEM_INSTALL", "1");
        descriptor = {{"schema_version",1}, {"sha256",QString(64,'a')}, {"bytes",12},
                      {"url","https://example.invalid/cpu.zip"}};
        marker();
    }
    void cleanup() { closeDialog(); QTest::qWait(2); }
    void cleanupTestCase() {
        for (auto it=saved.begin(); it!=saved.end(); ++it)
            if (existed.contains(it.key())) qputenv(it.key(),it.value()); else qunsetenv(it.key());
    }
    void noBuildMachineFallback() {
        QVERIFY(vision::cpuPython().isEmpty()); QVERIFY(vision::cpuRuntimeDownloadNeeded());
    }
    void explicitOverrideWinsAndNeverDownloads() {
        qputenv("VISION_STUDIO_PYTHON", "Z:/invalid/python.exe");
        QVERIFY(vision::cpuPython().isEmpty()); QVERIFY(!vision::cpuRuntimeDownloadNeeded());
    }
    void bundledRuntimeWins() {
        fakePython(home + "/runtime/python.exe");
        QCOMPARE(vision::cpuPython(), home + "/runtime/python.exe"); QVERIFY(!vision::cpuRuntimeDownloadNeeded());
    }
    void cacheRequiresMatchingReadyMarker() {
        const auto cache=vision::cpuRuntimeDirectory(descriptor);
        fakePython(cache + "/python.exe");
        QVERIFY(vision::cpuPython().isEmpty());
        write(cache + "/ready.json", QJsonDocument(QJsonObject{{"sha256",QString(64,'b')}}).toJson());
        QVERIFY(vision::cpuPython().isEmpty());
        write(cache + "/ready.json", QJsonDocument(descriptor).toJson());
        QCOMPARE(vision::cpuPython(),cache + "/python.exe");
    }
    void cacheIdentifierRejectsTraversal() {
        descriptor.insert("sha256","../../escape"); marker();
        QVERIFY(vision::cpuRuntimeDirectory(descriptor).isEmpty());
    }
    void promptingDoesNotDownloadAndCancelDropsRetries() {
        QWidget owner; bool retried=false;
        QVERIFY(!ui::ensureCpuRuntime(&owner,[&]{retried=true;}));
        QTRY_VERIFY(dialog());
        QCOMPARE(dialog()->findChild<QPushButton *>("prepareCpuRuntimeButton")->text(),QString("下载并准备"));
        QVERIFY(!QFileInfo::exists(vision::cpuRuntimeCacheRoot()));
        closeDialog(); QVERIFY(!retried);
    }
    void concurrentRequestsUseOneDialogAndResumeTogether() {
        QWidget owner; int retried=0;
        QVERIFY(!ui::ensureCpuRuntime(&owner,[&]{++retried;}));
        QVERIFY(!ui::ensureCpuRuntime(&owner,[&]{++retried;}));
        QTRY_VERIFY(dialog()); auto *d=dialog();
        fakePython(home + "/runtime/python.exe");
        d->findChild<QPushButton *>("prepareCpuRuntimeButton")->click();
        QTRY_COMPARE(retried,2);
    }
    void checksumFailureCanRetryWithoutExecutingArchive() {
        QTcpServer server; QVERIFY(server.listen(QHostAddress::LocalHost)); int requests=0;
        connect(&server,&QTcpServer::newConnection,this,[&] {
            auto *socket=server.nextPendingConnection(); ++requests;
            connect(socket,&QTcpSocket::readyRead,socket,[socket] {
                socket->readAll(); socket->write("HTTP/1.1 200 OK\r\nContent-Length: 12\r\nConnection: close\r\n\r\ninvalid-zip!");
                socket->disconnectFromHost();
            }); connect(socket,&QTcpSocket::disconnected,socket,&QObject::deleteLater);
        });
        descriptor.insert("url",QString("http://127.0.0.1:%1/cpu.zip").arg(server.serverPort())); marker();
        QWidget owner; bool retried=false;
        QVERIFY(!ui::ensureCpuRuntime(&owner,[&]{retried=true;})); QTRY_VERIFY(dialog());
        auto *button=dialog()->findChild<QPushButton *>("prepareCpuRuntimeButton"); button->click();
        QTRY_COMPARE(button->text(),QString("重试")); QVERIFY(button->isEnabled()); QVERIFY(!retried);
        QVERIFY(!QFileInfo::exists(vision::cpuRuntimeDirectory(descriptor)+"/python.exe"));
        button->click(); QTRY_COMPARE(requests,2); QTRY_VERIFY(button->isEnabled()); QVERIFY(!retried);
    }
    void cancellationReleasesPreparationLock() {
        QTcpServer server; QVERIFY(server.listen(QHostAddress::LocalHost));
        descriptor.insert("url",QString("http://127.0.0.1:%1/cpu.zip").arg(server.serverPort())); marker();
        QWidget owner;
        QVERIFY(!ui::ensureCpuRuntime(&owner,[]{})); QTRY_VERIFY(dialog());
        dialog()->findChild<QPushButton *>("prepareCpuRuntimeButton")->click();
        const auto lockPath=vision::cpuRuntimeDirectory(descriptor)+".lock";
        QTRY_VERIFY(QFileInfo::exists(lockPath)); closeDialog();
        QLockFile lock(lockPath); QVERIFY(lock.tryLock(0));
    }
};
QTEST_MAIN(CpuRuntimeTests)
#include "cpu_runtime_tests.moc"
