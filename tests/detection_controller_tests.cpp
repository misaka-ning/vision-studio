// SPDX-License-Identifier: AGPL-3.0-only
#include "app/detectioncontroller.h"
#include <QSignalSpy>
#include <QThread>
#include <QtTest>

class DetectionControllerTests : public QObject
{
    Q_OBJECT
  private slots:
    void asynchronousShutdownCanBeRequestedRepeatedly()
    {
        vision::app::DetectionController controller;
        QSignalSpy ready(&controller, &vision::app::DetectionController::shutdownReady);
        controller.requestShutdown(); controller.requestShutdown();
        QTRY_VERIFY_WITH_TIMEOUT(controller.isShutdownReady(), 3000);
        QCOMPARE(ready.size(), 1);
    }
    void destructionAfterWorkerExitBeforeQueuedFinishedIsDelivered()
    {
        // Reproduce the shutdown race: the worker can already be destroyed
        // while its queued GUI-thread finished notification is still pending.
        for (int iteration = 0; iteration < 25; ++iteration) {
            auto controller = std::make_unique<vision::app::DetectionController>();
            controller->requestShutdown();
            QThread::msleep(20);
            controller.reset();
            QCoreApplication::processEvents();
        }
    }
    void preparationThenImmediateStopIsPreserved()
    {
        vision::app::DetectionController controller;
        controller.prepare(); controller.requestStop();
        QVERIFY(controller.snapshot().stopping);
        controller.requestShutdown();
        QTRY_VERIFY_WITH_TIMEOUT(controller.isShutdownReady(), 3000);
    }
};
QTEST_GUILESS_MAIN(DetectionControllerTests)
#include "detection_controller_tests.moc"
