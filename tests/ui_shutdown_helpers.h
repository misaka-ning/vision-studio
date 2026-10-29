// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include "ui/mainwindow.h"
#include <QTest>

namespace vision::test
{
// A failed init may leave no window. Otherwise keep the GUI event loop running
// while its owners finish accepted work; only the caller then resets ownership.
// This deliberately does not shut down any application-wide singleton.
inline bool shutdownWindow(MainWindow *window, int timeoutMs = 30000)
{
    if (!window) return true;
    window->close();
    return QTest::qWaitFor([window] { return window->isShutdownReady(); }, timeoutMs);
}
}
