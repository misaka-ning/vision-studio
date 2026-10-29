// SPDX-License-Identifier: AGPL-3.0-only
#include "mainwindow.h"
#include "smoke/smokedrivers.h"

void MainWindow::runPreviewDropSmoke(const QString &directory, const QString &requestPath)
{
    vision::ui::smoke::runPreviewDropSmoke(*this, directory, requestPath);
}
