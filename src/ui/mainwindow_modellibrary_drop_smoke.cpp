// SPDX-License-Identifier: AGPL-3.0-only
#include "mainwindow.h"
#include "smoke/smokedrivers.h"

void MainWindow::runModelLibraryDropSmoke(const QString &directory)
{
    vision::ui::smoke::runModelLibraryDropSmoke(*this, directory);
}
