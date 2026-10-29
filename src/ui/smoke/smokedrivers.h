// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <QString>
class MainWindow;
namespace vision::ui::smoke
{
void runInferenceSmoke(MainWindow &window, const QString &directory, bool pt);
void runModelDisplaySmoke(MainWindow &window, const QString &directory, const QString &model);
void runPreviewDropSmoke(MainWindow &window, const QString &directory, const QString &requestPath);
void runModelLibraryDropSmoke(MainWindow &window, const QString &directory);
void runEvaluationSmoke(MainWindow &window, const QString &directory, const QString &requestPath);
void runDatasetSessionSmoke(MainWindow &window, const QString &directory);
}
