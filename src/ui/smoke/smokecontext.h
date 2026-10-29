// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include "ui/mainwindow.h"
#include "ui/pages/workbenchpage.h"
#include <QLabel>
#include <QPushButton>
#include <QDebug>

namespace vision::ui::smoke
{
// Test-only event driver. UI handles are obtained from the actual presentation
// page; business state is queried through the same public application services.
// The window owns the context so pending callbacks cannot outlive it.
class SmokeContext : public QObject
{
  public:
    explicit SmokeContext(MainWindow &window) : QObject(&window), window_(window)
    {
        auto *page = window.findChild<WorkbenchPage *>();
        if (page) ui_ = page->controls();
        status_ = window.findChild<QLabel *>("statusText");
    }
    bool hasWorkbench() const
    {
        return ui_.canvas && ui_.queue && ui_.evaluation && ui_.runButton &&
            ui_.stopButton && ui_.workbenchTabs && status_;
    }
  protected:
    void selectRoute(int index)
    {
        if (auto *button = window_.findChild<QPushButton *>(QString("navigation-%1").arg(index))) button->click();
    }
    MainWindow &window_;
    WorkbenchPage::Controls ui_;
    QLabel *status_ = nullptr;
};
}
