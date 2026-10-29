// SPDX-License-Identifier: AGPL-3.0-only
#include "guidepage.h"
#include <QFile>
#include <QTextBrowser>
#include <QVBoxLayout>

GuidePage::GuidePage(QWidget *parent) : QFrame(parent)
{
    setObjectName("guidePage");
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(24, 18, 24, 18);
    controls_.browser = new QTextBrowser;
    controls_.browser->setObjectName("guideBrowser");
    controls_.browser->setOpenExternalLinks(true);
    QFile guide(":/guide.html");
    if (guide.open(QIODevice::ReadOnly))
        controls_.browser->setHtml(QString::fromUtf8(guide.readAll()));
    else
        controls_.browser->setPlainText("使用指南资源无法读取，请查阅随版本提供的 docs/使用指南.md。");
    layout->addWidget(controls_.browser);
}
