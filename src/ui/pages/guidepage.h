// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <QFrame>

class QTextBrowser;

class GuidePage final : public QFrame
{
    Q_OBJECT
public:
    struct Controls { QTextBrowser *browser = nullptr; };
    explicit GuidePage(QWidget *parent = nullptr);
    const Controls &controls() const { return controls_; }

private:
    Controls controls_;
};
