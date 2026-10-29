// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include "ui/icons.h"
#include <QFrame>
#include <QHeaderView>
#include <QLabel>
#include <QPushButton>
#include <QTableWidget>

namespace vision::ui::pages
{
inline QLabel *text(const QString &value, const char *name = "body")
{
    auto *w = new QLabel(value);
    w->setObjectName(name);
    w->setTextFormat(Qt::PlainText);
    return w;
}

inline QPushButton *button(const QString &title, const QString &iconName = {}, const char *role = "secondary")
{
    auto *b = new QPushButton(title);
    b->setObjectName(role);
    b->setProperty("fluentRole", QString::fromLatin1(role));
    b->setCursor(Qt::PointingHandCursor);
    b->setMinimumHeight(36);
    if (!iconName.isEmpty())
        b->setIcon(::ui::icon(iconName, role == QString("primary") ? QColor("#00374D") : QColor("#D2D2D2")));
    b->setIconSize(QSize(18, 18));
    return b;
}

inline QFrame *card(const char *name = "card")
{
    auto *w = new QFrame;
    w->setObjectName(name);
    return w;
}

inline void tableStyle(QTableWidget *t)
{
    t->setAlternatingRowColors(false);
    t->setShowGrid(false);
    t->setSelectionBehavior(QAbstractItemView::SelectRows);
    t->setSelectionMode(QAbstractItemView::SingleSelection);
    t->setEditTriggers(QAbstractItemView::NoEditTriggers);
    t->verticalHeader()->hide();
    t->verticalHeader()->setDefaultSectionSize(42);
    t->horizontalHeader()->setHighlightSections(false);
    t->setFrameShape(QFrame::NoFrame);
    t->setFocusPolicy(Qt::StrongFocus);
}
}
