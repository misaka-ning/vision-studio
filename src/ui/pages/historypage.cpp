// SPDX-License-Identifier: AGPL-3.0-only
#include "historypage.h"
#include "pagehelpers.h"
#include <QDateTime>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonObject>
#include <QVBoxLayout>

using namespace vision::ui::pages;

HistoryPage::HistoryPage(const QString &historyFilePath, QWidget *parent) : QWidget(parent)
{
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(16);
    auto *row = new QHBoxLayout;
    row->addWidget(text("最近 200 次检测与评估记录 · 双击评估记录回看报告", "sectionTitle"), 1);
    controls_.exportButton = button("导出运行记录", "export");
    row->addWidget(controls_.exportButton);
    connect(controls_.exportButton, &QPushButton::clicked, this, &HistoryPage::exportRequested);
    layout->addLayout(row);
    auto *frame = card();
    auto *frameLayout = new QVBoxLayout(frame);
    frameLayout->setContentsMargins(12, 12, 12, 12);
    controls_.table = new QTableWidget(0, 6);
    controls_.table->setHorizontalHeaderLabels({"时间", "输入", "模型", "目标数", "推理耗时", "任务"});
    tableStyle(controls_.table);
    controls_.table->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    controls_.table->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    controls_.table->setColumnWidth(0, 165);
    controls_.table->setColumnWidth(3, 85);
    controls_.table->setColumnWidth(4, 100);
    controls_.table->setColumnWidth(5, 120);
    frameLayout->addWidget(controls_.table);
    layout->addWidget(frame, 1);
    controls_.hint = text("运行记录已自动保存。导出图像与原始预测数据可在工作台完成。", "muted");
    setHistoryFilePath(historyFilePath);
    layout->addWidget(controls_.hint);
    connect(controls_.table, &QTableWidget::cellDoubleClicked, this,
            [this](int row, int) { emit activated(row); });
}

void HistoryPage::setHistoryFilePath(const QString &path)
{
    controls_.hint->setToolTip(path);
}

void HistoryPage::setRecords(const QJsonArray &records)
{
    controls_.table->setRowCount(records.size());
    for (int row = 0; row < records.size(); ++row) {
        const auto record = records[row].toObject();
        const auto stereo = record["stereo_view"].toString();
        const QStringList values = {
            QDateTime::fromString(record["time"].toString(), Qt::ISODateWithMs).toString("MM-dd  HH:mm:ss"),
            QFileInfo(record["source"].toString()).fileName()
                + (stereo == "left" ? QStringLiteral(" · 双目左目")
                   : stereo == "right" ? QStringLiteral(" · 双目右目") : QString()),
            record["model"].toString()
                + (record.contains("actual_device")
                       ? " · " + (record["actual_device"].toString() == "cuda" ? QStringLiteral("GPU") : QStringLiteral("CPU"))
                       : QString()),
            QString::number(record["objects"].toInt()),
            QString::number(record["inference_ms"].toDouble(), 'f', 1) + " ms",
            record["task"].toString()};
        for (int column = 0; column < values.size(); ++column) {
            auto *item = new QTableWidgetItem(values[column]);
            item->setToolTip(column == 1 ? record["source"].toString()
                             : column == 2 ? values[column] + "\n" + record["backend"].toString() + "\n"
                                                 + record["device_name"].toString() + "\n" + record["device_notice"].toString()
                                           : values[column]);
            if (column == 0) item->setData(RowData, record);
            controls_.table->setItem(row, column, item);
        }
    }
}
