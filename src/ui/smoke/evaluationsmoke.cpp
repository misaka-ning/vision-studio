// SPDX-License-Identifier: AGPL-3.0-only
#include "smokedrivers.h"
#include "smokecontext.h"
#include "ui/evaluationpanel.h"
#include "core/evaluationworker.h"
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QCryptographicHash>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QPushButton>
#include <QSaveFile>
#include <QSpinBox>
#include <QSplitter>
#include <QTabWidget>
#include <QTableWidget>
#include <QTimer>
#include <cmath>
namespace
{
QString sha256(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) return {};
    QCryptographicHash hash(QCryptographicHash::Sha256);
    while (!file.atEnd()) {
        auto chunk = file.read(1024 * 1024);
        if (chunk.isEmpty() && file.error() != QFile::NoError) return {};
        hash.addData(chunk);
    }
    return QString::fromLatin1(hash.result().toHex());
}
QJsonObject jsonFile(const QString &path)
{
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) return {};
    return QJsonDocument::fromJson(f.readAll()).object();
}
// Smoke-only observations use the real widget tree and clipping ancestors.
// They do not run a nested event loop or alter ordinary workbench behavior.
QJsonObject smokeWidgetGeometry(QWidget *widget, QWidget *window)
{
    if (!widget) return {{"exists", false}, {"fully_visible", false}};
    const QRect bounds(widget->mapTo(window, QPoint()), widget->size());
    bool visible = widget->isVisibleTo(window) && !widget->size().isEmpty() && window->rect().contains(bounds);
    QJsonArray clips;
    for (auto *parent = widget->parentWidget(); parent && parent != window; parent = parent->parentWidget()) {
        const QRect clipping(widget->mapTo(parent, QPoint()), widget->size());
        clips.append(QJsonObject{{"ancestor", parent->objectName()}, {"width", parent->width()}, {"height", parent->height()},
            {"child_x", clipping.x()}, {"child_y", clipping.y()}, {"child_width", clipping.width()}, {"child_height", clipping.height()}});
        visible = visible && parent->rect().contains(clipping);
    }
    QJsonObject result{{"exists", true}, {"object_name", widget->objectName()}, {"fully_visible", visible},
        {"x", bounds.x()}, {"y", bounds.y()}, {"width", bounds.width()}, {"height", bounds.height()},
        {"enabled", widget->isEnabled()}, {"clipping_ancestors", clips}};
    if (auto *label = qobject_cast<QLabel *>(widget)) {
        const auto content = label->contentsRect();
        const int required = label->fontMetrics().boundingRect(QRect(0, 0, qMax(1, content.width()), 100000),
            label->wordWrap() ? Qt::TextWordWrap : 0, label->text()).height();
        const int layoutRequired = label->heightForWidth(label->width());
        const bool textFits = content.height() >= required && (layoutRequired < 0 || label->height() >= layoutRequired) &&
            (label->wordWrap() || label->fontMetrics().horizontalAdvance(label->text()) <= content.width());
        result.insert("text_geometry", QJsonObject{{"text", label->text()}, {"word_wrap", label->wordWrap()},
            {"contents_width", content.width()}, {"contents_height", content.height()}, {"font_line_height", label->fontMetrics().height()},
            {"required_text_height", required}, {"layout_height_for_width", layoutRequired}, {"actual_height", label->height()},
            {"unwrapped_text_width", label->fontMetrics().horizontalAdvance(label->text())}, {"text_fits", textFits}});
    }
    return result;
}
QJsonArray smokeSplitterSizes(QSplitter *splitter)
{
    QJsonArray sizes;
    if (splitter) for (int size : splitter->sizes()) sizes.append(size);
    return sizes;
}
} // namespace
namespace vision::ui::smoke
{
namespace
{
class EvaluationSmokeDriver final : public SmokeContext
{
  public:
    using SmokeContext::SmokeContext;
    void run(const QString &directory, const QString &requestPath);
};
}
void EvaluationSmokeDriver::run(const QString &directory, const QString &requestPath)
{
    const QString output = QFileInfo(directory).absoluteFilePath();
    const auto request = jsonFile(requestPath);
    if (!QDir().mkpath(output) || request.isEmpty()) {
        qCritical("Invalid evaluation smoke request/output"); window_.requestExit(2); return;
    }
    const auto done = std::make_shared<bool>(false);
    const bool verifyDecision = request.value("verify_decision_layout").toBool(false);
    const auto decision = std::make_shared<QJsonObject>();
    const auto inferenceStarts = std::make_shared<int>(0);
    const auto finish = [this, output, request, done, decision, verifyDecision, inferenceStarts](bool success, const QString &error) {
        if (*done) return;
        *done = true;
        QJsonObject summary{{"schema_version", 1}, {"success", success}, {"error", error},
            {"version", QCoreApplication::applicationVersion()}, {"request", request},
            {"report_directory", window_.evaluationState().directory}, {"report", window_.evaluationState().report}};
        if (verifyDecision) {
            decision->insert("completed", success);
            decision->insert("ordinary_inference_start_requests", *inferenceStarts);
            summary.insert("decision_layout", *decision);
        }
        QJsonArray exports;
        for (const auto &file : QDir(output + "/selected-result").entryInfoList(QDir::Files))
            exports.append(QJsonObject{{"path", file.absoluteFilePath()}, {"sha256", sha256(file.absoluteFilePath())}, {"size", file.size()}});
        summary.insert("exports", exports);
        QSaveFile f(output + "/evaluation-smoke-report.json");
        const auto bytes = QJsonDocument(summary).toJson();
        const bool saved = f.open(QIODevice::WriteOnly) && f.write(bytes) == bytes.size() && f.commit();
        if (!success) window_.saveScreenshot(output + "/failure.png");
        window_.requestExit(success && saved ? 0 : 1);
    };
    if (!QFileInfo::exists(request.value("model_path").toString()) ||
        !QFileInfo::exists(request.value("dataset_yaml").toString())) {
        finish(false, "Missing model or dataset YAML"); return;
    }
    window_.configuration()->setLabels({}, {}); ui_.autoExport->setChecked(false);
    window_.selectModel(request.value("model_path").toString());
    ui_.taskBox->setCurrentIndex(request.value("task").toString() == "yolov5" ? 0 : 1);
    ui_.inputSize->setValue(request.value("input_size").toInt(640));
    ui_.confidence->setValue(request.value("confidence").toDouble(.25));
    ui_.iou->setValue(request.value("nms_iou").toDouble(.45));
    const QString color = request.value("color_mode").toString("rgb");
    ui_.inputColorMode->setCurrentIndex(ui_.inputColorMode->findData(color == "grayscale" ? "grayscale" : color == "bgr" ? "bgr" : "rgb"));
    window_.setComputeDevice(request.value("device").toString() == "cuda" ? vision::ComputeDevice::CUDA : vision::ComputeDevice::CPU,
                     request.value("device_index").toInt());
    window_.openDataset(request.value("dataset_yaml").toString(), request.value("split").toString("val"), request.value("metadata_path").toString());
    auto *timer = new QTimer(this); timer->setInterval(40);
    if (verifyDecision) connect(&window_, &MainWindow::startRequested, timer,
        [inferenceStarts](const vision::JobRequest &) { ++*inferenceStarts; });
    const auto stage = std::make_shared<int>(0);
    connect(timer, &QTimer::timeout, this, [this, timer, stage, request, output, finish, done, decision, verifyDecision, inferenceStarts] {
        if (*done) { timer->stop(); return; }
        if (window_.isWorkbenchBusy() || window_.modelState().inspectionPending) return;
        const auto check = [decision, finish](const QString &name, bool value) {
            auto checks = decision->value("checks").toObject(); checks.insert(name, value); decision->insert("checks", checks);
            if (!value) finish(false, "Decision-layout acceptance failed: " + name);
            return value;
        };
        auto *tabs = ui_.evaluation->findChild<QTabWidget *>("evaluationAnalysisTabs");
        auto *table = ui_.evaluation->findChild<QTableWidget *>("evaluationThresholdTable");
        auto *curve = ui_.evaluation->findChild<QWidget *>("evaluationThresholdCurve");
        auto *recommend = ui_.evaluation->findChild<QPushButton *>("evaluationUseRecommendationButton");
        auto *review = ui_.evaluation->findChild<QPushButton *>("evaluationReviewErrorsButton");
        auto *apply = ui_.evaluation->findChild<QPushButton *>("evaluationApplyThresholdButton");
        auto *splitter = ui_.evaluation->findChild<QSplitter *>("evaluationDecisionSplit");
        auto *strategy = ui_.evaluation->findChild<QComboBox *>("evaluationRecommendationStrategyCombo");
        auto *constraint = ui_.evaluation->findChild<QDoubleSpinBox *>("evaluationRecommendationConstraintSpin");
        const auto point = [this, table, curve, tabs] {
            QJsonArray metrics;
            for (int index = 2; index < 6; ++index) {
                const auto *value = ui_.evaluation->findChild<QLabel *>(QString("evaluationMetric%1").arg(index));
                metrics.append(value ? value->text() : QString());
            }
            const auto *item = table ? table->item(table->currentRow(), 0) : nullptr;
            const QRect selected = item ? table->visualItemRect(item) : QRect();
            return QJsonObject{{"threshold", ui_.evaluation->analysisThreshold()},
                {"curve_threshold", curve ? curve->property("analysisThreshold").toDouble() : -1},
                {"recommended_threshold", curve ? curve->property("recommendedThreshold").toDouble() : -1},
                {"selected_row", table ? table->currentRow() : -1}, {"table_confidence", item ? item->data(Qt::UserRole).toDouble() : -1},
                {"selected_row_visible", item && table->viewport()->rect().contains(selected)},
                {"metrics", metrics}, {"analysis_tab", tabs ? tabs->currentIndex() : -1}};
        };
        const auto unchanged = [this, decision, inferenceStarts] {
            return vision::EvaluationWorker::modelConfigJson(window_.currentConfig()) == decision->value("config_before").toObject() &&
                sha256(window_.evaluationState().directory + "/report.json") == decision->value("report_sha256_before").toString() &&
                QJsonArray::fromStringList(QDir(window_.dataRoot() + "/evaluations").entryList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name)) ==
                    decision->value("evaluation_directories_before").toArray() &&
                QString::fromLatin1(QCryptographicHash::hash(QJsonDocument(window_.historySnapshot()).toJson(QJsonDocument::Compact), QCryptographicHash::Sha256).toHex()) ==
                    decision->value("history_sha256_before").toString() && *inferenceStarts == 0 &&
                !window_.isWorkbenchBusy() && !window_.evaluations()->isBusy();
        };
        if (*stage == 0) {
            if (window_.inputState().datasetManifest.isEmpty() || window_.modelState().classCount <= 0) {
                finish(false, "Dataset prepare or model inspection failed: " + status_->text()); return;
            }
            auto *controls = ui_.evaluation->controlsWidget();
            controls->findChild<QSpinBox *>("evaluationModelClassIdSpin")->setValue(request.value("model_class_id").toInt());
            controls->findChild<QSpinBox *>("evaluationDatasetClassIdSpin")->setValue(request.value("dataset_class_id").toInt());
            controls->findChild<QDoubleSpinBox *>("evaluationMatchIouSpin")->setValue(request.value("match_iou").toDouble(.5));
            controls->findChild<QDoubleSpinBox *>("evaluationScanStartSpin")->setValue(request.value("scan_start").toDouble(.05));
            controls->findChild<QDoubleSpinBox *>("evaluationScanEndSpin")->setValue(request.value("scan_end").toDouble(.95));
            controls->findChild<QDoubleSpinBox *>("evaluationScanStepSpin")->setValue(request.value("scan_step").toDouble(.01));
            controls->findChild<QSpinBox *>("evaluationBatchSizeSpin")->setValue(request.value("batch_size").toInt(1));
            controls->findChild<QSpinBox *>("evaluationCpuThreadsSpin")->setValue(request.value("cpu_threads").toInt());
            controls->findChild<QSpinBox *>("evaluationMaxDetectionsSpin")->setValue(request.value("max_detections").toInt(300));
            controls->findChild<QDoubleSpinBox *>("evaluationScoreFloorSpin")->setValue(request.value("score_floor").toDouble(.001));
            if (request.value("figure_formats").isArray()) {
                const auto formats = request.value("figure_formats").toArray();
                ui_.evaluation->findChild<QCheckBox *>("evaluationFigurePngCheck")->setChecked(formats.contains("png"));
                ui_.evaluation->findChild<QCheckBox *>("evaluationFigureSvgCheck")->setChecked(formats.contains("svg"));
                ui_.evaluation->findChild<QCheckBox *>("evaluationFigurePdfCheck")->setChecked(formats.contains("pdf"));
            }
            ui_.evaluation->findChild<QSpinBox *>("evaluationFigureDpiSpin")->setValue(request.value("figure_dpi").toInt(300));
            ui_.evaluation->setMode(true); ui_.detectionRange->setCurrentIndex(0); ui_.queue->setCurrentRow(0);
            if (!ui_.evaluation->canEvaluate() || !ui_.runButton->isEnabled()) {
                finish(false, "Cannot start evaluation: " + ui_.evaluation->validationMessage()); return;
            }
            ui_.workbenchTabs->setCurrentIndex(1);
            ui_.evaluationStartButton->click();
            if (!window_.isWorkbenchBusy() || !window_.evaluations()->isBusy()) { finish(false, "Workbench failed to start actual evaluation"); return; }
            *stage = 1;
        } else if (*stage == 1) {
            if (window_.evaluationState().report.isEmpty() || window_.evaluationState().report.value("status").toString() != "complete" ||
                window_.evaluationState().report.value("completed_images").toInt() != window_.inputState().files.size()) {
                finish(false, "Incomplete evaluation: " + status_->text()); return;
            }
            if (verifyDecision) {
                if (!check("required_widgets_exist", tabs && table && curve && recommend && review && apply && splitter && strategy && constraint)) return;
                decision->insert("schema_version", 1);
                decision->insert("config_before", vision::EvaluationWorker::modelConfigJson(window_.currentConfig()));
                decision->insert("report_sha256_before", sha256(window_.evaluationState().directory + "/report.json"));
                decision->insert("evaluation_directories_before", QJsonArray::fromStringList(QDir(window_.dataRoot() + "/evaluations").entryList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name)));
                decision->insert("history_sha256_before", QString::fromLatin1(QCryptographicHash::hash(QJsonDocument(window_.historySnapshot()).toJson(QJsonDocument::Compact), QCryptographicHash::Sha256).toHex()));
                decision->insert("initial_threshold", ui_.evaluation->analysisThreshold());
                decision->insert("initial_filter", ui_.evaluation->imageFilter());
                decision->insert("initial_point", point());
                QJsonArray titles;
                for (int index = 0; index < tabs->count(); ++index) titles.append(tabs->tabText(index));
                decision->insert("tabs", titles);
                if (!check("four_pages_and_order", titles == QJsonArray{"阈值调优", "能力诊断", "场景分析", "报告管理"})) return;
                auto *page = ui_.evaluation->findChild<QWidget *>("evaluationThresholdPage");
                if (!check("decision_controls_share_threshold_page", page && page->isAncestorOf(table) && page->isAncestorOf(curve) &&
                    page->isAncestorOf(recommend) && page->isAncestorOf(review) && page->isAncestorOf(apply) &&
                    page->isAncestorOf(ui_.evaluation->findChild<QWidget *>("evaluationCurrentPointCard")))) return;
                tabs->setCurrentIndex(0); ui_.workbenchTabs->setCurrentIndex(1); window_.resize(1260, 820); *stage = 10; return;
            }
            window_.saveScreenshot(output + "/evaluation-analysis.png");
            window_.saveScreenshot(output + "/evaluation-thresholds.png");
            if (auto *tabs = ui_.evaluation->findChild<QTabWidget *>("evaluationAnalysisTabs")) {
                const QStringList names{"thresholds", "diagnostics", "scenes", "report"};
                for (int page = 1; page < tabs->count(); ++page) {
                    tabs->setCurrentIndex(page);
                    window_.saveScreenshot(output + "/evaluation-" + names.value(page, QString::number(page)) + ".png");
                }
                tabs->setCurrentIndex(0);
            }
            ui_.workbenchTabs->setCurrentIndex(0); ui_.queue->setCurrentRow(0); window_.reviewEvaluationRow(0); *stage = 2;
        } else if (*stage == 10 || *stage == 11) {
            const int expectedWidth = *stage == 10 ? 1260 : 1366, expectedHeight = *stage == 10 ? 820 : 900;
            const QString sizeName = QString::number(expectedWidth) + "x" + QString::number(expectedHeight);
            const QStringList geometryStrategies{"f1", "precision", "fpr"};
            const int strategyIndex = decision->value("geometry_strategy_index").toInt();
            const QString strategyName = geometryStrategies.value(strategyIndex);
            const QString suffix = strategyName == "f1" ? QString() : "_" + strategyName;
            QJsonObject widgets;
            bool visible = window_.width() == expectedWidth && window_.height() == expectedHeight && strategy->currentData() == strategyName;
            QStringList geometryWidgets{"evaluationAnalysisThresholdSpin", "evaluationThresholdCurve", "evaluationThresholdTable",
                "evaluationDecisionSidebar", "evaluationRecommendationStrategyCombo", "evaluationUseRecommendationButton",
                "evaluationReviewErrorsButton", "evaluationApplyThresholdButton", "evaluationRecommendationHint", "evaluationCounts",
                "evaluationMetric2", "evaluationMetric3", "evaluationMetric4", "evaluationMetric5"};
            if (strategyName != "f1") geometryWidgets.append("evaluationRecommendationConstraintSpin");
            for (const auto &name : geometryWidgets) {
                const auto geometry = smokeWidgetGeometry(ui_.evaluation->findChild<QWidget *>(name), &window_);
                widgets.insert(name, geometry); visible = visible && geometry.value("fully_visible").toBool() &&
                    (!geometry.contains("text_geometry") || geometry.value("text_geometry").toObject().value("text_fits").toBool());
            }
            widgets.insert("window", QJsonObject{{"width", window_.width()}, {"height", window_.height()}});
            const auto *selectedItem = table->item(table->currentRow(), 0);
            const QRect rowRect = selectedItem ? table->visualItemRect(selectedItem) : QRect();
            const bool rowVisible = selectedItem && table->viewport()->rect().contains(rowRect);
            widgets.insert("selected_row", QJsonObject{{"row", table->currentRow()},
                {"confidence", selectedItem ? selectedItem->data(Qt::UserRole).toDouble() : -1}, {"fully_visible", rowVisible},
                {"x", rowRect.x()}, {"y", rowRect.y()}, {"width", rowRect.width()}, {"height", rowRect.height()},
                {"viewport_width", table->viewport()->width()}, {"viewport_height", table->viewport()->height()}});
            visible = visible && rowVisible;
            // Observe settled asynchronous Qt layouts; never scroll/select on
            // behalf of the UI under test. Two identical consecutive timer
            // observations are required, with all originals retained.
            auto waits = decision->value("geometry_waits").toObject();
            const QString geometryKey = sizeName + suffix;
            auto waiting = waits.value(geometryKey).toObject();
            auto observations = waiting.value("observations").toArray(); observations.append(widgets);
            waiting.insert("observations", observations); waiting.insert("ticks", observations.size());
            const bool stable = observations.size() >= 2 && observations.at(observations.size() - 2).toObject() == widgets;
            waiting.insert("stable", stable); waits.insert(geometryKey, waiting); decision->insert("geometry_waits", waits);
            if (!stable) {
                if (observations.size() >= 10) { finish(false, "Decision layout did not stabilize within 10 GUI timer observations: " + geometryKey); }
                return;
            }
            auto geometries = decision->value("window_geometries").toObject(); geometries.insert(sizeName + suffix, widgets); decision->insert("window_geometries", geometries);
            window_.saveScreenshot(output + "/decision-" + sizeName + suffix + ".png");
            if (!check("visible_at_" + sizeName + suffix, visible)) return;
            if (strategyIndex < 2) {
                decision->insert("geometry_strategy_index", strategyIndex + 1);
                strategy->setCurrentIndex(strategy->findData(geometryStrategies[strategyIndex + 1])); return;
            }
            decision->insert("geometry_strategy_index", 0); strategy->setCurrentIndex(strategy->findData("f1"));
            if (*stage == 10) { window_.resize(1366, 900); *stage = 11; return; }
            decision->insert("splitter_before", smokeSplitterSizes(splitter));
            const auto sizes = splitter->sizes();
            if (!check("splitter_has_two_noncollapsed_sections", sizes.size() == 2 && sizes[0] > 30 && sizes[1] > 60)) return;
            splitter->setSizes({sizes[0] - 20, sizes[1] + 20}); *stage = 12;
        } else if (*stage == 12) {
            decision->insert("splitter_resized", smokeSplitterSizes(splitter));
            if (!check("splitter_resizes_chart_and_table", decision->value("splitter_before") != decision->value("splitter_resized"))) return;
            const double recommended = curve->property("recommendedThreshold").toDouble();
            if (!check("real_full_val_recommendation_available", recommend->isEnabled() && recommended >= 0 && strategy->currentData() == "f1")) return;
            // Display both distinct markers before the actual button selects it.
            const int other = std::abs(table->item(0, 0)->data(Qt::UserRole).toDouble() - recommended) < 1e-8 ? table->rowCount() - 1 : 0;
            table->setCurrentCell(other, 0); *stage = 13;
        } else if (*stage == 13) {
            if (window_.evaluationState().review.pending) return;
            decision->insert("distinct_markers", point());
            if (!check("current_and_recommended_markers_are_distinct", std::abs(ui_.evaluation->analysisThreshold() - curve->property("recommendedThreshold").toDouble()) > 1e-8 &&
                std::abs(ui_.evaluation->analysisThreshold() - curve->property("analysisThreshold").toDouble()) < 1e-8)) return;
            if (!check("configuration_unchanged_before_recommendation", unchanged())) return;
            window_.saveScreenshot(output + "/decision-distinct-markers.png"); recommend->click(); *stage = 14;
        } else if (*stage == 14) {
            if (window_.evaluationState().review.pending) return;
            const auto observed = point(); decision->insert("recommended_point", observed);
            const double threshold = observed.value("threshold").toDouble();
            if (!check("recommendation_click_syncs_curve_table_and_visible_row", observed.value("selected_row_visible").toBool() &&
                std::abs(threshold - observed.value("recommended_threshold").toDouble()) < 1e-8 &&
                std::abs(threshold - observed.value("table_confidence").toDouble()) < 1e-8 &&
                std::abs(threshold - observed.value("curve_threshold").toDouble()) < 1e-8)) return;
            if (!check("recommendation_does_not_adopt_or_reinfer", unchanged())) return;
            window_.saveScreenshot(output + "/decision-recommended-point.png");
            table->setCurrentCell(table->rowCount() - 1, 0); *stage = 15;
        } else if (*stage == 15) {
            if (window_.evaluationState().review.pending) return;
            const auto observed = point(); decision->insert("table_selected_point", observed);
            if (!check("table_selection_syncs_current_point_and_visible_row", observed.value("selected_row_visible").toBool() &&
                std::abs(observed.value("threshold").toDouble() - observed.value("table_confidence").toDouble()) < 1e-8 &&
                std::abs(observed.value("threshold").toDouble() - observed.value("curve_threshold").toDouble()) < 1e-8)) return;
            decision->insert("review_threshold", ui_.evaluation->analysisThreshold()); decision->insert("review_tab", tabs->currentIndex());
            decision->insert("review_strategy", strategy->currentData().toString()); decision->insert("review_constraint", constraint->value());
            decision->insert("review_splitter", smokeSplitterSizes(splitter));
            if (!check("real_review_button_enabled", review->isEnabled())) return;
            review->click(); *stage = 16;
        } else if (*stage == 16) {
            if (window_.evaluationState().review.pending) return;
            if (!check("review_button_opens_real_error_queue", ui_.workbenchTabs->currentIndex() == 0 && window_.evaluationState().review.active &&
                window_.evaluationState().review.available && window_.evaluationState().review.filter == "errors" && window_.evaluationState().review.evaluated == window_.inputState().files.size() &&
                window_.evaluationState().review.rows == window_.evaluationState().review.errorRows)) return;
            if (!window_.evaluationState().review.rows.isEmpty() && (window_.displayedResult().image.isNull() || !window_.findChild<QPushButton *>("exportResultButton")->isEnabled())) return;
            decision->insert("review", QJsonObject{{"errors", window_.evaluationState().review.errorRows.size()}, {"evaluated", window_.evaluationState().review.evaluated},
                {"collection", window_.inputState().datasetManifest.value("samples").toArray().size()}, {"queue_row", ui_.queue->currentRow()},
                {"visible_rows", QJsonArray::fromVariantList([this] { QVariantList rows; for (int row : window_.evaluationState().review.rows) rows.append(row); return rows; }())},
                {"preview_available", !window_.displayedResult().image.isNull()}, {"filter", window_.evaluationState().review.filter}});
            window_.saveScreenshot(output + "/decision-error-review.png");
            auto *back = window_.findChild<QPushButton *>("evaluationReviewAnalysisButton");
            if (!check("real_return_button_visible", back && back->isVisibleTo(&window_) && back->isEnabled())) return;
            back->click(); *stage = 17;
        } else if (*stage == 17) {
            const auto observed = point(); decision->insert("after_review_point", observed);
            if (!check("return_preserves_threshold_page_strategy_and_splitter", ui_.workbenchTabs->currentIndex() == 1 &&
                tabs->currentIndex() == decision->value("review_tab").toInt() &&
                std::abs(ui_.evaluation->analysisThreshold() - decision->value("review_threshold").toDouble()) < 1e-8 &&
                strategy->currentData().toString() == decision->value("review_strategy").toString() &&
                std::abs(constraint->value() - decision->value("review_constraint").toDouble()) < 1e-8 &&
                smokeSplitterSizes(splitter) == decision->value("review_splitter").toArray() && ui_.evaluation->imageFilter() == "errors")) return;
            if (!check("review_does_not_adopt_reinfer_or_rewrite_report", unchanged())) return;
            decision->insert("returned_review", QJsonObject{{"threshold", ui_.evaluation->analysisThreshold()}, {"analysis_tab", tabs->currentIndex()},
                {"strategy", strategy->currentData().toString()}, {"constraint", constraint->value()},
                {"splitter", smokeSplitterSizes(splitter)}, {"filter", ui_.evaluation->imageFilter()}});
            decision->insert("history_sha256_before_apply", QString::fromLatin1(QCryptographicHash::hash(QJsonDocument(window_.historySnapshot()).toJson(QJsonDocument::Compact), QCryptographicHash::Sha256).toHex()));
            decision->insert("evaluation_directories_before_apply", QJsonArray::fromStringList(QDir(window_.dataRoot() + "/evaluations").entryList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name)));
            decision->insert("config_before_explicit_apply", vision::EvaluationWorker::modelConfigJson(window_.currentConfig()));
            decision->insert("apply_button_enabled", apply->isEnabled());
            if (!check("explicit_apply_allowed_for_verified_native_report", apply->isEnabled() && window_.evaluationState().report.value("allow_apply_profile").toBool(true))) return;
            apply->click(); *stage = 18;
        } else if (*stage == 18) {
            const auto profile = window_.modelState().evaluationProfiles.value(window_.modelState().activeModel).toMap();
            decision->insert("config_after_explicit_apply", vision::EvaluationWorker::modelConfigJson(window_.currentConfig()));
            decision->insert("adopted_threshold", ui_.confidence->value());
            if (!check("only_explicit_apply_updates_detection_profile", profile.value("adopted").toBool() &&
                std::abs(profile.value("adopted_threshold").toDouble() - decision->value("review_threshold").toDouble()) < 1e-7 &&
                std::abs(ui_.confidence->value() - decision->value("review_threshold").toDouble()) < 1e-7 &&
                ui_.classFilter->text() == QString::number(request.value("model_class_id").toInt()) &&
                profile.value("adopted_report_directory").toString() == window_.evaluationState().directory &&
                sha256(window_.evaluationState().directory + "/report.json") == decision->value("report_sha256_before").toString() && !window_.isWorkbenchBusy() && !window_.evaluations()->isBusy())) return;
            // Return to the original smoke export contract: first image at the
            // request's original analysis threshold, irrespective of review.
            ui_.evaluation->setAnalysisThreshold(decision->value("initial_threshold").toDouble());
            ui_.evaluation->setImageFilter("all"); *stage = 19;
        } else if (*stage == 19) {
            if (window_.evaluationState().review.pending) return;
            decision->insert("final_point", point());
            decision->insert("report_sha256_after", sha256(window_.evaluationState().directory + "/report.json"));
            if (!check("original_report_preserved_and_initial_export_threshold_restored", decision->value("report_sha256_before") == decision->value("report_sha256_after") &&
                std::abs(ui_.evaluation->analysisThreshold() - decision->value("initial_threshold").toDouble()) < 1e-8 && ui_.evaluation->imageFilter() == "all")) return;
            window_.saveScreenshot(output + "/evaluation-analysis.png"); window_.saveScreenshot(output + "/evaluation-thresholds.png");
            const QStringList names{"thresholds", "diagnostics", "scenes", "report"};
            for (int index = 1; index < tabs->count(); ++index) { tabs->setCurrentIndex(index); window_.saveScreenshot(output + "/evaluation-" + names[index] + ".png"); }
            tabs->setCurrentIndex(0); ui_.workbenchTabs->setCurrentIndex(0); ui_.queue->setCurrentRow(0); window_.reviewEvaluationRow(0); *stage = 2;
        } else if (*stage == 2 && !window_.displayedResult().image.isNull() && window_.findChild<QPushButton *>("exportResultButton")->isEnabled()) {
            QString error;
            if (!window_.exportDisplayedResult(output + "/selected-result", nullptr, &error)) { finish(false, error); return; }
            window_.saveScreenshot(output + "/evaluation-preview.png");
            finish(true, {});
        }
    });
    timer->start();
    QTimer::singleShot(240000, this, [this, finish, done] {
        if (*done) return;
        if (window_.evaluations()->isBusy()) window_.evaluations()->requestStop();
        finish(false, "Evaluation smoke timed out: " + status_->text());
    });
}


void runEvaluationSmoke(MainWindow &window, const QString &directory, const QString &requestPath)
{
    auto *driver = new EvaluationSmokeDriver(window);
    if (!driver->hasWorkbench()) {
        qCritical("Missing actual workbench widgets for evaluation smoke");
        window.requestExit(2); return;
    }
    driver->run(directory, requestPath);
}
} // namespace vision::ui::smoke
