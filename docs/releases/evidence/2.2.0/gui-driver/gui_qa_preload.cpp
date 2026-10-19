// SPDX-License-Identifier: AGPL-3.0-only
// Standalone acceptance driver for Vision Studio 2.2.0. Uses Qt public UI APIs
// only; it does not link application classes or inference/result functions.
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QMimeData>
#include <QProgressBar>
#include <QPushButton>
#include <QSaveFile>
#include <QSet>
#include <QTableWidget>
#include <QTimer>
#include <QUrl>
#include <dlfcn.h>
#include <algorithm>

namespace {
QByteArray digest(const QString &path) {
    QFile file(path); if (!file.open(QIODevice::ReadOnly)) return {};
    QCryptographicHash hash(QCryptographicHash::Sha256);
    hash.addData(&file); return hash.result().toHex();
}
QPushButton *caption(QWidget *page, const QString &name) {
    for (auto *b : page->findChildren<QPushButton *>()) if (b->text() == name) return b;
    return nullptr;
}
struct Scenario { QString name; int count, mode, row; QString filter; bool stop = false, invalid = false; };
class Driver final : public QObject {
    QString output_, input_, report_, resultDirectory_;
    QWidget *window_ = nullptr, *page_ = nullptr, *canvas_ = nullptr;
    QListWidget *queue_ = nullptr;
    QComboBox *range_ = nullptr;
    QLineEdit *filter_ = nullptr;
    QPushButton *run_ = nullptr, *stop_ = nullptr, *clear_ = nullptr, *export_ = nullptr;
    QProgressBar *progress_ = nullptr;
    QTableWidget *table_ = nullptr;
    QTimer timer_;
    QElapsedTimer clock_, phaseClock_, stopClock_;
    QVector<Scenario> scenarios_;
    int current_ = -1, stage_ = 0, acceptedAtStop_ = -1;
    qint64 previousTick_ = 0, maximumGap_ = 0;
    QStringList allInputs_, expected_;
    QSet<QString> beforeFiles_;
    QJsonArray observations_;
    bool done_ = false;

    bool check(bool condition, const QString &message) {
        if (!condition) finish(false, message); return condition;
    }
    QSet<QString> jsonFiles() const {
        QSet<QString> result;
        for (const auto &path : QDir(resultDirectory_).entryList({"*.json"}, QDir::Files)) result.insert(path);
        return result;
    }
    void finish(bool success, const QString &error = {}) {
        if (done_) return;
        done_ = true; timer_.stop();
        QJsonObject report{{"schema_version",1}, {"application_version",QCoreApplication::applicationVersion()},
            {"qt",qVersion()}, {"success",success}, {"error",error}, {"cases",observations_},
            {"executable",QCoreApplication::applicationFilePath()},
            {"executable_sha256",QString::fromLatin1(digest(QCoreApplication::applicationFilePath()))},
            {"maximum_gui_heartbeat_gap_ms",maximumGap_}, {"elapsed_ms",clock_.elapsed()},
            {"driver","LD_PRELOAD QApplication::exec + public Qt widgets/drop events"},
            {"recorded_at",QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs)}};
        QSaveFile file(report_);
        const bool written = file.open(QIODevice::WriteOnly) && file.write(QJsonDocument(report).toJson()) > 0 && file.commit();
        qInfo().noquote() << "installed_gui_qa:" << QJsonDocument(report).toJson(QJsonDocument::Compact);
        if (window_ && !success && stop_ && stop_->isVisible()) stop_->click();
        QTimer::singleShot(250, qApp, [success,written] { QCoreApplication::exit(success && written ? 0 : 42); });
    }
    bool drop(const QString &file) {
        QMimeData mime; mime.setUrls({QUrl::fromLocalFile(file)});
        const QPoint point = canvas_->rect().center();
        QDragEnterEvent enter(point,Qt::CopyAction,&mime,Qt::LeftButton,Qt::NoModifier);
        QApplication::sendEvent(canvas_, &enter);
        if (!enter.isAccepted()) return false;
        QDropEvent event(point,Qt::CopyAction,&mime,Qt::LeftButton,Qt::NoModifier);
        QApplication::sendEvent(canvas_, &event); return event.isAccepted();
    }
    void beginNext() {
        if (++current_ >= scenarios_.size()) { finish(true); return; }
        const auto &s = scenarios_[current_];
        filter_->clear(); range_->setCurrentIndex(0); clear_->click();
        for (int i = 0; i < s.count; ++i) if (!check(drop(allInputs_[i]), "Image drop rejected")) return;
        if (!check(queue_->count() == s.count, "Imported queue count mismatch")) return;
        range_->setCurrentIndex(s.mode); queue_->setCurrentRow(s.row); filter_->setText(s.filter);
        beforeFiles_ = jsonFiles();
        expected_ = s.mode == 0 ? allInputs_.mid(0,s.count) : s.mode == 1 ? allInputs_.mid(s.row,s.count-s.row) : QStringList{allInputs_[s.row]};
        stage_ = 1; acceptedAtStop_ = -1; phaseClock_.restart();
        qInfo().noquote() << "installed_gui_case:" << s.name;
    }
    void collectCase() {
        const auto &s = scenarios_[current_];
        const QSet<QString> newFiles = jsonFiles() - beforeFiles_;
        const int processed = progress_->value();
        if (!check(progress_->maximum() == expected_.size(), "Task progress maximum did not match fixed range")) return;
        if (!check(newFiles.size() == processed, "Automatic export count did not match accepted progress")) return;
        if (!check(s.stop ? processed > 0 && processed < expected_.size() : processed == expected_.size(), "Unexpected completed count")) return;
        QSet<QString> actualSources;
        QJsonArray ids;
        int totalPredictions = 0;
        for (const QString &name : newFiles) {
            QFile file(QDir(resultDirectory_).filePath(name));
            if (!check(file.open(QIODevice::ReadOnly), "Cannot read actual exported JSON")) return;
            const auto object = QJsonDocument::fromJson(file.readAll()).object();
            const QString source = object["source"].toString();
            if (!check(expected_.contains(source), "Exported a source outside the frozen task range")) return;
            if (!check(!actualSources.contains(source), "Duplicate source exported in a task")) return;
            actualSources.insert(source);
            if (!check(object["actual_device"].toString() == "cpu", "Expected actual CPU inference")) return;
            const auto predictions = object["predictions"].toArray();
            totalPredictions += predictions.size();
            if (s.filter == "32" && !check(predictions.isEmpty(), "Real bus fixture unexpectedly matched class 32")) return;
            for (const auto &value : predictions) {
                const int id = value.toObject()["class_id"].toInt(-1);
                if (!check(s.filter.isEmpty() || s.filter == "32" ? s.filter.isEmpty() || id == 32 : id == 0 || id == 5,
                           "Filter leaked a different original class ID")) return;
                if (!ids.contains(id)) ids.append(id);
            }
            queue_->setCurrentRow(allInputs_.indexOf(source));
            if (!check(export_->isEnabled(), "Completed selected result is not exportable")) return;
            if (!check(table_->rowCount() == predictions.size(), "Reviewed target list differs from JSON predictions")) return;
        }
        for (int row = 0; row < s.count; ++row) if (!actualSources.contains(allInputs_[row])) {
            queue_->setCurrentRow(row);
            if (!check(!export_->isEnabled() && table_->rowCount() == 0, "Unprocessed image incorrectly reused a detection result")) return;
        }
        QJsonObject record{{"name",s.name}, {"queue_count",s.count}, {"mode",s.mode}, {"selected_row",s.row},
            {"submitted_count",expected_.size()}, {"completed",processed}, {"exported_json",newFiles.size()},
            {"filter",s.filter}, {"original_class_ids",ids}, {"total_predictions",totalPredictions},
            {"review_matches_export",true}, {"outside_range_unprocessed",true}, {"success",true}};
        if (s.stop) { record.insert("stop_elapsed_ms",stopClock_.elapsed()); record.insert("progress_no_growth_after_stop",true); }
        observations_.append(record);
        beginNext();
    }
    void tick() {
        const qint64 now = clock_.elapsed(); maximumGap_ = std::max(maximumGap_,now-previousTick_); previousTick_=now;
        for (QWidget *widget : QApplication::topLevelWidgets())
            if (auto *dialog = qobject_cast<QMessageBox *>(widget); dialog && dialog->isVisible()) {
                finish(false,"Unexpected dialog: " + dialog->text()); dialog->accept(); return;
            }
        if (phaseClock_.elapsed() > 120000) { finish(false,"Scenario timeout"); return; }
        if (stage_ == 0) { beginNext(); return; }
        const auto &s = scenarios_[current_];
        if (stage_ == 1) {
            if (s.invalid) {
                if (!check(!run_->isEnabled(),"Invalid filter did not disable Start")) return;
                auto *hint=window_->findChild<QLabel *>("classFilterHint");
                if (!check(hint && !hint->text().isEmpty(),"Invalid filter lacks inline hint")) return;
                const auto countBefore=jsonFiles().size(); run_->click();
                if (!check(!stop_->isVisible() && countBefore==jsonFiles().size(),"Invalid filter started a task")) return;
                observations_.append(QJsonObject{{"name",s.name},{"filter",s.filter},{"invalid_start_blocked",true},{"inline_hint",hint->text()},{"success",true}});
                beginNext(); return;
            }
            if (!run_->isEnabled()) return; // Native async model validation and result writer settle here.
            run_->click();
            if (!check(stop_->isVisible() && !range_->isEnabled() && !queue_->isEnabled(),"Task did not lock range and queue")) return;
            stage_=2; phaseClock_.restart(); return;
        }
        if (stage_ == 2) {
            if (s.stop && acceptedAtStop_<0 && progress_->value()>0) {
                acceptedAtStop_=progress_->value(); stopClock_.start(); stop_->click();
            }
            if (!run_->isVisible() || !run_->isEnabled()) return;
            if (s.stop) { acceptedAtStop_=progress_->value(); stage_=3; phaseClock_.restart(); }
            else collectCase();
        } else if (stage_==3 && phaseClock_.elapsed()>250) {
            if (!check(progress_->value()==acceptedAtStop_,"Progress grew after the stopped task completed")) return;
            collectCase();
        }
    }
public:
    Driver(QString output, QString input) : output_(std::move(output)),input_(std::move(input)) {
        report_=QDir(output_).filePath("installed-gui-qa.json");
        clock_.start(); phaseClock_.start();
        if (QFileInfo::exists(report_)) { qCritical()<<"Refusing to overwrite GUI QA report"; QCoreApplication::exit(43); return; }
        if (!QDir().mkpath(output_)) { QCoreApplication::exit(43); return; }
        for (QWidget *w : QApplication::topLevelWidgets()) if (w->findChild<QWidget *>("workbenchPage")) { window_=w; break; }
        if (!check(window_!=nullptr,"Vision Studio main window not found")) return;
        page_=window_->findChild<QWidget *>("workbenchPage");
        queue_=window_->findChild<QListWidget *>("inputQueue"); range_=window_->findChild<QComboBox *>("detectionRange");
        filter_=window_->findChild<QLineEdit *>("classFilter"); run_=window_->findChild<QPushButton *>("startDetectionButton");
        stop_=caption(page_,"停止运行"); clear_=caption(page_,"清空"); export_=window_->findChild<QPushButton *>("exportResultButton");
        progress_=window_->findChild<QProgressBar *>("detectionProgress");
        for (QWidget *w : page_->findChildren<QWidget *>()) if (QString::fromLatin1(w->metaObject()->className())=="ImageCanvas") { canvas_=w; break; }
        for (auto *t : page_->findChildren<QTableWidget *>()) if (t->columnCount()==3) { table_=t; break; }
        if (!check(queue_&&range_&&filter_&&run_&&stop_&&clear_&&export_&&progress_&&canvas_&&table_,"Required 2.2.0 UI controls missing")) return;
        for (auto *c : page_->findChildren<QCheckBox *>()) if (c->text()=="自动保存每张图片的结果") c->setChecked(true);
        const QString data=qEnvironmentVariable("VISION_STUDIO_DATA_DIR");
        if (!check(!data.isEmpty()&&QDir::isAbsolutePath(data),"Explicit isolated absolute data directory required")) return;
        resultDirectory_=QDir(data).filePath("results");
        for (const auto &file : QDir(input_).entryInfoList({"*.png","*.jpg"},QDir::Files,QDir::Name)) allInputs_.append(file.absoluteFilePath());
        if (!check(allInputs_.size()>=61,"At least 61 real image fixtures required")) return;
        scenarios_={
            {"whole-folder-3-selected-middle",3,0,1,""},
            {"tail-3-selected-middle",3,1,1,""},
            {"current-3-selected-middle",3,2,1,""},
            {"whole-folder-61-selected-51",61,0,50,"0,5"},
            {"tail-61-selected-51",61,1,50,"0,5"},
            {"current-61-selected-51",61,2,50,"0,5"},
            {"class32-zero-match",3,0,0,"32"},
            {"negative-id-blocked",3,0,0,"-1",false,true},
            {"out-of-range-id-blocked",3,0,0,"80",false,true},
            {"stop-tail-61-selected-11",61,1,10,"0,5",true,false}
        };
        connect(&timer_,&QTimer::timeout,this,[this]{tick();}); timer_.start(10);
    }
};
}
int QApplication::exec() {
    using Function=int(*)();
    auto original=reinterpret_cast<Function>(dlsym(RTLD_NEXT,"_ZN12QApplication4execEv"));
    if (!original) { qFatal("Cannot resolve original QApplication::exec"); }
    const QString output=qEnvironmentVariable("VISION_STUDIO_GUI_QA_DIR");
    const QString input=qEnvironmentVariable("VISION_STUDIO_GUI_QA_INPUTS");
    if (!output.isEmpty()) {
        if (QCoreApplication::applicationVersion()!="2.2.0" || !QDir::isAbsolutePath(output) || !QDir::isAbsolutePath(input)) {
            qCritical()<<"GUI acceptance helper only accepts Vision Studio 2.2.0 and explicit absolute QA paths"; return 43;
        }
        QTimer::singleShot(0,qApp,[output,input]{ auto *driver=new Driver(output,input); driver->setParent(qApp); });
    }
    return original();
}
