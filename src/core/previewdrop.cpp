// SPDX-License-Identifier: AGPL-3.0-only
#include "previewdrop.h"
#include <QFileInfo>
#include <QImageReader>
#include <QSet>
#include <algorithm>
namespace vision
{
bool classifyPreviewDrop(const QStringList &input, QStringList *paths, PreviewDropKind *kind, QString *error)
{
    paths->clear();
    if (input.isEmpty()) { *error = "请拖入本地文件。"; return false; }
    const QStringList videos{"mp4", "avi", "mkv", "mov", "webm", "m4v"};
    QSet<QString> imageFormats;
    for (const auto &format : QImageReader::supportedImageFormats()) imageFormats.insert(QString::fromLatin1(format).toLower());
    // Drag-move events must not stat thousands of files on the GUI thread.
    // Actual files, readability and canonical identities are checked atomically
    // by importDroppedImages before any queue change.
    if (input.size() > 96 && std::all_of(input.begin(), input.end(), [&imageFormats](const QString &path) {
            return imageFormats.contains(QFileInfo(path).suffix().toLower()); })) {
        *paths = input; *kind = PreviewDropKind::Images; return true;
    }
    QSet<QString> seen;
    int category = -1;
    for (const QString &path : input) {
        const QFileInfo info(path);
        if (!info.exists() || !info.isReadable() || (!info.isFile() && !info.isDir())) {
            *error = "无法读取拖入的路径：" + path; return false;
        }
        const QString suffix = info.suffix().toLower();
        const int next = info.isDir() || suffix == "yaml" || suffix == "yml" ? 1 :
            videos.contains(suffix) ? 2 : imageFormats.contains(suffix) ? 0 : -1;
        if (next < 0) { *error = "不支持此文件。请拖入图片、视频或 YOLO 数据集目录 / YAML：" + path; return false; }
        if (category >= 0 && category != next) {
            *error = "一次拖入请只包含图片，或一个数据集目录 / YAML，或一个视频；不能混合输入类型。"; return false;
        }
        category = next;
        const QString canonical = info.canonicalFilePath();
        const QString absolute = next == 0 && !canonical.isEmpty() ? canonical : info.absoluteFilePath();
        if (!seen.contains(absolute)) { seen.insert(absolute); paths->append(absolute); }
    }
    if (category != 0 && input.size() != 1) {
        *error = "数据集或视频每次只能拖入一个；多张图片可以一起拖入。"; return false;
    }
    *kind = category == 1 ? PreviewDropKind::Dataset : category == 2 ? PreviewDropKind::Video : PreviewDropKind::Images;
    return true;
}
}
