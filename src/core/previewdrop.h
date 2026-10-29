// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <QStringList>
namespace vision
{
enum class PreviewDropKind { Images, Dataset, Video };
// Classifies an external input drop before any session mutation. Large image
// batches retain their asynchronous validation path in InputSessionController.
bool classifyPreviewDrop(const QStringList &input, QStringList *paths,
                         PreviewDropKind *kind, QString *error);
}
