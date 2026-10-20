// SPDX-License-Identifier: AGPL-3.0-only
#include "evaluationmatch.h"
#include <algorithm>
#include <map>
#include <set>

namespace vision
{
void matchEvaluation(InferenceResult &result, double threshold)
{
    result.hasEvaluationMatches = true;
    result.matchedPredictions.clear();
    result.matchedGroundTruth.clear();
    result.matchingIou = threshold;
    struct Pair { int gt, pred; double iou; };
    QVector<Pair> pairs;
    for (int g = 0; g < result.groundTruth.size(); ++g)
        for (int p = 0; p < result.predictions.size(); ++p)
        {
            if (result.ignoredPredictions.contains(p)) continue;
            const auto a = result.groundTruth[g].box, b = result.predictions[p].box;
            const auto overlap = a.intersected(b);
            const double intersection = std::max(0., overlap.width()) * std::max(0., overlap.height());
            const double iou = intersection / (a.width() * a.height() + b.width() * b.height() - intersection + 1e-7);
            if (iou >= threshold) pairs.append({g, p, iou});
        }
    std::sort(pairs.begin(), pairs.end(), [](const Pair &a, const Pair &b) {
        if (a.iou != b.iou) return a.iou > b.iou;
        if (a.pred != b.pred) return a.pred < b.pred;
        return a.gt < b.gt;
    });
    // Reproduce the evaluator's unique(prediction), then unique(GT) order.
    std::map<int, Pair> predictions;
    for (const auto &pair : pairs) predictions.emplace(pair.pred, pair);
    std::set<int> groundTruth;
    for (const auto &entry : predictions)
        if (groundTruth.insert(entry.second.gt).second)
        {
            result.matchedPredictions.append(entry.second.pred);
            result.matchedGroundTruth.append(entry.second.gt);
        }
}
}
