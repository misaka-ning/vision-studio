// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include "visiontypes.h"

namespace vision
{
// Single selected evaluation class. Original model/GT IDs remain independent.
// Stable ties: IoU descending, prediction index ascending, GT index ascending.
void matchEvaluation(InferenceResult &result, double threshold);
}
