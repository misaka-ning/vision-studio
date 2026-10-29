// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

namespace ui
{
// Values preserve the CLI/smoke and historical stacked-page mapping.
enum class Route
{
    Workbench = 0,
    Models = 1,
    ModelDisplay = 2,
    History = 3,
    Guide = 4,
    Recordings = 5,
    ModelConversion = 6,
    More = 7
};
constexpr int routeIndex(Route route) noexcept { return static_cast<int>(route); }
}
