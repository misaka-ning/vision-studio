// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <functional>
class QWidget;
namespace ui {
// Returns true when the caller can continue now. Otherwise queues a retry that
// runs only after the shared, explicitly started preparation succeeds.
bool ensureCpuRuntime(QWidget *parent, std::function<void()> retry);
}
