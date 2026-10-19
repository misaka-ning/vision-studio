# Vision Studio 2.2.0 standalone GUI acceptance helper

This helper does not link Vision Studio classes, inference methods or result objects. It intercepts the Qt `QApplication::exec` entry point with `LD_PRELOAD`, then drives the existing public widgets and actual image drop events. Actual ONNX CPU inference, native result tables and written JSON exports provide the observations. It is QA instrumentation, not a component of the release application.

The helper requires application version **2.2.0**, explicit absolute fixture/report/data paths and a fresh report destination. An existing report is never replaced. All preferences, temporary inputs, exports and evidence live in a new QA directory; the application package tree is read-only.

Compile locally with the same Qt 6.8.3 ABI used by the package:

```bash
PKG_CONFIG_PATH=/home/misaka/Qt/6.8.3/gcc_64/lib/pkgconfig g++ -std=c++17 -fPIC -shared gui_qa_preload.cpp -o libgui_qa.so $(PKG_CONFIG_PATH=/home/misaka/Qt/6.8.3/gcc_64/lib/pkgconfig pkg-config --cflags --libs Qt6Widgets) -ldl
```

Run against a read-only extracted final DEB launcher, supplying its genuine bus sample:

```bash
./run_gui_qa.sh /absolute/extracted/opt/VisionStudio/run-installed.sh /tmp/vision-studio-2.2.0-installed-gui-proof /absolute/extracted/opt/VisionStudio/assets/bus.jpg
```

Or pass the production build executable and the repository sample. Source builds without a bundled Qt tree use the SDK libraries above; `VISION_STUDIO_GUI_QA_QT_LIBS` may override the library directory. Pillow in the existing CPU runtime prepares 61 resized copies of the real sample; there is no new dependency. The source application, weights and runtime are never modified.

The helper checks ten scenarios:

1. Entire 3-image queue, while a middle image is selected: 3 exports.
2. Tail from the selected middle image: 2 exports.
3. Only the selected middle image: 1 export.
4. Entire 61-image queue, selected row 51 and original IDs 0/5: 61 exports.
5. Tail from row 51 and original IDs 0/5: 11 exports.
6. Only row 51 and original IDs 0/5: 1 export.
7. ID 32 on the genuine bus fixture: successful zero-match results.
8. Negative ID: inline explanation, Start disabled, no task or export.
9. Out-of-range ID 80 on the real 80-class model: inline explanation, no task or export.
10. Stop a 51-image tail starting at row 11 after at least one accepted result; no later progress growth, completed results remain reviewable, export count equals accepted progress.

For every executed range, submitted count, progress maximum, completed count, native export source paths, actual device, original class IDs, reviewed target table and unprocessed-image export state are checked. Reports bind the actual running executable SHA256. `stop_elapsed_ms` is a conservative end-to-end observation including background save completion, a 250 ms no-growth observation and review; it is not an inference benchmark or raw stop-slot timing. `maximum_gui_heartbeat_gap_ms` covers imports and review as well as detection, so it is not specifically a stop-latency metric.

The shell driver applies offscreen/WebEngine rendering flags only to this isolated QA process. Production launchers and preferences remain unchanged. A prototype success is not final-DEB acceptance; rerun on the exact final package and bind its DEB SHA256 in the release evidence.
