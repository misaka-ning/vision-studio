# Qt 6.8.3 finished 信号之后的线程退出边界

只核查此次具体风险。没有修改冻结生产、测试或 Git 索引，没有 GPU 推理。程序使用本机 Qt 6.8.3 的 QCoreApplication 主事件循环；主事件循环线程收到 queued finished 后执行删除，其线程语义与 GUI 主线程相同，但本实验不运行完整 Vision Studio 窗口。

## 原始 Qt 源码

从已保留的 Qtbase 6.8.3 原始归档只读取三个源码成员到本 Git 外目录，来源与选取内容 SHA 见 `qt-source-provenance.json`。此次没有再次计算整个归档 SHA；库存中的原始期望 SHA 与选取文件 SHA 分开记录，不混为重新核验结论。

- `qt-source/qthread.cpp:462–474`：QThread 析构发现 `isInFinish` 时执行无限期 wait。
- `qt-source/qthread.cpp:482–487`：`isFinished()` 返回 `finished || isInFinish`。
- `qt-source/qthread_unix.cpp:399–406`：先置 `isInFinish`、发 QThread::finished，然后才执行 DeferredDelete 与 Qt 线程局部存储清理。
- `qt-source/qthread_unix.cpp:413–442`：后续 cleanup 才设 finished、清 isInFinish 并唤醒等待者。
- `qt-source/qthread.cpp:1272–1277`：QThread::create 的派生析构还会明确 requestInterruption、quit、wait。

所以 finished/isFinished/isRunning 的状态不能证明所有退出清理已完成；在该中间态直接 delete 线程对象仍可能同步等待。

## 本机真实复现

`main.cpp` 和 `CMakeLists.txt` 是单独的 Git 外最小工程，只链接本机 QtCore。plain QThread 两个模式复用生产中的 finished→worker.deleteLater 和 queued 主线程 finished 观察顺序；thread_local/worker 析构分别设置受控 1200ms 延迟。QThread::create 模式独立核对其析构约定。所有模式在真正收到 finished 后记录状态、执行零期限 wait 就绪探针，然后测主线程 delete 与 1ms heartbeat。

- `plain-tls`：进程 exit 0；isFinished=True，isRunning=False，wait(0)=False；delete 耗时 1200ms，期间 heartbeat 0 次。
- `plain-deferred-delete`：进程 exit 0；isFinished=True，isRunning=False，wait(0)=False；delete 耗时 1200ms，期间 heartbeat 0 次。
- `create-tls`：进程 exit 0；isFinished=True，isRunning=False，wait(0)=False；delete 耗时 1200ms，期间 heartbeat 0 次。

原 stdout/stderr、命令对应模式、二进制/本机 QtCore/最小源码 SHA 在 `execution.json`。exit0 表示复现了该阻塞边界，不是应用验收通过。受控慢析构仅证明可达时序，不能宣称真实模型已经产生相同1200ms延迟。

## 当前生产可达链与 PT 结论

`DetectionController` 把 queued QThread::finished 当作 threadFinished_；isShutdownReady 仅要求该标记与结果写入就绪。析构看到 isFinished 为真会在主线程直接 delete QThread。上述 Qt finishing 中间态与这条生产路径相容，因此当前门禁不能保证线程对象析构始终不等待。

PT 的等待必须另作准确区分：InferenceWorker 没有 engine/process 成员或自定义析构。VisionEngine 是 `InferenceWorker::run` 的局部对象，正常业务 InferenceWorker::finished 发出后、run 返回前析构；早停 return 和异常也在工作线程展开。VisionEngine 释放 PtBackend，后者 reset 的 waitForBytesWritten/waitForFinished/terminate/kill 真实可达，但发生于 QThread::finished 之前。当前 worker DeferredDelete 不会重新触发这条 PT 进程等待，不能将 PT reset 作为本次 finished 之后慢析构的真实实例。

## 建议边界

在主线程允许删除 QThread 和关闭窗口之前，须得到线程确已完成退出/清理的异步确认，而不是只看 finished 或 isFinished。可以由独立后台收尾任务执行线程同步与回收，成功后才向拥有者主线程交付就绪；拥有者提前销毁时收尾任务仍须可靠持有对象并完成回收。此处只提出边界，没有实施或认可某个尚未测试的修复。
