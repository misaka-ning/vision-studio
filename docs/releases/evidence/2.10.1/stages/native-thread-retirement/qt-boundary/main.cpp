#include <QCoreApplication>
#include <QElapsedTimer>
#include <QJsonDocument>
#include <QJsonObject>
#include <QThread>
#include <QTimer>
#include <atomic>
#include <iostream>
#include <memory>
struct State {
    std::atomic<bool> workerDtorEntered{false}, workerDtorDone{false}, tlsDtorEntered{false}, tlsDtorDone{false};
    int workerDelay=0, tlsDelay=0;
};
struct SlowTls {
    std::shared_ptr<State> state;
    ~SlowTls() { state->tlsDtorEntered=true; QThread::msleep(state->tlsDelay); state->tlsDtorDone=true; }
};
void establishTls(const std::shared_ptr<State> &state) { thread_local SlowTls slow{state}; (void)slow.state; }
struct Worker final : QObject {
    std::shared_ptr<State> state;
    explicit Worker(std::shared_ptr<State> value):state(std::move(value)){}
    ~Worker() override { state->workerDtorEntered=true; QThread::msleep(state->workerDelay); state->workerDtorDone=true; }
};
int main(int argc,char **argv) {
    QCoreApplication app(argc,argv);
    const QString mode=argc>1?QString::fromLocal8Bit(argv[1]):QStringLiteral("plain-tls");
    auto state=std::make_shared<State>();
    state->workerDelay=mode=="plain-deferred-delete"?1200:0;
    state->tlsDelay=mode=="plain-deferred-delete"?0:1200;
    QThread *thread=nullptr;
    if(mode=="create-tls") thread=QThread::create([state] { establishTls(state); });
    else {
        thread=new QThread;
        auto *worker=new Worker(state);worker->moveToThread(thread);
        QObject::connect(thread,&QThread::finished,worker,&QObject::deleteLater);
        QObject::connect(thread,&QThread::started,worker,[state] { establishTls(state); QThread::currentThread()->quit(); });
    }
    QTimer beat;beat.setInterval(1);int beats=0;
    QObject::connect(&beat,&QTimer::timeout,&app,[&] { ++beats; });beat.start();
    QObject::connect(thread,&QThread::finished,&app,[&] {
        QJsonObject record{{"mode",mode},{"qt_version",qVersion()},{"finished_signal_delivered",true},
            {"isFinished_before_delete",thread->isFinished()},{"isRunning_before_delete",thread->isRunning()},
            {"wait_zero_before_delete",thread->wait(0)},
            {"worker_destructor_entered_before_delete",state->workerDtorEntered.load()},
            {"worker_destructor_done_before_delete",state->workerDtorDone.load()},
            {"tls_destructor_entered_before_delete",state->tlsDtorEntered.load()},
            {"tls_destructor_done_before_delete",state->tlsDtorDone.load()}};
        const int before=beats;
        QElapsedTimer elapsed;elapsed.start();delete thread;thread=nullptr;
        record.insert("delete_milliseconds",double(elapsed.elapsed()));
        record.insert("heartbeat_events_during_delete",beats-before);
        record.insert("worker_destructor_done_after_delete",state->workerDtorDone.load());
        record.insert("tls_destructor_done_after_delete",state->tlsDtorDone.load());
        const bool reproduced=record["isFinished_before_delete"].toBool() && !record["wait_zero_before_delete"].toBool()
            &&record["delete_milliseconds"].toDouble()>=1000&&beats==before;
        record.insert("finished_did_not_imply_completed_cleanup",reproduced);
        std::cout<<QJsonDocument(record).toJson(QJsonDocument::Compact).constData()<<std::endl;
        app.exit(reproduced?0:2);
    },Qt::QueuedConnection);
    QTimer::singleShot(10,&app,[thread] { thread->start(); });
    QTimer::singleShot(6000,&app,[&] { std::cerr<<"timeout"<<std::endl;app.exit(3); });
    return app.exec();
}
