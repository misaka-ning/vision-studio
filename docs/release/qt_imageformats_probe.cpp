#include <QGuiApplication>
#include <QImageReader>
#include <QImageWriter>
#include <QTemporaryDir>
#include <QDebug>
int main(int argc,char **argv){
 QGuiApplication app(argc,argv); QTemporaryDir d;
 auto formats=QImageReader::supportedImageFormats(); qInfo()<<formats;
 QImage image(64,48,QImage::Format_RGB32); image.fill(QColor(18,174,160));
 for(auto f:{QByteArray("png"),QByteArray("jpeg"),QByteArray("webp"),QByteArray("tiff")}){
  if(!formats.contains(f)){qCritical()<<"missing"<<f;return 2;}
  const auto path=d.path()+"/test."+QString::fromLatin1(f);
  QImageWriter writer(path,f);if(!writer.write(image)){qCritical()<<writer.errorString();return 3;}
  QImageReader reader(path);const auto decoded=reader.read();if(decoded.isNull()||decoded.size()!=image.size()){qCritical()<<reader.errorString();return 4;}
 }
 return formats.contains("svg")?0:5;
}
