#include "modelyaml.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QRegularExpression>
#include <QSaveFile>
#include <QSet>
#include <QTemporaryFile>
#include <cmath>
#include <filesystem>
#include <limits>

namespace vision
{
namespace
{
constexpr qsizetype maximumBytes = 256 * 1024;
constexpr int maximumLayers = 2000;
constexpr int maximumValues = 20000;
constexpr int maximumDepth = 12;

bool integer(const QJsonValue &value, double minimum, double maximum)
{
    const double number = value.toDouble(std::numeric_limits<double>::quiet_NaN());
    return value.isDouble() && std::isfinite(number) && number == std::floor(number) &&
           number >= minimum && number <= maximum;
}

bool positive(const QJsonValue &value)
{
    const double number = value.toDouble();
    return value.isDouble() && std::isfinite(number) && number > 0 && number <= 1e9;
}

bool primitive(const QJsonValue &value, int depth, int &count)
{
    if (depth > maximumDepth || ++count > maximumValues || value.isUndefined()) return false;
    if (value.isNull() || value.isBool()) return true;
    if (value.isDouble()) return std::isfinite(value.toDouble()) && std::abs(value.toDouble()) <= 1e12;
    if (value.isString()) return value.toString().size() <= 1024;
    if (value.isArray())
    {
        const auto array = value.toArray();
        if (array.size() > maximumLayers) return false;
        for (const auto &item : array)
            if (!primitive(item, depth + 1, count)) return false;
        return true;
    }
    if (value.isObject())
    {
        const auto object = value.toObject();
        if (object.size() > 128) return false;
        for (auto it = object.begin(); it != object.end(); ++it)
            if (it.key().size() > 128 || !primitive(it.value(), depth + 1, count)) return false;
        return true;
    }
    return false;
}

QString jsonScalar(const QJsonValue &value)
{
    const QByteArray encoded = QJsonDocument(QJsonArray{value}).toJson(QJsonDocument::Compact);
    return QString::fromUtf8(encoded.mid(1, encoded.size() - 2));
}

// JSON scalar/flow syntax is a subset of YAML 1.2. All user-supplied strings
// remain quoted, so tags, anchors, newlines and Python expressions stay data.
QString flow(const QJsonValue &value)
{
    if (value.isArray())
    {
        QStringList items;
        for (const auto &item : value.toArray()) items.append(flow(item));
        return '[' + items.join(", ") + ']';
    }
    if (value.isObject())
    {
        QStringList items;
        const auto object = value.toObject();
        for (auto it = object.begin(); it != object.end(); ++it)
            items.append(jsonScalar(it.key()) + ": " + flow(it.value()));
        return '{' + items.join(", ") + '}';
    }
    return jsonScalar(value);
}

bool sourceIndex(const QJsonValue &value, int layer)
{
    if (!integer(value, -maximumLayers, maximumLayers)) return false;
    const int index = value.toInt();
    return index == -1 || (index >= 0 ? index < layer : layer + index >= 0);
}

bool layerDefinition(const QJsonValue &value, int index)
{
    if (!value.isArray()) return false;
    const auto row = value.toArray();
    if (row.size() != 4 || !integer(row[1], 1, 10000) || !row[2].isString() || !row[3].isArray() ||
        row[3].toArray().size() > 64) return false;
    static const QRegularExpression module(QStringLiteral("^[A-Za-z_][A-Za-z0-9_.]{0,159}$"));
    if (!module.match(row[2].toString()).hasMatch()) return false;
    if (row[0].isArray())
    {
        const auto from = row[0].toArray();
        if (from.isEmpty() || from.size() > 64) return false;
        for (const auto &item : from)
            if (!sourceIndex(item, index)) return false;
        return true;
    }
    return sourceIndex(row[0], index);
}
}

ModelYamlDocument modelYamlDocument(const QJsonValue &architecture)
{
    ModelYamlDocument result;
    const auto fail = [&result](const QString &message) {
        result.notice = message; return result;
    };
    if (!architecture.isObject())
        return fail(QStringLiteral("此模型未包含可安全恢复的 YOLO 架构配置。可继续查看结构图、层级树或参数表。"));
    int count = 0;
    if (!primitive(architecture, 0, count) || QJsonDocument(architecture.toObject()).toJson(QJsonDocument::Compact).size() > maximumBytes)
        return fail(QStringLiteral("架构元信息超过大小、深度或条目限制，未生成不完整 YAML。"));
    const auto object = architecture.toObject();
    const QStringList fields = {"nc", "depth_multiple", "width_multiple", "scales", "scale", "max_channels",
        "anchors", "ch", "channels", "activation", "inplace", "kpt_shape", "reg_max", "end2end", "backbone", "head"};
    QSet<QString> allowed(fields.begin(), fields.end());
    allowed.insert("yaml_file"); // A saved name is provenance, never opened or used to guess a blueprint.
    for (auto it = object.begin(); it != object.end(); ++it)
        if (!allowed.contains(it.key()))
            return fail(QStringLiteral("架构包含暂不支持的字段，未猜测或省略生成配置：%1").arg(it.key().left(128)));
    if (!integer(object.value("nc"), 1, 100000) || !object.value("backbone").isArray() || !object.value("head").isArray())
        return fail(QStringLiteral("架构缺少有效的 nc、backbone 或 head，无法生成 YOLO YAML。"));
    const auto backbone = object.value("backbone").toArray(), head = object.value("head").toArray();
    if (backbone.isEmpty() || head.isEmpty() || backbone.size() + head.size() > maximumLayers)
        return fail(QStringLiteral("架构层数为空或超过 2000 层限制。"));
    for (const QString &key : {QStringLiteral("depth_multiple"), QStringLiteral("width_multiple")})
        if (object.contains(key) && !positive(object.value(key)))
            return fail(QStringLiteral("架构缩放参数无效：%1").arg(key));
    for (const QString &key : {QStringLiteral("ch"), QStringLiteral("channels"), QStringLiteral("max_channels"), QStringLiteral("reg_max")})
        if (object.contains(key) && !integer(object.value(key), 1, 1000000))
            return fail(QStringLiteral("架构整数参数无效：%1").arg(key));
    if ((object.contains("activation") && !object.value("activation").isString()) ||
        (object.contains("yaml_file") && !object.value("yaml_file").isString()))
        return fail(QStringLiteral("架构字符串字段类型无效。"));
    for (const auto &key : {QStringLiteral("inplace"), QStringLiteral("end2end")})
        if (object.contains(key) && !object.value(key).isBool()) return fail(QStringLiteral("架构布尔参数无效：%1").arg(key));
    if (object.contains("kpt_shape"))
    {
        const auto shape = object.value("kpt_shape").toArray();
        if (!object.value("kpt_shape").isArray() || shape.size() != 2 || !integer(shape[0], 1, 100000) || !integer(shape[1], 2, 3))
            return fail(QStringLiteral("关键点形状参数无效。"));
    }
    if (object.contains("scales"))
    {
        const auto scales = object.value("scales").toObject();
        if (!object.value("scales").isObject() || scales.isEmpty() || scales.size() > 32)
            return fail(QStringLiteral("架构 scales 参数无效。"));
        for (auto it = scales.begin(); it != scales.end(); ++it)
        {
            const auto row = it.value().toArray();
            if (it.key().isEmpty() || it.key().size() > 64 || !it.value().isArray() || row.size() != 3 ||
                !positive(row[0]) || !positive(row[1]) || !integer(row[2], 1, 1000000))
                return fail(QStringLiteral("架构 scales 条目无效。"));
        }
        if (object.contains("scale") && (!object.value("scale").isString() || !scales.contains(object.value("scale").toString())))
            return fail(QStringLiteral("架构选中的 scale 不在保存的 scales 中。"));
    }
    else if (object.contains("scale") && !object.value("scale").isString())
        return fail(QStringLiteral("架构 scale 类型无效。"));
    if (object.contains("anchors"))
    {
        const auto value = object.value("anchors");
        if (value.isArray())
        {
            const auto groups = value.toArray();
            if (groups.isEmpty() || groups.size() > 16) return fail(QStringLiteral("anchors 分组无效。"));
            for (const auto &group : groups)
            {
                const auto pairs = group.toArray();
                if (!group.isArray() || pairs.isEmpty() || pairs.size() > 128 || pairs.size() % 2)
                    return fail(QStringLiteral("anchors 必须由有效宽高对组成。"));
                for (const auto &number : pairs) if (!positive(number)) return fail(QStringLiteral("anchors 尺寸无效。"));
            }
        }
        else if (!integer(value, 1, 64)) return fail(QStringLiteral("anchors 类型无效。"));
    }
    int layer = 0;
    for (const auto &section : {backbone, head})
        for (const auto &row : section)
            if (!layerDefinition(row, layer++))
                return fail(QStringLiteral("第 %1 层不是有效的 [from, repeats, module, args] 定义。未输出猜测配置。").arg(layer - 1));
    QString text = QStringLiteral("# Generated from the architecture dictionary stored in this model.\n"
                                  "# Not the original YAML file; original comments and formatting are unavailable.\n"
                                  "# Blueprint repeats/channels may be scaled by depth/width at model construction.\n\n");
    for (const auto &key : fields)
    {
        if (!object.contains(key) || key == "backbone" || key == "head") continue;
        text += key + ": " + flow(object.value(key)) + '\n';
    }
    for (const auto &key : {QStringLiteral("backbone"), QStringLiteral("head")})
    {
        text += '\n' + key + ":\n  # [from, repeats, module, args]\n";
        for (const auto &value : object.value(key).toArray())
        {
            const auto row = value.toArray();
            text += "  - [" + flow(row[0]) + ", " + flow(row[1]) + ", " + flow(row[2]) + ", " + flow(row[3]) + "]\n";
        }
    }
    if (text.toUtf8().size() > maximumBytes) return fail(QStringLiteral("生成的架构 YAML 超过 256 KiB 展示限制。"));
    result.available = true;
    result.text = text;
    result.layers = layer;
    result.notice = QStringLiteral("从模型保存的架构字典生成等价配置；不是原始 YAML，原注释不可恢复。重复数与通道可能受深度／宽度缩放影响，不代表实际算子总数。");
    return result;
}

bool saveModelYaml(const QString &destination, const QString &sourceModel, const ModelYamlDocument &document, QString *error)
{
    const auto fail = [error](const QString &message) { if (error) *error = message; return false; };
    if (error) error->clear();
    if (!document.available || document.text.isEmpty() || document.text.toUtf8().size() > maximumBytes)
        return fail(QStringLiteral("当前模型没有可导出的架构 YAML。"));
    const QFileInfo target(destination), source(sourceModel);
    if (destination.isEmpty() || target.fileName().isEmpty() || !target.dir().exists())
        return fail(QStringLiteral("请选择已存在的可写文件夹与新文件名。"));
    if (target.exists() || target.isSymLink() || target.absoluteFilePath() == source.absoluteFilePath())
        return fail(QStringLiteral("目标已存在或与源模型相同；请使用新文件名，原文件不会被覆盖。"));
    if (target.suffix().compare("yaml", Qt::CaseInsensitive) && target.suffix().compare("yml", Qt::CaseInsensitive))
        return fail(QStringLiteral("架构文件请使用 .yaml 或 .yml 扩展名。"));
    const auto bytes = document.text.toUtf8();
    QString stagedPath;
    {
        // QTemporaryFile may retain its native handle across close(). Destroy
        // it before publishing the hard link, including on Windows.
        QTemporaryFile temporary(target.dir().absoluteFilePath(".vision-studio-model-yaml-XXXXXX"));
        if (!temporary.open() || temporary.write(bytes) != bytes.size() || !temporary.flush())
            return fail(QStringLiteral("无法在目标目录写入安全临时文件。"));
        stagedPath = temporary.fileName();
        temporary.setAutoRemove(false);
    }
    std::error_code code;
    std::filesystem::create_hard_link(std::filesystem::u8path(stagedPath.toUtf8().toStdString()),
                                     std::filesystem::u8path(target.absoluteFilePath().toUtf8().toStdString()), code);
    QFile::remove(stagedPath);
    if (code) return fail(QStringLiteral("无法原子创建新文件（目标可能已存在，或目录不支持安全另存）。原文件未覆盖。"));
    return true;
}

QString modelYamlExtractionScript()
{
    return QString::fromUtf8(R"JS((() => {
        const missing = message => JSON.stringify({schema: 1, available: false, message});
        try {
            const model = window.__view__ && window.__view__.model;
            if (!model) return missing('模型尚未解析。');
            if (typeof model.format !== 'string' || !/PyTorch|TorchScript/.test(model.format))
                return missing('此文件未提供可安全恢复的 YOLO 架构配置。ONNX 运算图不能可靠还原原始架构 YAML。');
            let values = 0, scanned = 0;
            const active = new WeakSet();
            const decode = (value, depth = 0) => {
                if (depth > 12 || ++values > 20000) throw new Error('架构元信息超过深度或条目限制');
                if (value === null || typeof value === 'boolean') return value;
                if (typeof value === 'number' && Number.isFinite(value) && Math.abs(value) <= 1e12) return value;
                if (typeof value === 'string' && value.length <= 1024) return value;
                if (!value || typeof value !== 'object' || active.has(value)) throw new Error('架构含非原始值或循环引用');
                active.add(value);
                try {
                    if (Array.isArray(value)) {
                        if (value.length > 2000) throw new Error('架构数组超过限制');
                        return value.map(item => decode(item, depth + 1));
                    }
                    const descriptor = value.type;
                    const type = descriptor && (typeof descriptor.identifier === 'string' ? descriptor.identifier :
                        typeof descriptor.name === 'string' ? descriptor.name : '');
                    if (!['builtins.dict', 'collections.OrderedDict', 'builtins.list', 'builtins.tuple'].includes(type))
                        throw new Error('架构含不支持的对象类型');
                    if (!Array.isArray(value.inputs) || value.inputs.length > 2000) throw new Error('架构容器无效或超限');
                    const list = type === 'builtins.list' || type === 'builtins.tuple';
                    if (!list && value.inputs.length > 128) throw new Error('架构字典超过字段限制');
                    const output = list ? [] : Object.create(null);
                    for (let index = 0; index < value.inputs.length; ++index) {
                        const arg = value.inputs[index];
                        if (!arg || typeof arg.name !== 'string' || arg.name.length > 128 ||
                            !['attribute', 'string[]', 'object', 'object[]'].includes(arg.type)) throw new Error('架构字段类型无效');
                        if (list ? arg.name !== String(index) : Object.hasOwn(output, arg.name)) throw new Error('架构索引或重复字段无效');
                        const item = decode(arg.value, depth + 1);
                        if (list) output.push(item); else output[arg.name] = item;
                    }
                    return output;
                } finally { active.delete(value); }
            };
            const seen = new WeakSet();
            const find = (node, depth = 0) => {
                if (!node || typeof node !== 'object' || seen.has(node)) return null;
                if (depth > 8 || ++scanned > 256) throw new Error('模型元信息查找超过限制');
                seen.add(node);
                const inputs = Array.isArray(node.inputs) ? node.inputs : [];
                if (inputs.length > 256) throw new Error('模型元信息字段超过限制');
                const yaml = inputs.filter(arg => arg && arg.name === 'yaml');
                if (yaml.length > 1) throw new Error('模型含重复架构字段');
                if (yaml.length) {
                    if (yaml[0].type !== 'object') throw new Error('架构不是受支持的原始字典');
                    return decode(yaml[0].value);
                }
                // Checkpoint envelopes only: never scan weights/tensor data or module tensors.
                for (const key of ['model', 'ema']) {
                    const arg = inputs.find(value => value && value.name === key && value.type === 'object');
                    if (arg) { const value = find(arg.value, depth + 1); if (value) return value; }
                }
                return null;
            };
            let graphs = Array.isArray(model.modules) ? model.modules : [];
            if (graphs.length > 32) throw new Error('模型图数量超过查找限制');
            graphs = [...graphs].sort((a, b) => (a.name === 'model' ? -1 : a.name === 'ema' ? 0 : 1) - (b.name === 'model' ? -1 : b.name === 'ema' ? 0 : 1));
            for (const graph of graphs) {
                const nodes = Array.isArray(graph.nodes) ? graph.nodes : [];
                if (nodes.length > 256) throw new Error('模型根节点超过查找限制');
                for (const node of nodes) {
                    const architecture = find(node);
                    if (architecture) {
                        const result = {schema: 1, available: true, architecture};
                        const encoded = JSON.stringify(result);
                        if (new TextEncoder().encode(encoded).length > 256 * 1024) throw new Error('架构元信息超过 256 KiB 限制');
                        return encoded;
                    }
                }
            }
            return missing('此模型未包含可恢复的 YOLO 架构字典。仅参数权重、TorchScript 或运算图不能可靠还原原始 YAML。');
        } catch (error) { return missing('无法安全读取架构：' + String(error && error.message || error).slice(0, 256)); }
    })())JS");
}
}
