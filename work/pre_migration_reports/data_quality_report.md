# 数据质量报告 / Data quality report

状态：阻塞（BLOCKED），不能提供实际数据统计。

## 已实际执行

- 从桌面项目目录读取 proposal.txt 和 Proposal Feedback.rtf。
- 创建 data/raw、data/processed、notebooks、src、reports。
- 使用 Python 3.14.6 创建独立 `.venv`，没有修改系统 Python。
- curl 请求 OpenML API 失败：退出码 6，Could not resolve host: www.openml.org。
- pip 安装依赖失败：pypi.org NameResolutionError，无法获取 numpy。
- 在独立环境真实运行 src/data_preparation.py，返回非零退出码；详情见 run_status.json 和 execution_stderr.txt。
- 没有下载到数据、没有生成数据划分、没有拟合 scaler 或模型。

## 尚未验证的事项

284807 条交易、492 个欺诈样本仅为 proposal 中的预期，未被实际验证。维度、列类型、标签数量比例、缺失/无穷值、常量列、重复记录、冲突标签、数值统计均未获得。
来源版本与文件校验值必须下载后计算，当前不填写。依赖安装失败，因此 requirements.txt 不冒充经过验证的版本锁定文件。

## 已实现但尚未在真实数据运行的规则

保留原始 ARFF。明确 Class 为标签，删除 Time。
先检查完整重复记录；在保留的输入特征上统计重复与同输入异标签。
为避免重复输入跨集合泄漏，在分层划分前按原始顺序保留每组输入的首条记录，不根据标签挑选。保存排除行号，可追溯且可复现。不依据异常值范围删除记录，不静默填补缺失。
该规则会改变原始频率与标签比例，对冲突组保留首条具有任意性，不能据此认定其他标签错误。
去重后以 random_state=42、Class 分层划分 80% 开发集与 20% 测试集，保存原始行号索引，并断言跨集合重复输入为零。
测试集封存，不训练、不调参、不报告性能。
StandardScaler 与预留 LogisticRegression 放在同一未拟合 Pipeline 中供后续五折 CV 使用；没有在全数据或完整开发集拟合 scaler，没有标准化标签。

## 局限与下一步

网络 DNS 限制需解除后重跑。桌面路径不在当前写入权限范围，本次文件保存在当前工作目录。
随机划分不能代表时间外推。老师提出的固定成本比和 baseline CV、误差条属于后续工作，尚未执行。
