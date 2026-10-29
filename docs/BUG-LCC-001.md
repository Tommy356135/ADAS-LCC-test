# BUG-LCC-001：LCC 在曲线车道上的横向跟踪误差超限

## 1. 基本信息

- 缺陷编号：BUG-LCC-001
- 缺陷标题：LCC 在曲线车道上的横向跟踪误差超限
- 缺陷状态：Open / 待分析
- 优先级：P2（学习项目暂定，待评审）
- 缺陷类型：功能 / 性能
- 测试用例：TC-LCC-CURVE-001
- 关联需求：LCC-REQ-CURVE-001
- 正式执行编号：LCC-EXEC-003
- 实际执行记录：LCC-EXEC-003_20261029T085523_828404Z
- 被测对象：当前冻结版本的 LCCController

## 2. 缺陷描述

在 HighwayEnv 的正弦曲线车道场景中，LCC 控制器未满足预先冻结的
横向跟踪误差限值和最终稳定性要求。

本轮有效执行记录显示，最大绝对横向误差为 2.6398 m，
最终横向误差绝对值为 2.1099 m。
REQ-CURVE-01 和 REQ-CURVE-02 均未满足。

车辆在本次仿真中没有发生车道外状态或碰撞。
因此当前记录支持“曲线跟踪性能要求不满足”的结论，
不支持“本轮已发生车道偏离或碰撞”的结论。

## 3. 测试配置

- 环境：HighwayEnv lane-keeping-v0
- 车道：SineLane
- 随机种子：42
- 初始纵向位置：5.0 m
- 初始横向偏移：0.0 m
- 初始航向：与车道初始切线方向一致
- 初始速度：以运行记录中的实测值为准
- 执行步数：60
- state_noise：0.0
- derivative_noise：0.0
- 控制器参数：以本次 JSON 汇总中的记录为准

## 4. 复现步骤

1. 激活 adas-test 环境并进入项目根目录。
2. 运行 tests/test_lcc_curve_requirement.py。
3. 确认环境为 lane-keeping-v0，车道为 SineLane。
4. 确认测试运行 60 步且噪声关闭。
5. 查看本次运行生成的 JSON 汇总和原始 CSV。
6. 检查最大绝对横向误差以及最后 10 个样本的横向误差。

## 5. 预期结果

- REQ-CURVE-01：最大绝对横向误差不超过 1.00 m。
- REQ-CURVE-02：最后连续 10 个样本的绝对横向误差均不超过 0.25 m。
- REQ-CURVE-03：所有评价样本均在车道内，且无车辆碰撞。
- REQ-CURVE-04：观测及评价数据有限，控制输出合法。

## 6. 实际结果

- 最大绝对横向误差：2.6397835915 m。
- 最终横向误差：+2.1099077179 m。
- 最终横向误差绝对值：2.1099077179 m。
- 车道外样本数：0。
- 碰撞样本数：0。
- 非法控制输出样本数：0。
- REQ-CURVE-01：FAIL。
- REQ-CURVE-02：FAIL。
- REQ-CURVE-03：PASS。
- REQ-CURVE-04：PASS。
- 正式用例结论：FAIL。

## 7. 证据路径

- 原始逐步数据：
  outputs/LCC-EXEC-003/raw_logs/
  LCC-EXEC-003_20261029T085523_828404Z.csv

- 汇总 JSON：
  outputs/LCC-EXEC-003/run_summaries/
  LCC-EXEC-003_20261029T085523_828404Z.json

- pytest 控制台日志：
  outputs/LCC-EXEC-003/reports/curve_requirement_retry1_console.log

- JUnit XML：
  outputs/LCC-EXEC-003/reports/curve_requirement_retry1_junit.xml

- HTML 报告：
  outputs/LCC-EXEC-003/reports/curve_requirement_retry1_report.html

- 初次环境异常记录：
  outputs/LCC-EXEC-003/reports/curve_requirement_env_error_console.log
  outputs/LCC-EXEC-003/reports/curve_requirement_env_error_junit.xml
  outputs/LCC-EXEC-003/reports/curve_requirement_env_error_report.html

## 8. 初步分析与后续行动

初步结论：
当前冻结版本在规定曲线场景下未满足横向跟踪性能要求。
单次仿真不足以证明缺陷在所有场景下都会出现；
本缺陷仅描述上述配置下的可复现测试结果。

后续行动：
1. 将缺陷提交给被测软件负责人进行原因分析。
2. 核对曲线几何、横向误差定义、控制器输入和输出记录。
3. 如需修改软件，由开发方提供单独标识的修复版本及变更说明。
4. 测试方使用相同冻结场景及判据重新执行本用例。
5. 执行相关直线回归用例，检查是否引入回归问题。
6. 所有回归证据齐全且相关判据通过后，再评审是否关闭缺陷。

当前状态：Open。尚无修复版本，本缺陷不得关闭。