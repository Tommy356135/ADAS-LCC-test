# ADAS LCC Testing Lab

基于 HighwayEnv 与 pytest 的车道居中控制（Lane Centering Control, LCC）功能测试学习项目。

## 1. 项目简介

本项目围绕智能驾驶测试工程的基础流程展开，重点实践测试需求分析、测试用例设计、自动化执行、运行数据采集、需求判定、缺陷记录和测试证据归档。

项目中的 LCC 控制器仅作为被测对象，不以优化控制器性能为主要目标。

## 2. 技术栈

- Python 3.11
- HighwayEnv 1.12.1
- Gymnasium 1.4.0
- NumPy 2.4.6
- pytest 9.1.1
- pytest-html 4.2.0

## 3. 测试工程内容

- LCC 测试需求、测试计划与需求追踪
- 直线路段横向偏移测试
- 曲线车道跟踪测试
- pytest 自动化执行与断言
- 每个控制周期的原始数据采集
- JSON 指标汇总
- JUnit XML 与 HTML 测试报告
- 缺陷描述、证据关联与状态跟踪

## 4. 当前测试结果

### 直线基线测试

LCC-EXEC-002 的测试执行记录显示，专项测试 18 项通过，整套测试 19 项通过。这是曲线正式测试加入之前的结果，不代表当前所有用例全部通过。

### 曲线车道测试

正式用例：TC-LCC-CURVE-001

- 执行编号：LCC-EXEC-003
- 执行步数：60
- 最大绝对横向误差：2.6398 m
- 最大绝对横向误差限值：1.00 m
- 最终横向误差绝对值：2.1099 m
- 最后连续 10 个样本的误差均须不超过 0.25 m
- 车道外状态样本：0
- 碰撞样本：0

**当前判定：FAIL。**

最大绝对横向误差与最终稳定性要求未满足，相关缺陷记录为 BUG-LCC-001，状态保持 Open。本项目暂不开展修复验证和回归闭环。

## 5. 运行方式

创建 Python 3.11 环境后，执行：

```bash
pip install -r requirements.txt
python -m pytest -v tests/test_lcc_function.py
```

运行正式曲线用例：

```bash
python -m pytest -v tests/test_lcc_curve_requirement.py
```

目前曲线用例会因测试要求未满足而返回非零退出码。这是当前已记录的功能测试结果，不应通过修改阈值来人为获得通过。

## 6. 证据目录

- `docs/`：测试设计、需求追踪与缺陷文档
- `results/LCC-EXEC-001/`：直线基线执行记录
- `results/LCC-EXEC-003/`：曲线测试原始数据、结构化汇总和测试报告

## 7. 项目边界与局限

本项目是基于仿真环境的学习项目，测试阈值是项目级示例，不代表任何整车企业的量产验收标准。仿真结果不能替代真实车辆道路测试、硬件在环测试或安全验证。

当前版本没有完成 BUG-LCC-001 的修复验证与回归关闭。

## 8. 关键词

ADAS Testing, LCC, HighwayEnv, Python, pytest, Test Automation, Test Case Design, Test Evidence, Defect Tracking
