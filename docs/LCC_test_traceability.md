# LCC 需求—用例追踪矩阵

| 属性 | 内容 |
|---|---|
| 文档编号 | LCC-RTM-001 |
| 版本 | V1.0-DRAFT |
| 日期 | 2026-10-29 |
| 项目 | HighwayEnv LCC 学习型功能测试 |

> 本矩阵用于学习项目追踪，不是主机厂量产需求追踪文件。

## 1. 需求与用例映射

| 需求 ID | 需求简述 | 用例 ID | 验证方式 | 当前状态 |
|---|---|---|---|---|
| LCC-REQ-ENV-001 | 环境正确创建并执行 | LCC-ENV-001 | 环境初始化、reset、step 和完整步数断言 | 已有脚本/测试；需统一记录执行 ID |
| LCC-REQ-DATA-001 | 关键记录为有限数值 | LCC-DATA-001 | 对日志必需字段执行有限值、缺失值和类型检查 | 当前有限值检查已覆盖；缺失字段/空文件负向测试待补 |
| LCC-REQ-ACT-001 | 动作在允许范围内 | LCC-ACT-001 | 环境动作空间 contains 检查及数值范围断言 | 已有 pytest 覆盖 |
| LCC-REQ-FUNC-001 | 正/负初始横向偏差可以被纠正 | LCC-FUN-001A、LCC-FUN-001B | 偏差纠正、最终偏差、稳定尾段检查 | 两方向现有实例通过 |
| LCC-REQ-FUNC-002 | 直线中心起始的无扰动基线行为合理 | LCC-FUN-002 | 中心起始、车道方向对齐、控制量/误差基线检查 | 已有诊断记录；不代表曲线能力 |
| LCC-REQ-CURVE-001 | 曲线车道上的表现可测量并可分析 | LCC-CURVE-001 | 记录横向误差、航向误差、动作和车道状态 | CHARACTERIZATION；缺已批准阈值 |
| LCC-REQ-ROB-001 | 多条件执行可追溯 | LCC-ROB-001 | 配置矩阵、随机种子和结果记录 | 探索性数据已有；比较方法需注意路段距离一致性 |
| LCC-REQ-REPORT-001 | 每次运行可追溯至配置和证据 | LCC-REPORT-001 | 核对执行记录、版本、配置及输出路径 | 待实施统一 run_id 和记录模板 |
| LCC-REQ-REG-001 | 能进行固定用例回归比较 | LCC-REG-001 | 相同 SUT、seed、场景重跑并比较指标 | 待实施 |
| LCC-REQ-DEFECT-001 | 失败可登记并追踪 | LCC-DEFECT-001 | 使用缺陷模板记录复现、证据、责任状态和回归 | 待实施 |

## 2. 当前自动化测试映射

| 当前代码 | 覆盖内容 | 对应需求 | 注意事项 |
|---|---|---|---|
| `tests/test_lcc_function.py::test_lcc_executes_all_steps` | 仿真步数 | LCC-REQ-ENV-001 | 断言依赖当前 fixture 的设定 |
| `tests/test_lcc_function.py::test_lcc_data_is_finite` | 横向误差、航向误差、动作有限值 | LCC-REQ-DATA-001 | 后续仍需补缺失字段/NaN 的工具负向测试 |
| `tests/test_lcc_function.py::test_lcc_actions_are_valid` | 动作空间约束 | LCC-REQ-ACT-001 | 不代表动作控制性能合格 |
| `tests/test_lcc_function.py::test_lcc_reduces_lateral_error` | 最终偏差比初始小 | LCC-REQ-FUNC-001 | 单独使用此断言不充分，需结合最终误差和稳定性 |
| `tests/test_lcc_function.py::test_lcc_final_lateral_error_within_tolerance` | 最终绝对偏差不超过 0.25 m | LCC-REQ-FUNC-001 | 学习项目判据，仅适用当前用例配置 |
| `tests/test_lcc_function.py::test_lcc_settles_and_remains_within_tolerance` | 收敛及末尾样本稳定性 | LCC-REQ-FUNC-001 | 项目自定义阈值，不能当作车规要求 |
| `tests/test_lcc_function.py::test_lcc_vehicle_remains_on_lane` | HighwayEnv 车道范围检查 | LCC-REQ-FUNC-001 | 不能替代车道中心误差评价 |

## 3. 追踪规则

- 每个学习项目需求至少关联一条测试用例。
- 每条正式用例至少关联一条需求或明确标记为探索性用例。
- 每个执行结果要关联用例 ID、SUT 版本、配置、随机种子和输出证据。
- 需求/判据变更必须更新本矩阵并记录变更原因。
- 无验收阈值的曲线探索测试只能标记为 `CHARACTERIZATION`，不可计入正式 PASS 数。
