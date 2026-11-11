# 消融配置

这些 YAML 通过 `_base_` 继承正式 Stage A 配置，只覆盖被检验的组件。

| 配置 | 含义 |
|---|---|
| `a2_direction_only.yaml` | Vanilla DDPM + 方向条件；无目标置信门控和专用损失 |
| `a3_direction_prior.yaml` | A2 + 目标置信监督与软门控 |
| `a5_no_target_weight.yaml` | 完整模型移除目标加权扩散损失 |
| `a6_no_direction_loss.yaml` | 完整模型移除方向/极性损失 |

完整 Stage A 配置本身对应 A4；Stage A 与 Stage B 的比较对应 A8。DDIM 10/20/50 步直接在冻结 checkpoint 上修改配置的 `diffusion.sampling_steps` 后评价，不重新训练。
