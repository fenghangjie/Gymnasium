# Gymnasium Learning Lab

这是一个用来学习 Gymnasium 的小项目。建议按编号顺序运行示例，每个示例对应强化学习里一个重要概念。

## 1. 使用现有环境

本项目使用你已经准备好的 conda 环境：

```powershell
D:\Software\anaconda3\envs\robot\python.exe --version
```

如果需要补装依赖，使用这个环境的 Python：

```powershell
D:\Software\anaconda3\envs\robot\python.exe -m pip install -r requirements.txt
```

## 2. 学习路线

### 第一课：随机策略

运行：

```powershell
D:\Software\anaconda3\envs\robot\python.exe examples/01_random_cartpole.py
```

重点看：

- `env.reset()` 如何开始一局
- `env.step(action)` 返回了什么
- `terminated` 和 `truncated` 分别表示什么
- `action_space.sample()` 为什么只是随机试动作

### 第二课：Q-learning

运行：

```powershell
D:\Software\anaconda3\envs\robot\python.exe examples/02_frozenlake_q_learning.py
```

重点看：

- Q 表如何记录“某个状态下做某个动作”的价值
- `epsilon` 如何平衡探索和利用
- 奖励如何通过 Bellman 更新向前传播

### 第三课：自定义环境

运行：

```powershell
D:\Software\anaconda3\envs\robot\python.exe examples/03_custom_line_world.py
```

重点看：

- 如何继承 `gym.Env`
- 如何定义 `observation_space` 和 `action_space`
- 如何实现 `reset()` 和 `step()`

### 第四课：规则策略

运行：

```powershell
D:\Software\anaconda3\envs\robot\python.exe examples/04_rule_based_cartpole.py
```

重点看：

- `policy` 如何把 observation 转成 action
- 随机策略和规则策略的 reward 差异
- 为什么一个简单规则也可能比随机动作更好

### 动画演示：Q-learning 更新

打开：

```powershell
visualizations/q_learning_update_animation.html
```

重点看：

- Q 值如何从终点附近开始变大
- `best_next_value` 如何把未来价值传回当前状态
- `old_value` 如何逐步靠近 `learned_value`

### 动画演示：强化学习整体流程

打开：

```powershell
visualizations/rl_learning_story_animation.html
```

重点看：

- `observation -> policy -> action -> reward -> Q update` 的完整链路
- `epsilon` 如何控制探索和利用
- `alpha` 和 `gamma` 如何影响 Q 值更新

### HyperFrames 动画：FrozenLake Q-learning

打开：

```powershell
hyperframes/frozenlake-q-learning/index.html
```

这是一个适合渲染成视频的 HyperFrames composition，按场景讲解 `train a small Q-learning agent on FrozenLake`：

- FrozenLake 的 S/F/H/G 地图含义
- Q 表如何表示状态动作价值
- epsilon 如何从探索转向利用
- Q-learning 四行更新公式
- 奖励如何从 G 往前传播

## 3. 推荐理解顺序

先不要急着上深度学习。你可以先问自己三个问题：

1. 当前 `obs` 表示什么？
2. 当前 `action_space` 允许智能体做什么？
3. `reward` 在鼓励智能体学会什么？

等这三个问题变得自然之后，再学习 DQN、PPO、Stable-Baselines3 会顺很多。

## 4. 常见命令

不显示动画，只打印训练结果：

```powershell
D:\Software\anaconda3\envs\robot\python.exe examples/02_frozenlake_q_learning.py
```

检查代码能否编译：

```powershell
D:\Software\anaconda3\envs\robot\python.exe -m compileall examples src
```
