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
