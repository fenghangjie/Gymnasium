"""Lesson 1: interact with a Gymnasium environment using a random policy."""

import gymnasium as gym


def main() -> None:
    env = gym.make("CartPole-v1")

    observation, info = env.reset(seed=100)
    total_reward = 0.0
    print()
    """ceshi"""
    print("Observation space:", env.observation_space)
    '''Observation space:
                        Box([-4.8 -inf -0.41887903 -inf],
                            [ 4.8  inf  0.41887903  inf],
                             (4,),
                            float32)
                            环境每次给你 4 个 float32 数字；
                            第 1 个大概在 -4.8 到 4.8；
                            第 2 个速度没有固定边界；
                            第 3 个角度大概在 -24° 到 24°'''
    '''[小车位置, 小车速度, 杆子角度, 杆子角速度] '''
    print("Action space:", env.action_space)
    print("Initial observation:", observation)
    '''Initial observation: [ 0.0273956  -0.00611216  0.03585979  0.0197368 ]'''

    print("Initial info:", info)
    print()

    for step in range(1, 501):
        action = env.action_space.sample() #随机策略
        # if observation[2] < 0:
        #     action = 0
        # else:
        #     action = 1    杆子往哪边倾斜，小车就往哪里移动
        observation, reward, terminated, truncated, info = env.step(action)
        '''
            observation  执行动作后的新观察
            reward       这一步得到的奖励
            terminated   是否自然结束，比如失败或成功
            truncated    是否因为时间限制等外部原因结束
            info         额外信息
            
            
            杆子倒了                      terminated = True
            小车跑出边界                   terminated = True
            坚持到最大步数但没倒            truncated = True
        '''
        total_reward += reward

        print(
            f"step={step:03d} action={action} reward={reward:.1f} "
            f"terminated={terminated} truncated={truncated}"
        )

        if terminated or truncated:
            print(f"\nEpisode finished after {step} steps.")
            print(f"Total reward: {total_reward:.1f}")
            break

    env.close()


if __name__ == "__main__":
    main()

