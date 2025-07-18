from cross_env import CrosswalkEnv

def main():
    env = CrosswalkEnv()
    for i in range(5):
        obs, info = env.reset()
        terminated = False
        while not terminated:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
    env.close()

if __name__ == "__main__":
    main()
