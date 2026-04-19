import matplotlib.pyplot as plt

def plot_training(agent):

    plt.figure()
    plt.plot(agent.episode_rewards)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()

    plt.figure()
    plt.plot(agent.losses)
    plt.title("Loss Curve")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.show()