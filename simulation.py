import pygame
from Rocket import Rocket, background_img, screen
from DQN_Agent import DQNAgent

def run_simulation():

    rocket = Rocket()
    agent = DQNAgent()
    clock = pygame.time.Clock()

    episode = 0
    episode_reward = 0

    running = True

    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        state = rocket.get_state()
        action = agent.select_action(state)
        reward = rocket.update(action)

        total_reward = -0.1
        total_reward += rocket.apply_rotation_penalty()
        total_reward += rocket.stabilize_rotation_reward()

        if rocket.vy > 0:
            total_reward += 1
        elif rocket.vy < 0:
            total_reward -= 0.5

        final_reward = total_reward + reward
        episode_reward += final_reward

        agent.store_experience(
            state,
            action,
            final_reward,
            rocket.get_state(),
            rocket.landed or rocket.crashed
        )

        agent.train()

        screen.blit(background_img, (0, 0))
        rocket.draw()
        pygame.display.flip()
        clock.tick(30)

        if rocket.landed or rocket.crashed:

            agent.episode_rewards.append(episode_reward)
            agent.total_episodes += 1

            if rocket.landed:
                agent.success_count += 1

            print("=================================")
            print(f"Episode {episode}")
            print(f"Reward: {episode_reward}")
            print("Success Rate:",
                  agent.success_count / agent.total_episodes * 100)
            print("=================================")

            episode_reward = 0
            episode += 1
            rocket.reset()

            if episode % 50 == 0:
                agent.save_model()

    pygame.quit()

run_simulation()