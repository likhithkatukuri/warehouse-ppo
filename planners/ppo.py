from stable_baselines3 import PPO
from planners.custom_cnn import CustomCNN
import os


class PPOTrainer:

    def __init__(self, env):

        self.env = env
        self.model = None

    ##################################################

    def train(
        self,
        total_timesteps=50000,
        save_path="models/ppo_model"
    ):

        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(
                features_dim=256
            ),
            normalize_images=False
        )

        self.model = PPO(
            policy="CnnPolicy",
            env=self.env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=3e-4,
            gamma=0.99,
            n_steps=1024,
            batch_size=64,
            device="auto"
        )

        self.model.learn(
            total_timesteps=total_timesteps
        )

        os.makedirs("models", exist_ok=True)

        self.model.save(save_path)

        print("\nModel Saved Successfully")

    ##################################################

    def load(
        self,
        path="models/ppo_model"
    ):

        self.model = PPO.load(
            path,
            env=self.env,
            device="auto"
        )

        print("Model Loaded Successfully")

    ##################################################

    def predict(self, observation):

        action, _ = self.model.predict(
            observation,
            deterministic=True
        )

        return action