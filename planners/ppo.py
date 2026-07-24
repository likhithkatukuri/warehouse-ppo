from stable_baselines3 import PPO
from planners.custom_cnn import CustomCNN
import os


class PPOTrainer:

    def __init__(self, env):
        self.env = env
        self.model = None

    ##################################################
    # Train PPO
    ##################################################

    def train(
        self,
        total_timesteps=500000,
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

            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,

            gamma=0.99,
            gae_lambda=0.95,

            clip_range=0.2,

            ent_coef=0.01,
            vf_coef=0.5,

            max_grad_norm=0.5,

            tensorboard_log="./logs/",

            verbose=1,

            device="auto"
        )

        self.model.learn(
            total_timesteps=total_timesteps,
            progress_bar=True
        )

        os.makedirs("models", exist_ok=True)

        self.model.save(save_path)

        print("\n===================================")
        print("Model Saved Successfully")
        print("===================================")

    ##################################################
    # Load PPO Model
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

        print("\n===================================")
        print("Model Loaded Successfully")
        print("===================================")

    ##################################################
    # Predict Action
    ##################################################

    def predict(self, observation):

        action, _ = self.model.predict(
            observation,
            deterministic=True
        )

        return action