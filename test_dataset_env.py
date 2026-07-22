import random

from environment.warehouse_dataset_env import WarehouseDatasetEnvironment

env = WarehouseDatasetEnvironment(

    dataset_path=r"C:\Users\likhi\Downloads\warehouse_ppo\Dataset\bin"

)

obs, info = env.reset()

print("=" * 50)

print("Observation Shape :", obs.shape)

print("Scan :", info["scan_name"])

print("Start :", info["start"])

print("Goal :", info["goal"])

print("=" * 50)

done = False

while not done:

    action = random.randint(0, 3)

    obs, reward, terminated, truncated, info = env.step(action)

    print(

        f"Action={action}",

        f"Reward={reward}",

        f"Robot={env.robot_position}"

    )

    done = terminated or truncated

print("\nEpisode Finished")