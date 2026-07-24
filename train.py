from training.trainer import DDPMTrainer


def main():
    trainer = DDPMTrainer(
        dataset_path="dataset/chest_xray/train",
        image_size=128,
        batch_size=4,
        learning_rate=1e-4,
        noise_steps=1000
    )

    trainer.train(epochs=3)


if __name__ == "__main__":
    main()