
from training.conditional_trainer import ConditionalDDPMTrainer


def main():
    trainer = ConditionalDDPMTrainer(
        dataset_path="dataset/chest_xray/train",
        image_size=128,
        batch_size=4,
        learning_rate=1e-4,
        noise_steps=1000,
        number_of_classes=2,
        ema_decay=0.999,
    )

    trainer.train(total_epochs=1)


if __name__ == "__main__":
    main()
