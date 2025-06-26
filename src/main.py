from training.training_config import TrainingConfig
from training.training_loop import TrainingLoop
from training.training_setup import TrainingSetup


def main(inference_mode: bool = False):
    training_loop: TrainingLoop = (
        TrainingSetup(training_config=TrainingConfig())
        .create_modules()
        .build_training_loop()
    )

    if not inference_mode:
        training_loop.run()

    training_loop.test(inference_mode)

    training_loop.finish()


if __name__ == "__main__":
    main()
