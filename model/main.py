from hockey.hockey_env import HockeyEnv_BasicOpponent
from .lightning import SoftActorCritic

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

def main():
    env = HockeyEnv_BasicOpponent(weak_opponent=True)
    model = SoftActorCritic(env)

    logger = TensorBoardLogger(
        save_dir="tb_logs",
        name="soft_actor_critic"
    )

    trainer = pl.Trainer(
        accelerator="cpu",
        log_every_n_steps=1,
        check_val_every_n_epoch=1_000,
        max_epochs=500_000,
        logger=logger
    )

    trainer.fit(model)

if __name__ == "__main__":
    main()