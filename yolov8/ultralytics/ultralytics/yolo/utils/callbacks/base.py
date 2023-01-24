# Ultralytics YOLO 🚀, GPL-3.0 license
"""
Base callbacks
"""

import wandb


# Trainer callbacks ----------------------------------------------------------------------------------------------------
def on_pretrain_routine_start(trainer):
    pass


def on_pretrain_routine_end(trainer):
    pass


def on_train_start(trainer):
    pass


def on_train_epoch_start(trainer):
    pass


def on_train_batch_start(trainer):
    pass


def optimizer_step(trainer):
    pass


def on_before_zero_grad(trainer):
    pass


def on_train_batch_end(trainer):
    ##
    pass


def on_train_epoch_end(trainer):
    if trainer.wandb:
        trainer.current_epoch += 1
        print(trainer.current_epoch)


def on_fit_epoch_end(trainer, metrics, fitness):
    if trainer.wandb:
        
        keys, vals = list(metrics.keys()), list(metrics.values())
        ### keys : ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'val/box_loss', 'val/cls_loss', 'val/dfl_loss']

        # log best results
        if trainer.best_fitness == fitness:
            best_keys = ['best/epoch', 'best/precision', 'best/recall', 'best/mAP50', 'best/mAP50-95']
            best_results = [trainer.epoch] + vals[0:4]
            for i, name in enumerate(best_keys):
                trainer.wandb.summary[name] = best_results[i]  # log best results in the summary
        trainer.wandb.log(metrics)


def on_model_save(trainer, ckpt, final_epoch, fitness, best_model = False):
    # if trainer.wandb and not final_epoch:
    #         path = trainer.last.parent
    #         model_artifact = wandb.Artifact(
    #             'run_' + wandb.run.id + '_model',
    #             type='model',
    #             metadata={
    #                 'original_url': str(path),
    #                 'epochs_trained': trainer.epoch + 1,
    #                 'total_epochs': trainer.epochs,
    #                 'fitness_score': fitness,
    #                 **ckpt
    #             }
    #         )

    #         model_artifact.add_file(str(path / 'last.pt'))
    #         wandb.log_artifact(
    #             model_artifact,
    #             aliases=['latest', 'last', 'epoch ' + str(trainer.current_epoch), 'best' if best_model else '']
    #         )
    #         print(f"Saving model artifact on epoch {trainer.epoch + 1}")

    pass


def on_train_end(trainer):
    if trainer.wandb:
        path = trainer.last.parent

        model_artifact = wandb.Artifact(
            'run_' + wandb.run.id + '_model',
            type='model',
            metadata={
                'original_url': str(path),
                'epochs_trained': trainer.epoch + 1,
                'total_epochs': trainer.epochs,
                'fitness_score': trainer.fitness,
                'best_fitness_score': trainer.best_fitness
            }
        )

        model_artifact.add_file(str(path / 'best.pt'))
        model_artifact.add_file(str(path / 'last.pt'))
        wandb.log_artifact(
            model_artifact,
            aliases=['latest', 'last', 'epoch ' + str(trainer.current_epoch), 'best']
        )
        print(f"Saving model artifacts")


def on_params_update(trainer):
    pass


def teardown(trainer):
    pass


# Validator callbacks --------------------------------------------------------------------------------------------------
def on_val_start(validator):
    pass


def on_val_batch_start(validator):
    pass


def on_val_batch_end(validator):
    pass


def on_val_end(validator):
    pass


# Predictor callbacks --------------------------------------------------------------------------------------------------
def on_predict_start(predictor):
    pass


def on_predict_batch_start(predictor):
    pass


def on_predict_batch_end(predictor):
    pass


def on_predict_end(predictor):
    pass


# Exporter callbacks ---------------------------------------------------------------------------------------------------
def on_export_start(exporter):
    pass


def on_export_end(exporter):
    pass


default_callbacks = {
    # Run in trainer
    'on_pretrain_routine_start': on_pretrain_routine_start,
    'on_pretrain_routine_end': on_pretrain_routine_end,
    'on_train_start': on_train_start,
    'on_train_epoch_start': on_train_epoch_start,
    'on_train_batch_start': on_train_batch_start,
    'optimizer_step': optimizer_step,
    'on_before_zero_grad': on_before_zero_grad,
    'on_train_batch_end': on_train_batch_end,
    'on_train_epoch_end': on_train_epoch_end,
    'on_fit_epoch_end': on_fit_epoch_end,  # fit = train + val
    'on_model_save': on_model_save,
    'on_train_end': on_train_end,
    'on_params_update': on_params_update,
    'teardown': teardown,

    # Run in validator
    'on_val_start': on_val_start,
    'on_val_batch_start': on_val_batch_start,
    'on_val_batch_end': on_val_batch_end,
    'on_val_end': on_val_end,

    # Run in predictor
    'on_predict_start': on_predict_start,
    'on_predict_batch_start': on_predict_batch_start,
    'on_predict_batch_end': on_predict_batch_end,
    'on_predict_end': on_predict_end,

    # Run in exporter
    'on_export_start': on_export_start,
    'on_export_end': on_export_end}


def add_integration_callbacks(instance):
    # from .clearml import callbacks as clearml_callbacks
    # from .comet import callbacks as comet_callbacks
    # from .hub import callbacks as hub_callbacks
    # from .tensorboard import callbacks as tb_callbacks

    # for x in clearml_callbacks, comet_callbacks, hub_callbacks, tb_callbacks:
    #     for k, v in x.items():
    #         instance.callbacks[k].append(v)  # callback[name].append(func)

    pass