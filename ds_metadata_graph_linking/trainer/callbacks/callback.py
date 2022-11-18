class TrainerCallback:
    def on_train_begin(self, config, trainer_state, trainer_control, **kwargs):
        pass

    def on_train_end(self, config, trainer_state, trainer_control, **kwargs):
        pass

    def on_epoch_begin(self, config, trainer_state, trainer_control, **kwargs):
        pass

    def on_epoch_end(self, config, trainer_state, trainer_control, **kwargs):
        pass

    def on_train_step_begin(self, config, trainer_state, trainer_control, **kwargs):
        pass

    def on_train_step_end(self, config, trainer_state, trainer_control, **kwargs):
        pass

    def on_validation_begin(self, config, state, control, **kwargs):
        pass

    def on_validation_step_end(self, config, trainer_state, trainer_control, **kwargs):
        pass


class CallbackHandler(TrainerCallback):
    def __init__(self, callbacks):
        self.callbacks = callbacks

    def on_train_begin(self, config, trainer_state, trainer_control, **kwargs):
        return self.call_event("on_train_begin", config, trainer_state, trainer_control)

    def on_train_end(self, config, trainer_state, trainer_control, **kwargs):
        return self.call_event("on_train_end", config, trainer_state, trainer_control)

    def on_epoch_begin(self, config, trainer_state, trainer_control, **kwargs):
        return self.call_event("on_epoch_begin", config, trainer_state, trainer_control)

    def on_epoch_end(self, config, trainer_state, trainer_control, **kwargs):
        return self.call_event("on_epoch_end", config, trainer_state, trainer_control)

    def on_train_step_begin(self, config, trainer_state, trainer_control, **kwargs):
        return self.call_event("on_train_step_begin", config, trainer_state, trainer_control)

    def on_train_step_end(self, config, trainer_state, trainer_control, **kwargs):
        return self.call_event("on_train_step_end", config, trainer_state, trainer_control)

    def on_validation_begin(self, config, state, control, **kwargs):
        return self.call_event("on_validation_begin", config, state, control)

    def on_validation_step_end(self, config, trainer_state, trainer_control, **kwargs):
        return self.call_event("on_validation_step_end", config, trainer_state, trainer_control)

    def call_event(self, event, config, state, control, **kwargs):
        for callback in self.callbacks:
            getattr(callback, event)(config, state, control, **kwargs)
        return control
