class TrainerControl:
    should_save = False
    should_log = False
    should_training_stop = False

    def reset(self):
        self.should_log = False
        self.should_save = False
        self.should_training_stop = False

    def continue_training(self):
        return not self.should_training_stop
