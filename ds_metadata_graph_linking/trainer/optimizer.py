import torch


class OptimizerManager:
    def __init__(self, config, train_dataloader, model_manager):
        self.config = config
        self.epsilon = config.epsilon
        self.betas = (config.beta1, config.beta2)
        self.learning_rate = config.learning_rate
        self.total_steps = len(train_dataloader) * self.config.epochs
        self.params, self.optimizer = self._build_optimizer(model_manager)

    def load_from_checkpoint(self, optimizer_state):
        self.optimizer.load_state_dict(optimizer_state)

        # optimizer state should be moved to corresponding device
        for optimizer_state in self.optimizer.state.values():
            for k, v in optimizer_state.items():
                if isinstance(v, torch.Tensor):
                    optimizer_state[k] = v.to(self.config.device)

    def step(self):
        self.optimizer.step()

    def state(self):
        return self.optimizer.state_dict()

    def reset(self):
        self.optimizer.zero_grad()

    def _build_optimizer(self, model_manager):
        params = model_manager.parameters()
        optimizer = torch.optim.Adam(params, lr=self.learning_rate, betas=self.betas, eps=self.epsilon)
        return params, optimizer
