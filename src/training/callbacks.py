class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.best_val = None
        self.counter = 0
        self.stop = False

    def step(self, current_val):
        if self.best_val is None or current_val > self.best_val:
            self.best_val = current_val
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
