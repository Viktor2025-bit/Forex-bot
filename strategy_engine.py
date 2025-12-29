from enum import Enum, auto

class TradeAction(Enum):
    DO_NOTHING = auto()
    GO_LONG = auto()
    GO_SHORT = auto()
    CLOSE_POSITION = auto()

class BaseStrategy:
    def __init__(self, config: dict):
        self.config = config

    def get_decision(self, prediction: float, position_info: dict) -> TradeAction:
        """
        Based on a prediction and current position, return a trade action.
        """
        raise NotImplementedError("Should implement get_decision()")

class AIStrategy(BaseStrategy):
    def __init__(self, config: dict):
        super().__init__(config)
        risk_config = self.config.get('risk', {})
        self.long_entry_threshold = risk_config.get('min_confidence', 0.55)
        self.short_entry_threshold = 1 - self.long_entry_threshold
        # Using a neutral 0.5 as the exit threshold
        self.long_exit_threshold = 0.5
        self.short_exit_threshold = 0.5

    def get_decision(self, prediction: float, position_info: dict) -> TradeAction:
        has_position = position_info is not None
        
        if has_position:
            pos_side = position_info['side']
            if pos_side == 'long' and prediction < self.long_exit_threshold:
                return TradeAction.CLOSE_POSITION
            elif pos_side == 'short' and prediction > self.short_exit_threshold:
                return TradeAction.CLOSE_POSITION
        else: # No position
            if prediction > self.long_entry_threshold:
                return TradeAction.GO_LONG
            elif prediction < self.short_entry_threshold:
                return TradeAction.GO_SHORT
        
        return TradeAction.DO_NOTHING

class EnsembleStrategy(AIStrategy):
    def get_decision(self, predictions: dict, position_info: dict) -> TradeAction:
        has_position = position_info is not None
        
        # Check for exit first
        if has_position:
            pos_side = position_info['side']
            
            # Use a more sensitive exit: if ANY model signals an exit, close the position
            if pos_side == 'long':
                if any(p < self.long_exit_threshold for p in predictions.values()):
                    return TradeAction.CLOSE_POSITION
            elif pos_side == 'short':
                if any(p > self.short_exit_threshold for p in predictions.values()):
                    return TradeAction.CLOSE_POSITION
        
        # If no position, check for entry
        else:
            # Use a more conservative entry: require ALL models to agree
            if all(p > self.long_entry_threshold for p in predictions.values()):
                return TradeAction.GO_LONG
            elif all(p < self.short_entry_threshold for p in predictions.values()):
                return TradeAction.GO_SHORT

        return TradeAction.DO_NOTHING

# The MovingAverageCrossover strategy would need to be adapted to this new structure
# For now, it is left as-is to focus on the AI strategy refactoring.
# A full refactor would involve making it also return TradeAction enums.
class MovingAverageCrossover:
    pass

if __name__ == "__main__":
    print("This module defines strategy logic classes.")