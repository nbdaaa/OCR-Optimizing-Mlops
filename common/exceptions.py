class ChandraBaseError(Exception):
    pass

class StorageError(ChandraBaseError):
    pass

class GateFailure(ChandraBaseError):
    def __init__(self, gate: str, reason: str):
        self.gate = gate
        self.reason = reason
        super().__init__(f"[{gate}] {reason}")
