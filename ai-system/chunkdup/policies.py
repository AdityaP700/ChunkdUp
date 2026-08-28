from enum import Enum

class Decision(Enum):
    STORE = "store"
    UPDATE = "update"
    IGNORE = "ignore"
    MERGE = "merge"

class BasePolicy:
    def decide(self, existing, new):
        raise NotImplementedError

class EnvironmentPolicy(BasePolicy):
    def decide(self, existing, new):
        if existing is None:
            return Decision.STORE
        if existing.get("value") == new.get("value"):
            return Decision.MERGE
        return Decision.UPDATE

class PreferencePolicy(BasePolicy):
    def decide(self, existing, new):
        if existing is None:
            return Decision.STORE
        if existing.get("value") == new.get("value"):
            return Decision.MERGE
        return Decision.UPDATE

class ToolPolicy(BasePolicy):
    def decide(self, existing, new):
        if existing is None:
            return Decision.STORE
        if existing.get("value") == new.get("value"):
            return Decision.MERGE
        return Decision.UPDATE

class ProjectPolicy(BasePolicy):
    def decide(self, existing, new):
        if existing is None:
            return Decision.STORE

        # If value changed, or language changed, it's an update
        if existing.get("value") == new.get("value"):
            if existing.get("language") == new.get("language"):
                return Decision.MERGE
        return Decision.UPDATE

class EmploymentPolicy(BasePolicy):
    def decide(self, existing, new):
        if existing is None:
            return Decision.STORE
        if existing.get("value") == new.get("value"):
            return Decision.MERGE
        return Decision.UPDATE

class RolePolicy(BasePolicy):
    def decide(self, existing, new):
        if existing is None:
            return Decision.STORE
        if existing.get("value") == new.get("value"):
            return Decision.MERGE
        return Decision.UPDATE

class LearningPolicy(BasePolicy):
    def decide(self, existing, new):
        if existing is None:
            return Decision.STORE
        if existing.get("value") == new.get("value"):
            return Decision.MERGE
        return Decision.UPDATE

class GeneralPolicy(BasePolicy):
    def decide(self, existing, new):
        if existing is None:
            return Decision.STORE
        if existing.get("value") == new.get("value"):
            return Decision.MERGE
        return Decision.UPDATE

class PolicyFactory:
    _default_policy = GeneralPolicy()
    _policies = {
        "environment": EnvironmentPolicy(),
        "project": ProjectPolicy(),
        "tool": ToolPolicy(),
        "preference": PreferencePolicy(),
        "employment": EmploymentPolicy(),
        "role": RolePolicy(),
        "learning": LearningPolicy(),
        "general": GeneralPolicy()
    }

    @classmethod
    def get(cls, memory_type):
        return cls._policies.get(memory_type, cls._default_policy)