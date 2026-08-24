
class Constants:
    name_in_url = 'referential_task'
    players_per_group = None
    num_rounds = 5

class SessionConfig:
    def __init__(self, **kwargs):
        self.config = kwargs
        
    def get(self, key, default=None):
        return self.config.get(key, default)

class Session:
    def __init__(self, config_dict=None):
        self.config = config_dict or {}
        self.players = []

class Group:
    def __init__(self, shared_grid="[]", target_baskets="[]"):
        self.shared_grid = shared_grid
        self.target_baskets = target_baskets
        self.ai_partial_sequence = "[]"
        self.ai_messages = "[]"
        self.ai_reasoning_log = "[]"
        self.matcher_sequence = "[]"
        
        # Perceptions AI vs AI mode: Director's perceptions of Matcher
        self.ai_director_partner_capable = None
        self.ai_director_partner_helpful = None
        self.ai_director_partner_understood = None
        self.ai_director_partner_adapted = None
        self.ai_director_collaboration_improved = None
        self.ai_director_partner_comment = ""
        self.ai_director_perceptions_raw = ""

        # Perceptions AI vs AI mode: Matcher's perceptions of Director
        self.ai_matcher_partner_capable = None
        self.ai_matcher_partner_helpful = None
        self.ai_matcher_partner_understood = None
        self.ai_matcher_partner_adapted = None
        self.ai_matcher_collaboration_improved = None
        self.ai_matcher_partner_comment = ""
        self.ai_matcher_perceptions_raw = ""

class Participant:
    def __init__(self, role):
        self.vars = {"role": role}

class Player:
    def __init__(self, role="observer", group=None, session=None, round_number=1):
        self.player_role = role
        self.group = group or Group()
        self.session = session or Session()
        self.round_number = round_number
        self.participant = Participant(role)
        
        # Link this player instance into the session's cross-round list
        if self not in self.session.players:
            self.session.players.append(self)

    def field_maybe_none(self, field_name):
        return getattr(self, field_name, None)
        
    def in_all_rounds(self):
        # Filter to players up to the current round number
        return [p for p in self.session.players if p.round_number <= self.round_number]
