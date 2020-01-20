class BanditAlgorithm(object):
    """A bandit algorithm must be able to do two basic operations.
    1. Choose an action given a context.
    2. Update its internal model given a triple (context, played action, reward).
    """

    def action(self, context):
        pass

    def update(self, context, action, reward):
        pass
