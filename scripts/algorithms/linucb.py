import numpy as np 
from scripts.core.bandit_algorithm import BanditAlgorithm

class LinUcb(BanditAlgorithm):
    def __init__(self, name, hparams):
        """
        Args:
          name: Name of the algorithm.
          hparams: Hyper-parameters of the algorithm.
        """

        self.name = name
        self.hparams = hparams

        # Gaussian prior for each beta_i: zero mean
        self._alpha = self.hparams.alpha #\lambda

        self.b = [
            np.zeros(self.hparams.context_dim)
            for _ in range(self.hparams.num_actions)
        ]

        self.Ainv = [np.eye(self.hparams.context_dim)
                    for _ in range(self.hparams.num_actions)] 

       

       
        self.data_h = ContextualDataset(hparams.context_dim,
                                        hparams.num_actions,
                                        intercept=True)
        
    def action(self, context):
        """for given context Samples beta's from posterior, and chooses action with maximum X^T \beta_i + \epsilon_i.
        Args:
          context: Context for which the action need to be chosen.
        Returns:
          action: Selected action for the context.
        """

        
        preds = [
            self.Ainv[i].dot(self.b[i]).T.dot(context)
            for i in range(self.hparams.num_actions)
        ]
        
        cbs = [
            self.alpha * np.sqrt(np.linalg.multi_dot([context.T, self.Ainv[i], context]))
            for i in range(self.hparams.num_actions)
            
        ]

        
        # Compute sampled expected values, intercept is last component of beta
        vals = [
            preds[i]+cbs[i]
            for i in range(self.hparams.num_actions)
        ]

        return np.argmax(vals)
    
    def update(self, context, action, reward):
        
        
        self.Ainv[action] -= np.linalg.multi_dot([self.Ainv[action], context.reshape(-1,1), context.reshape(-1,1).T,self.Ainv[action]]) / \
                             (1.0 + np.linalg.multi_dot([context, self.Ainv[action], context]))
        self.b[action] = self.b[action] + np.dot(context.T, reward)

        
    @property
    def alpha(self):
        return self._alpha
        
