import numpy as np
import pandas as pd
import tensorflow as tf


def one_hot(df, cols):
    
    """Returns one-hot encoding of DataFrame df including columns in cols."""
    for col in cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(col, axis=1)
    return df

def sample_mushroom_data(file_name,
                         num_contexts,
                         r_noeat=0,
                         r_eat_safe=5,
                         r_eat_poison_bad=-35,
                         r_eat_poison_good=5,
                         prob_poison_bad=0.5):
    """Samples bandit game from Mushroom UCI Dataset.
    Args:
    file_name: Route of file containing the original Mushroom UCI dataset.
    num_contexts: Number of points to sample, i.e. (context, action rewards).
    r_noeat: Reward for not eating a mushroom.
    r_eat_safe: Reward for eating a non-poisonous mushroom.
    r_eat_poison_bad: Reward for eating a poisonous mushroom if harmed.
    r_eat_poison_good: Reward for eating a poisonous mushroom if not harmed.
    prob_poison_bad: Probability of being harmed by eating a poisonous mushroom.
    Returns:
    dataset: Sampled matrix with n rows: (context, eat_reward, no_eat_reward).
    opt_vals: Vector of expected optimal (reward, action) for each context.
    We assume r_eat_safe > r_noeat, and r_eat_poison_good > r_eat_poison_bad.
    ###########
    after one hot encoding
    first column: edibale or not
    second column: poisonious or not
    column 2: : context
    """

    # first two cols of df encode whether mushroom is edible or poisonous
    df = pd.read_csv(file_name)
    df = one_hot(df, df.columns)
    ind = np.random.choice(range(df.shape[0]), num_contexts, replace=True)

    contexts = df.iloc[ind, 2:]
    no_eat_reward = r_noeat * np.ones((num_contexts, 1))
    random_poison = np.random.choice(
      [r_eat_poison_bad, r_eat_poison_good],
      p=[prob_poison_bad, 1 - prob_poison_bad],
      size=num_contexts)
    eat_reward = r_eat_safe * df.iloc[ind, 0]
    eat_reward += np.multiply(random_poison, df.iloc[ind, 1])
    eat_reward = eat_reward.values.reshape((num_contexts, 1))

    # compute optimal expected reward and optimal actions
    exp_eat_poison_reward = r_eat_poison_bad * prob_poison_bad
    exp_eat_poison_reward += r_eat_poison_good * (1 - prob_poison_bad)
    opt_exp_reward = r_eat_safe * df.iloc[ind, 0] + max(
      r_noeat, exp_eat_poison_reward) * df.iloc[ind, 1]

    if r_noeat > exp_eat_poison_reward:
    # actions: no eat = 0 ; eat = 1
        # eat only edible mushrooms
        opt_actions = df.iloc[ind, 0]  # indicator of edible
    else:
        # should always eat (higher expected reward)
        
        opt_actions = np.ones((num_contexts, 1))

    opt_vals = (opt_exp_reward.values, opt_actions.values)

    return np.hstack((contexts.values, no_eat_reward, eat_reward)), opt_vals

def sample_census_data(file_name, num_contexts, shuffle_rows=True):
    """Returns bandit problem dataset based on the UCI census data.
    Args:
    file_name: Route of file containing the Census dataset.
    num_contexts: Number of contexts to sample.
    shuffle_rows: If True, rows from original dataset are shuffled.
    remove_underrepresented: If True, removes arms with very few rewards.
    Returns:
    dataset: Sampled matrix with rows: (context, action rewards).
    opt_vals: Vector of deterministic optimal (reward, action) for each context.
    Preprocessing:
    * drop rows with missing labels
    * convert categorical variables to 1 hot encoding
    Note: this is the processed (not the 'raw') dataset. It contains a subset
    of the raw features and they've all been discretized.

    """
    # 'USCensus1990.data.txt'
    df = pd.read_csv(file_name, sep=",")

    num_actions = 9

    if shuffle_rows:
        df = df.sample(frac=1)
    df = df.iloc[:num_contexts, :]

    # Assuming what the paper calls response variable is the label?
    labels = df['dOccup'].astype('category').cat.codes.as_matrix()
    # In addition to label, also drop the (unique?) key.
    df = df.drop(['dOccup', 'caseid'], axis=1)

    # All columns are categorical. Convert to 1 hot encoding.
    df = pd.get_dummies(df, columns=df.columns)

#     if remove_underrepresented:
#         df, labels = remove_underrepresented_classes(df, labels)
    contexts = df.as_matrix()

    return classification_to_bandit_problem(contexts, labels, num_actions)

def sample_adult_data(file_name, num_contexts, shuffle_rows=True):
    """Returns bandit problem dataset based on the UCI adult data.
    Args:
    file_name: Route of file containing the Adult dataset.
    num_contexts: Number of contexts to sample.
    shuffle_rows: If True, rows from original dataset are shuffled.
    remove_underrepresented: If True, removes arms with very few rewards.
    Returns:
    dataset: Sampled matrix with rows: (context, action rewards).
    opt_vals: Vector of deterministic optimal (reward, action) for each context.
    Preprocessing:
    * drop rows with missing values
    * convert categorical variables to 1 hot encoding
    """
    # adult.data
    df = pd.read_csv(file_name, sep=",", header=None,
                     na_values=[' ?']).dropna()

    num_actions = 14

    if shuffle_rows:
        df = df.sample(frac=1)
    df = df.iloc[:num_contexts, :]

    labels = df[6].astype('category').cat.codes.as_matrix()
    df = df.drop([6], axis=1)

    # Convert categorical variables to 1 hot encoding
    cols_to_transform = [1, 3, 5, 7, 8, 9, 13, 14]
    df = pd.get_dummies(df, columns=cols_to_transform)

#     if remove_underrepresented:
#     df, labels = remove_underrepresented_classes(df, labels)
    contexts = df.as_matrix()

    return classification_to_bandit_problem(contexts, labels, num_actions)

def safe_std(values):
    """Remove zero std values for ones."""
    return np.array([val if val != 0.0 else 1.0 for val in values])

def classification_to_bandit_problem(contexts, labels, num_actions=None):
    
        
    """
      Normalize contexts and encode deterministic rewards.
         returns (third output): optimal reward(always 1) and optimal actions(the labels)
    """
    
    if num_actions is None:
        num_actions = np.max(labels) + 1
    num_contexts = contexts.shape[0]

    # Due to random subsampling in small problems, some features may be constant
    sstd = safe_std(np.std(contexts, axis=0, keepdims=True)[0, :])

    # Normalize features
    contexts = ((contexts - np.mean(contexts, axis=0, keepdims=True)) / sstd)

    # One hot encode labels as rewards
    rewards = np.zeros((num_contexts, num_actions))
    rewards[np.arange(num_contexts), labels] = 1.0

    return contexts, rewards, (np.ones(num_contexts), labels)
