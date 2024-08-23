import numpy as np
from neuroCombat import neuroCombat, neuroCombatFromTraining
import os
import pickle
import sys
import io


# Run ComBat harmonization using neuroCombat with bootstrappingbale
def run_combat_bootstrap(df, batch_var, biological_vars, data_vars, num_boot, clobber):

    # Convert all to lists
    if isinstance(batch_var, str): batch_var = [batch_var]
    if isinstance(biological_vars, str): batch_var = [batch_var]
    if isinstance(data_vars, str): data_vars = [data_vars]

    # Get the list of variables that are continuous
    continuous_vars = df[batch_var + biological_vars].select_dtypes(include=['int', 'float']).columns.tolist()

    # Get the list of variables that are categorical
    categorical_vars = list((set(df[batch_var + biological_vars].columns) - set(continuous_vars)) - set(batch_var))

    # Get the number of rows to extract each time
    num_samp = round(0.632 * np.shape(df)[0])

    # Only run the boostrapping if the file doesn't exist or we we need to overwrite
    if (not os.path.exists(os.path.join(os.getcwd(), 'combat', 'combat_estimates.pickle'))) or clobber == 1:

        # Run ComBat once to get the estimates structure
        data_combat = neuroCombat(dat=np.transpose(df[data_vars].to_numpy()), covars=df[batch_var + biological_vars],
                                  batch_col=batch_var[0], categorical_cols=categorical_vars,
                                  continuous_cols=continuous_vars)
        estimates = data_combat['estimates']

        # Set all the keys to zeros
        for key in estimates.keys():
            if key != 'batches':
                estimates[key] = np.zeros(np.shape(estimates[key]))

        # Run the bootstraps
        counts = np.zeros((np.shape(df)[0], 1))
        for i in range(0, num_boot):
            print(i)
            text_trap = io.StringIO()
            sys.stdout = text_trap
            estimates, counts = run_combat_bootstrap_iteration(df, num_samp, batch_var, biological_vars, data_vars, categorical_vars, continuous_vars, i, num_boot, counts, estimates)
            sys.stdout = sys.__stdout__

        # Average the estimates
        for key in estimates.keys():
            if key != 'batches':
                if np.shape(df)[0] in np.shape(estimates[key]):
                    estimates[key] = np.divide(estimates[key], np.transpose(counts))
                else:
                    estimates[key] = np.divide(estimates[key], num_boot)

        # Write
    #    with open(os.path.join(os.getcwd(), 'combat', 'combat_estimates.pickle'), 'wb') as handle:
    #        pickle.dump(estimates, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:

        # Read
        with open(os.path.join(os.getcwd(), 'combat', 'combat_estimates.pickle'), 'rb') as handle:
            estimates = pickle.load(handle)

    # Now apply the estimates
    data_combat = neuroCombatFromTraining(dat=np.transpose(df[data_vars].to_numpy()), batch=df[batch_var[0]],
                                          estimates=estimates)

    # Extract the output matrix and transpose
    data_combat = np.transpose(data_combat['data'])

    # Reassign
    for count, var in enumerate(data_vars):
        df[var + '_combatted'] = data_combat[:, count]

    return df


# Single iteration of bootstrap combat
def run_combat_bootstrap_iteration(df, num_samp, batch_var, biological_vars, data_vars, categorical_vars, continuous_vars, ind, num_boot, counts, estimates):

    # Get a list of the unique sites
    batch_u = np.unique(df[batch_var])

    # Take a random sample (63.2%) and make sure all the sites are represented
    b = False
    mult = 0
    df_boot = df.copy()
    while not b:
        df_boot = df.sample(n=num_samp, random_state=ind + (mult * num_boot))
        if len(np.unique(df_boot[batch_var])) == len(batch_u):
            b = True
        else:
            mult = mult + 1

    # Increment the counts since these subjects are selected
    int_ind = [i for i, sub in enumerate(df['subject_id']) if sub in df_boot['subject_id'].to_list()]
    counts[int_ind] = counts[int_ind] + 1

    # Run NeuroCombat
    data_combat = neuroCombat(dat=np.transpose(df_boot[data_vars].to_numpy()),
                              covars=df_boot[batch_var + biological_vars], batch_col=batch_var[0],
                              categorical_cols=categorical_vars, continuous_cols=continuous_vars)
    estimates_boot = data_combat['estimates']

    # Iterate over and add
    for key in estimates.keys():
        if key != 'batches':
            if num_samp in np.shape(estimates_boot[key]):
                estimates[key][:, int_ind] = estimates[key][:, int_ind] + estimates_boot[key]
            else:
                estimates[key] = estimates[key] + estimates_boot[key]

    # Return the estimates
    return estimates, counts





