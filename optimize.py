import numpy as np
from util import kl_bern, entropy

# TODO: in order to run with warm start, need to call with orig_probs = prior 
@profile
def update(ordered_feats, ordered_judgments,
        converge_type, orig_probs,
        verbose=False,
        tolerance=0.001,
        log_log_alpha_ratio=45,
        feats_to_update=None,
        ):
    results = []

    probs = orig_probs.copy()

    # TODO: does python copy args (if list, modify in place and need to make a copy?)
    if verbose:
        print(f"Items in batch:")
        for feats, judgment in zip(ordered_feats, ordered_judgments):
            print(f"\t{feats}, {judgment}")
    target_item = False 

    # This is the same as manually updating once first
    error = np.inf
    if converge_type == "symmetric":
        do_converge = True
    # In asymmetric case, only converge for positive examples
    elif converge_type == "asymmetric":
        do_converge = (judgment == True)
    elif converge_type == "none":
        do_converge = False

    num_updates = 0

    # precompute feats_to_update for efficiency
    # contains a list of the unique features in the whole batch
    # (i.e. the ones we want to update)
    # features are represented as indices into probs
    if feats_to_update is None:
        feats_to_update = list(set([item for feats in ordered_feats for item in feats]))
    
    batch_feats_by_feat = [None] * len(feats_to_update)
    batch_other_feats_by_feat = [None] * len(feats_to_update)
    batch_judgments_by_feat = [None] * len(feats_to_update)
   
    for idx, curr_feat in enumerate(feats_to_update):
#            temp_feats = []
#            temp_judgments = []
#            for (j, feats) in zip(ordered_judgments, ordered_feats):
#                temp_feats.append(feats)
#                temp_judgments.append(j)
        temp_feats = [feats for feats in ordered_feats if curr_feat in feats]
        temp_judgments = [j for (j, feats) in zip(ordered_judgments, ordered_feats) if curr_feat in feats]
        # has features of all sequences in batch that contain curr_feat in feats_to_update
        batch_feats_by_feat[idx] = temp_feats 
        batch_judgments_by_feat[idx] = temp_judgments 
        # has features of all sequences in batch that contain curr_feat in feats_to_update *excluding curr_feat*
        batch_other_feats_by_feat[idx] = [[f for f in feats if f != curr_feat] for feats in temp_feats]
        """ 
        # other_feats has shape: b x num_feat (where b is # sequences with the current feature)
        other_feats = np.zeros((len(temp_feats), probs.shape[0]))
        # set to 1 all features that are in temp_feats but != curr_feat
        for seq_idx, feats in enumerate(temp_feats):
            for f in feats:
                if f != curr_feat:
                    other_feats[seq_idx, f] = 1
        batch_other_feats_by_feat[idx] = other_feats
        """ 

    best_error = np.inf
    update_sums = []
    # update if first update *or* have not converged yet
    while (do_converge and error > tolerance) or num_updates == 0:
        if verbose:
            print(f"  Update {num_updates}:")
        if not target_item:
            # update_one_step returns a copy of the new probs
            step_results = update_one_step(probs,
                    ordered_feats, ordered_judgments,
                    feats_to_update, 
                    batch_feats_by_feat, 
                    batch_judgments_by_feat,
                    batch_other_feats_by_feat,
                    log_log_alpha_ratio,
                    verbose=verbose)
            new_probs = step_results["new_probs"]
            # update probs after the step
            update_sums.append(step_results["update_sum"])

            # probs is orig_probs for the first iteration, then the last new_probs
            difference_vector = np.subtract(new_probs, probs)
            probs = new_probs
            error = abs(difference_vector).sum()
            if num_updates > 400:
                if num_updates % 250 == 0:
                    print(f"  Update {num_updates}, error: {error}")

            step_results["error"] = error
            results.append(step_results)
        else:
            raise NotImplementedError()
        
        """
        if do_plot_wandb:
            # TODO: these features will be different than the features in the other heatmap, 
            # these are only plotting the ones updating
            probs_to_plot = self.probs[feats_to_update]
            title = f'Prob vs Feature for Step: {len(ordered_feats)-1}, Update: {num_updates}' 
            str_feats_to_update = [str(f) for f in feats_to_update]
            feature_probs_plot = plot_feature_probs(
                    str_feats_to_update, probs_to_plot, orig_probs[feats_to_update], title=title)

            wandb.log({"intermediate_feature_probs/plot": wandb.Image(feature_probs_plot)})
            plt.close()
            wandb.log({"intermediate_updates/error": error, "intermediate_updates/step": len(ordered_feats)-1, "intermediate_updates/update": num_updates})
        """
        
        num_updates += 1
        
        if error <= best_error:
            best_error = error
        else:
            if verbose:
                print(f"error stopped decreasing after {num_updates} updates")
                print('difference: ', difference_vector.round(3))
#                assert False
            break
        
        if verbose:
#            print(f"Probs after updating: {new_probs}")
            feature_prob_changes = new_probs[feats_to_update]-orig_probs[feats_to_update]
            print(f"Update in probs of features in seq (new-orig): \n{(feature_prob_changes).round(5)}")
            print(f"Num updates: {num_updates}")

    return new_probs, results
    
@profile
def update_one_step(probs,
        ordered_feats, ordered_judgments, 
        feats_to_update, 
        batch_feats_by_feat, 
        batch_judgments_by_feat,
        batch_other_feats_by_feat,
        log_log_alpha_ratio,
        verbose=False): # was originally called update

    clip_val = np.inf
#        clip_val = 30 

    new_probs = probs.copy()
    
    # curr_feat is feat to update
    if verbose:
        print(f"Features to update: {feats_to_update}")
    for idx, curr_feat in enumerate(feats_to_update):
        this_prob = probs[curr_feat]
 
        featurized_seqs = batch_feats_by_feat[idx]
        other_feats = batch_other_feats_by_feat[idx]

        judgments = batch_judgments_by_feat[idx]

        # TODO: speed up this operation by vectorizing
        log_probs_all_off = np.array([np.log(1-probs[o]).sum() for o in other_feats])
        """
        probs_off = 1 - probs
        log_probs_all_off = np.ma.log(other_feats * probs_off).sum(1)
        """

        update_vector = (judgments * np.exp(np.clip(log_probs_all_off + log_log_alpha_ratio, -np.inf, clip_val)))
        update_sum = update_vector.sum()

        log_score = (
            np.log(this_prob) - np.log(1-this_prob) - update_sum
        )

        # TODO: want a one-sided clip?
        log_score = np.clip(log_score, -clip_val, clip_val)

        np.warnings.filterwarnings('ignore', 'overflow')

        posterior = 1 / (1 + np.exp(-log_score))
        posterior = np.clip(posterior, 1e-5, 1-1e-5)
        new_probs[curr_feat] = posterior

        if verbose:
            new_prob = new_probs[curr_feat]
            change = new_prob-this_prob
            print(f"feat: {curr_feat}, before: {this_prob.round(3)}, after: {(new_prob).round(3)} ({(change).round(3)})")

    # TODO: need to return updates for each feature
    results = {
            "new_probs": new_probs, 
            "update_sum": update_sum,
#                "update_unclipped": update_unclipped,
            "log_p_all_off": np.mean(log_probs_all_off), # TODO: do we want the sum?
            }

    return results


##########################################################
    
# Helper function to get the information gain from observing seq with label (call in get_eig and computing eig for an unobserved train example)
@profile
def get_info_gain(featurized_seq, orig_probs, observed_feats, observed_judgments, observed_feats_unique, converge_type, log_log_alpha_ratio, tolerance, label=True):
    # entropy over features before seeing

    entropy_over_features_before_observing_item = -1 * ((orig_probs * np.log(orig_probs) + (1 - orig_probs) * np.log(1 - orig_probs))).sum()
    assert entropy_over_features_before_observing_item > 0, f"Entropy should be positive. Entropy={entropy_over_features_before_observing_item_positive}. Probs={p.round(decimals=3)}"

    feats_to_update = {*observed_feats_unique, *featurized_seq}

    p, results = update(observed_feats+[featurized_seq], observed_judgments+[label], converge_type, orig_probs, feats_to_update=feats_to_update, verbose=False)
    entropy_over_features_after_observing_item = -1 * ((p * np.log(p) + (1 - p) * np.log(1 - p))).sum()
    assert entropy_over_features_after_observing_item > 0, f"Entropy should be positive. Entropy={entropy_over_features_after_observing_item_positive}. Probs={p.round(decimals=3)}"
        
    return entropy_over_features_before_observing_item - entropy_over_features_after_observing_item

def get_ig_pos(c):
    return get_info_gain(*c, label=True)

def get_ig_neg(c):
    return get_info_gain(*c, label=False) 

def get_kl_pos(c):
    return get_kl(*c, label=True)

def get_kl_neg(c):
    return get_kl(*c, label=False)

@profile
def get_kl(featurized_seq, orig_probs, observed_feats, observed_judgments, observed_feats_unique, converge_type, log_log_alpha_ratio, tolerance, label=True):
    
    feats_to_update = {*observed_feats_unique, *featurized_seq}

    p, results = update(observed_feats+[featurized_seq], observed_judgments+[label], converge_type, orig_probs, feats_to_update=feats_to_update, verbose=False)

    kl = kl_bern(p, orig_probs).sum()

    return kl
