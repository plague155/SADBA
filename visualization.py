
def vis_agg_weight(helper, names, weights, epoch, vis, adversarial_name_keys):
    print(names)
    print(adversarial_name_keys)
    for i in range(0, len(names)):
        _name = names[i]
        _weight = weights[i]
        _is_poison = False
        if _name in adversarial_name_keys:
            _is_poison = True
        helper.target_model.weight_vis(vis=vis, epoch=epoch, weight=_weight, eid=helper.params['environment_name'],
                                       name=_name, is_poisoned=_is_poison)


def vis_fg_alpha(helper, names, alphas, epoch, vis, adversarial_name_keys):
    print(names)
    print(adversarial_name_keys)
    for i in range(0, len(names)):
        _name = names[i]
        _alpha = alphas[i]
        _is_poison = False
        if _name in adversarial_name_keys:
            _is_poison = True
        helper.target_model.alpha_vis(vis=vis, epoch=epoch, alpha=_alpha, eid=helper.params['environment_name'],
                                      name=_name, is_poisoned=_is_poison)
