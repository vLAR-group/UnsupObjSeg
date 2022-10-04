def get_kl(z, q_z, p_z, montecarlo):
    if isinstance(q_z, list) or isinstance(q_z, tuple):
        assert len(q_z) == len(p_z)
        kl = []
        for i in range(len(q_z)):
            if montecarlo:
                assert len(q_z) == len(z)
                kl.append(get_mc_kl(z[i], q_z[i], p_z[i]))
            else:
                kl.append(kl_divergence(q_z[i], p_z[i]))
        return kl
    elif montecarlo:
        return get_mc_kl(z, q_z, p_z)
    return kl_divergence(q_z, p_z)

def get_mc_kl(z, q_z, p_z):
    return q_z.log_prob(z) - p_z.log_prob(z)
